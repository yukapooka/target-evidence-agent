"""Critic: deterministic provenance checks (code), then a critic agent that reviews every claim (LLM).

Objective:
    Check the ranking agent's claims against their sources before results are shown, and lower scores
    that the evidence doesn't support.

Contents:
    run_checks()    RankingResult -> list of CheckResult: PMIDs exist and mention the target (or a drug acting
                    on it); cited records exist; scores are not above what each record supports
    apply_checks()  (C3) drops failed citations, caps scores, then re-applies the citation rule
    critique()      (C4) runs the critic agent on the checked result and applies its verdicts

Rationale:
    - Deterministic checks come first: facts code can verify are never left to LLM judgement.
    - Checks set ceilings, not exact scores: a record shows the most a score can be, while judging literature
      and mechanism strength is left to the critic agent.
    - Abstracts about a drug often name the drug, not the target (dupilumab papers rarely say "IL4R"),
      so a known drug acting on the target counts as a mention.
    - All lookups reuse the Day 1 wrappers, so they are cached and cheap.
"""

import re
import os

from agent.schemas import CheckResult, CriticReport, RankingResult
from agent.tools.expression import SkinExpression, get_skin_expression
from agent.tools.opentargets import TargetAssociation, get_associated_targets, get_known_drugs, get_target_info
from agent.tools.pubmed import check_pmids, fetch_abstracts
from agent.rubric import enforce_citations
from agent.loop import make_client, run_agent, schema_validator
from agent.prompts import CRITIC_PROMPT_VERSION, CRITIC_SYSTEM_PROMPT, critic_user_prompt
from agent.tools.registry import CRITIC_TOOL_NAMES, CRITIC_TOOLS, run_tool

# --- C2a: score ceilings from records ---
# The highest rubric score each record supports (bands from CRITERIA in rubric.py). A claimed score above
# the ceiling fails the check.
# In:  _expression_ceiling(SkinExpression(skin_tissues=[TissueTPM(median_tpm=27.3), ...], skin_rank=28, ...))
# Out: 2   (10-50 TPM, skin not in the top 10 tissues)
# In:  _association_ceiling(TargetAssociation(overall_score=0.66, ...))   -> 3
# In:  _clinical_ceiling("PHASE_2")                                       -> 2

def _expression_ceiling(record: SkinExpression) -> int:
    tpm = max((t.median_tpm for t in record.skin_tissues), default=0.0)
    if tpm > 50 or record.skin_rank <= 10:
        return 3
    if tpm >= 10:
        return 2
    if tpm >= 1:
        return 1
    return 0


def _association_ceiling(record: TargetAssociation) -> int:
    if record.overall_score > 0.6:
        return 3
    if record.overall_score >= 0.3 or max(record.datatype_scores.values(), default=0.0) >= 0.6:
        return 2
    return 1


CLINICAL_CEILING = {"APPROVAL": 3, "PHASE_4": 3, "PHASE_3": 3, "PHASE_2_3": 3,
                    "PHASE_2": 2, "PHASE_1_2": 2, "PHASE_1": 1, "EARLY_PHASE_1": 1}


def _clinical_ceiling(stage: str) -> int:
    return CLINICAL_CEILING.get(stage, 1)

# --- C2b: PMID checks ---
# One check per cited PMID: it must exist in PubMed, and its title or abstract must name the target (symbol or
# alias) or a known drug acting on it in this disease. All PMIDs are checked and fetched in one batched call each.
# _norm() makes matching forgiving: lower case, no hyphens, Greek letters spelled out ("IL-4Rα" -> "il4ra").
# In:  _pmid_checks(result)   (IL4R cites PMID 38743155, a dupilumab trial; FLG cites a fake PMID 99999999)
# Out: [CheckResult(symbol="IL4R", criterion="literature", citation_id="38743155", check="pmid_mentions_target",
#                   passed=True, detail="mentions DUPILUMAB"),
#       CheckResult(symbol="FLG", criterion="literature", citation_id="99999999", check="pmid_exists",
#                   passed=False, detail="PMID not found in PubMed"), ...]

def _norm(text: str) -> str:
    return text.lower().replace("-", "").replace("α", "a").replace("β", "b")


def _target_names(ensembl_id: str, symbol: str, disease_id: str) -> list[str]:
    info = get_target_info(ensembl_id)
    drugs = [d.drug_name for d in get_known_drugs(ensembl_id, disease_id)]
    return [symbol] + info.synonyms + drugs


def _pmid_checks(result: RankingResult) -> list[CheckResult]:
    cited = []
    for c in result.candidates:
        for s in c.scores:
            cited += [(c, s.criterion, x.id) for x in s.citations if x.source == "pmid"]
        cited += [(c, "mechanism_statement", x.id) for x in c.mechanism_citations if x.source == "pmid"]
    if not cited:
        return []

    pmids = sorted({pmid for _, _, pmid in cited})
    exists = check_pmids(pmids)
    found = [p for p in pmids if exists[p]]
    texts = {a.pmid: _norm(a.title + " " + a.abstract) for a in fetch_abstracts(found)} if found else {}

    names = {}
    checks = []
    for c, criterion, pmid in cited:
        base = {"symbol": c.symbol, "criterion": criterion, "citation_id": pmid}
        if not exists[pmid]:
            checks.append(CheckResult(**base, check="pmid_exists", passed=False, detail="PMID not found in PubMed"))
            continue
        if c.symbol not in names:
            names[c.symbol] = _target_names(c.ensembl_id, c.symbol, result.disease_id)
        text = texts.get(pmid, "")
        hits = [n for n in names[c.symbol] if re.search(rf"\b{re.escape(_norm(n))}\b", text)]
        detail = f"mentions {hits[0]}" if hits else "does not name the target, an alias or a drug acting on it"
        checks.append(CheckResult(**base, check="pmid_mentions_target", passed=bool(hits), detail=detail))
    return checks

# --- C2c: record checks ---
# For the three criteria backed by a database record, per candidate:
#   record_matches       each cited Open Targets / expression ID is a real record for this target
#   score_within_record  a non-zero score is not above the ceiling from C2a
#   drug_multi_target    annotation, not pass/fail: a cited drug also acts on other targets; lists them and marks
#                        those associated with this disease
# Association citations may be written "ENSG...|MONDO...", so only the part before "|" is compared.
# Clinical status: a drug counts fully (up to 3) only if this target is its sole mechanism. A drug that also acts
# on other targets counts up to 2, and its other targets are listed, marking those associated with this disease
# In:  TYK2 claims clinical_status 3, citing UPADACITINIB (acts on JAK1, JAK2, JAK3, TYK2; approved for AD)
# Out: CheckResult(symbol="TYK2", criterion="clinical_status", citation_id="CHEMBL3622821", check="drug_multi_target",
#                  passed=True, annotation=True, detail="UPADACITINIB also acts on JAK1 (associated with this disease),
#                  JAK2 (associated with this disease), JAK3 (...); evidence is shared, so it supports TYK2 at most partly.")
#      CheckResult(symbol="TYK2", criterion="clinical_status", check="score_within_record", passed=False,
#                  detail="score 3, record supports up to 2")
# In:  _record_checks(result)   (IL4R claims skin_expression 3; its record is 27 TPM, skin rank 28)
# Out: [CheckResult(symbol="IL4R", criterion="skin_expression", citation_id="ENSG00000077238",
#                   check="record_matches", passed=True, detail="record found"),
#       CheckResult(symbol="IL4R", criterion="skin_expression", citation_id=None, check="score_within_record",
#                   passed=False, detail="score 3, record supports up to 2"), ...]

def _is_multi_target(drug, symbol: str) -> bool:
    # Unknown targets (empty list) are not penalised.
    return bool(drug.drug_targets) and set(drug.drug_targets) != {symbol}


def _multi_target_detail(drug, symbol: str, associated_symbols: set[str]) -> str:
    others = [f"{t} (associated with this disease)" if t in associated_symbols else t
              for t in drug.drug_targets if t != symbol]
    return f"{drug.drug_name} also acts on {', '.join(others)}; evidence is shared, so it supports {symbol} at most partly."


def _record_checks(result: RankingResult) -> list[CheckResult]:
    associations = {a.ensembl_id: a for a in get_associated_targets(result.disease_id, size=50)}
    associated_symbols = {a.symbol for a in associations.values()}
    checks = []
    for c in result.candidates:
        scores = {s.criterion: s for s in c.scores}
        drugs = {d.drug_id: d for d in get_known_drugs(c.ensembl_id, result.disease_id)}
        expression = get_skin_expression(c.ensembl_id)
        assoc = associations.get(c.ensembl_id)

        valid_ids = {
            "genetic_association": ("opentargets_association", {c.ensembl_id} if assoc else set()),
            "skin_expression": ("expression", {c.ensembl_id}),
            "clinical_status": ("opentargets_drug", set(drugs)),
        }
        for criterion, (source, ids) in valid_ids.items():
            for x in scores[criterion].citations if criterion in scores else []:
                if x.source != source:
                    continue
                ok = x.id.split("|")[0] in ids
                detail = "record found" if ok else f"no {source} record {x.id} for {c.symbol}"
                checks.append(CheckResult(symbol=c.symbol, criterion=criterion, citation_id=x.id,
                                          check="record_matches", passed=ok, detail=detail))

        drug_citations = scores["clinical_status"].citations if "clinical_status" in scores else []
        cited_drugs = [drugs[x.id] for x in drug_citations if x.id in drugs]
        for d in [d for d in cited_drugs if _is_multi_target(d, c.symbol)]:
            checks.append(CheckResult(symbol=c.symbol, criterion="clinical_status", citation_id=d.drug_id,
                                      check="drug_multi_target", passed=True, annotation=True,
                                      detail=_multi_target_detail(d, c.symbol, associated_symbols)))
        ceilings = {
            "genetic_association": _association_ceiling(assoc) if assoc else 0,
            "skin_expression": _expression_ceiling(expression),
            "clinical_status": max((min(_clinical_ceiling(d.stage_for_disease),
                                        2 if _is_multi_target(d, c.symbol) else 3) for d in cited_drugs), default=0),
        }
        for criterion, ceiling in ceilings.items():
            s = scores.get(criterion)
            if s is None or s.score == 0:
                continue
            checks.append(CheckResult(symbol=c.symbol, criterion=criterion, check="score_within_record",
                                      passed=s.score <= ceiling,  max_score=ceiling,
                                      detail=f"score {s.score}, record supports up to {ceiling}"))
    return checks

# --- C2d: all deterministic checks ---
# First confirms each candidate's Ensembl ID is a real target with the claimed symbol (an LLM can mix up IDs);
# candidates that fail are not checked further, so a bad ID can't crash the lookups. Then runs C2b and C2c.
# In:  run_checks(result)
# Out: [CheckResult(symbol="IL4R", criterion=None, citation_id="ENSG00000077238", check="target_matches",
#                   passed=True, detail="IL4R"), CheckResult(... "pmid_mentions_target" ...), ...]

def run_checks(result: RankingResult) -> list[CheckResult]:
    checks, valid = [], []
    for c in result.candidates:
        try:
            symbol = get_target_info(c.ensembl_id).symbol
        except ValueError:
            symbol = None
        ok = symbol == c.symbol
        detail = c.symbol if ok else f"{c.ensembl_id} is {symbol or 'not a known target'}, not {c.symbol}"
        checks.append(CheckResult(symbol=c.symbol, citation_id=c.ensembl_id, check="target_matches",
                                  passed=ok, detail=detail))
        if ok:
            valid.append(c)

    checked = result.model_copy(update={"candidates": valid})
    return checks + _pmid_checks(checked) + _record_checks(checked)



# --- C3: apply the check results ---
# Turns failed checks into changes on a copy of the result, then re-applies the citation rule:
#   target_matches failed           -> every citation of that candidate removed (all its scores become 0)
#   pmid_* or record_matches failed -> that citation removed (a score left without citations becomes 0)
#   score_within_record failed      -> score capped at the record's maximum
#   annotations (drug_multi_target) -> added as a note (nothing removed; any cap comes via score_within_record)
# Each change adds a flag. The original result is not modified.
# In:  apply_checks(result, [CheckResult(symbol="FLG", criterion="literature", citation_id="99999999",
#                                        check="pmid_exists", passed=False, ...)])   (FLG literature cites only that PMID)
# Out: RankingResult where FLG literature = 0, flags include
#      "literature: citation 99999999 removed (PMID not found in PubMed)" and "literature: score 2 set to 0 (no citation)",
#      totals recomputed and candidates re-ranked

def apply_checks(result: RankingResult, checks: list[CheckResult]) -> RankingResult:
    result = result.model_copy(deep=True)
    candidates = {c.symbol: c for c in result.candidates}
    for check in checks:
        c = candidates.get(check.symbol)
        if c is None:
            continue
        if check.annotation:
            c.flags.append(f"{check.criterion}: note: {check.detail}")
            continue
        if check.passed:
            continue
        scores = {s.criterion: s for s in c.scores}
        if check.check == "target_matches":
            for s in c.scores:
                s.citations = []
            c.mechanism_citations = []
            c.flags.append(f"target: {check.detail}; all citations removed")
        elif check.check == "score_within_record":
            s = scores[check.criterion]
            c.flags.append(f"{check.criterion}: score {s.score} capped at {check.max_score} (record)")
            s.score = check.max_score
        else:
            if check.criterion == "mechanism_statement":
                c.mechanism_citations = [x for x in c.mechanism_citations if x.id != check.citation_id]
            else:
                s = scores[check.criterion]
                s.citations = [x for x in s.citations if x.id != check.citation_id]
            c.flags.append(f"{check.criterion}: citation {check.citation_id} removed ({check.detail})")
    return enforce_citations(result)

# --- C4c: apply the critic's verdicts ---
# Turns the critic agent's report into changes on a copy of the result:
#   partial or unsupported verdict -> flag ("critic: literature partial (only a safety review is cited)")
#   unsupported on a score         -> that score set to 0, flagged
#   unsupported mechanism_statement, confidence flags -> flagged only (no score to change)
# Then totals are recomputed and candidates re-ranked (enforce_citations).
# In:  apply_verdicts(result, CriticReport(verdicts=[ClaimVerdict(symbol="TYK2", criterion="literature",
#                                                   verdict="unsupported", reason="review is not about AD")]))
# Out: RankingResult where TYK2 literature = 0, total lowered by 1, with two flags explaining why

def apply_verdicts(result: RankingResult, report: CriticReport) -> RankingResult:
    result = result.model_copy(deep=True)
    candidates = {c.symbol: c for c in result.candidates}
    for v in report.verdicts:
        c = candidates.get(v.symbol)
        if c is None or v.verdict == "supported":
            continue
        c.flags.append(f"critic: {v.criterion} {v.verdict} ({v.reason})")
        if v.verdict == "unsupported" and v.criterion != "mechanism_statement":
            for s in c.scores:
                if s.criterion == v.criterion and s.score > 0:
                    c.flags.append(f"{v.criterion}: score {s.score} set to 0 (critic: unsupported)")
                    s.score = 0
    for f in report.confidence_flags:
        if f.symbol in candidates:
            candidates[f.symbol].flags.append(f"critic: confidence too high ({f.reason})")
    return enforce_citations(result)


# --- C4d: run the critic ---
# Deterministic checks -> apply them -> critic agent (shared loop, read-only tools, 20 tool calls) -> apply its
# verdicts. Writes to the same trace file as the ranker, labelled agent="critic". If the critic fails to submit
# valid verdicts, the checked result is still returned, with "error" set.
# In:  critique(ranking_result, run_id="20261101-103000-atopic-dermatitis")
# Out: {"model": "google/gemini-3.8-flash", "prompt_version": "critic-v1", "stop_reason": "submitted",
#       "partial": False, "error": None, "usage": {"tool_calls": 6, "cost": 0.004, ...},
#       "checks": [...], "report": {"verdicts": [...], "confidence_flags": [...]}, "result": {...final ranking...}}

def _critic_run_tool(name: str, args: dict) -> str:
    if name not in CRITIC_TOOL_NAMES:
        return f"Error: {name} is not available to the critic"
    return run_tool(name, args)


def critique(ranking: RankingResult, run_id: str, client=None) -> dict:
    checks = run_checks(ranking)
    checked = apply_checks(ranking, checks)
    model = os.environ["CRITIC_MODEL"]
    reasoning_effort = os.getenv("CRITIC_REASONING_EFFORT") or None
    agent_result = run_agent(
        client=client or make_client(), agent="critic", model=model,
        system_prompt=CRITIC_SYSTEM_PROMPT, user_prompt=critic_user_prompt(checked, checks),
        tools=CRITIC_TOOLS, run_tool=_critic_run_tool, submit_tool="submit_verdicts",
        validate=schema_validator(CriticReport), reasoning_effort=reasoning_effort,
        max_tool_calls=20, max_cost=float(os.getenv("MAX_COST_PER_RUN_USD", "0.50")),
        timeout_s=600, trace_path=f"traces/{run_id}.jsonl", prompt_version=CRITIC_PROMPT_VERSION)

    report, error = None, None
    if agent_result.final_args is None:
        error = "no verdicts submitted"
    else:
        try:
            report = CriticReport.model_validate(agent_result.final_args)
        except ValidationError as e:
            error = f"verdicts failed validation: {e}"

    final = apply_verdicts(checked, report) if report else checked
    usage = {k: getattr(agent_result, k)
             for k in ("tool_calls", "prompt_tokens", "completion_tokens", "cost", "runtime_s")}
    return {"model": model, "prompt_version": CRITIC_PROMPT_VERSION, "reasoning_effort": reasoning_effort,
            "stop_reason": agent_result.stop_reason, "partial": agent_result.partial, "error": error, "usage": usage,
            "checks": [c.model_dump() for c in checks],
            "report": report.model_dump() if report else None, "result": final.model_dump()}

