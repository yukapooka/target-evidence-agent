"""Offline tests for the agent layer: rubric rules (and, from B1, the tool loop with a fake LLM).

Objective:
    Check the agent's code-enforced rules without calling an LLM or any API.

How:
    - Rubric tests build a RankingResult by hand, run enforce_citations, and check what changed.

Rationale:
    - The citation rule is the main guard against scores from model memory, so it is tested directly.
"""

from agent.rubric import enforce_citations
from agent.schemas import Candidate, Citation, CriterionScore, RankingResult
from agent.tools.registry import FUNCTIONS, TOOLS, run_tool

CRITERIA_NAMES = ["genetic_association", "skin_expression", "literature", "clinical_status", "mechanism"]
PMID = Citation(source="pmid", id="26454361", note="dupilumab phase 3")


# --- Helper: build a candidate ---
# Every criterion gets the same score and a citation, except those listed in `uncited`, which get no citations.
# In:  _candidate("IL4R", rank=1, score=2, uncited=["literature"])
# Out: Candidate with 5 scores of 2 (literature uncited), total_score=10, mechanism cited with a PMID

def _candidate(symbol: str, rank: int, score: int, uncited: list[str] = ()) -> Candidate:
    scores = [
        CriterionScore(criterion=name, score=score, reason="test",
                       citations=[] if name in uncited else [PMID])
        for name in CRITERIA_NAMES
    ]
    return Candidate(rank=rank, symbol=symbol, ensembl_id=f"ENSG_{symbol}", scores=scores,
                     total_score=score * 5, mechanism="test", mechanism_citations=[PMID],
                     novelty="known", confidence="high", rationale="test")


# --- A3c: citation rule ---
# Fake: hand-built ranking. IL4R is 1st and JAK1 2nd, both total 10, but IL4R's literature score of 2 has no citation.
# Check: IL4R literature -> 0 and flagged, IL4R total -> 8, so JAK1 (all cited) moves up to rank 1.

def test_enforce_citations_zeroes_uncited_score_and_reranks():
    il4r = _candidate("IL4R", rank=1, score=2, uncited=["literature"])
    jak1 = _candidate("JAK1", rank=2, score=2)
    result = RankingResult(condition="atopic dermatitis", disease_id="MONDO_0004980", candidates=[il4r, jak1])

    result = enforce_citations(result)

    first, second = result.candidates
    assert (first.symbol, first.rank, first.total_score) == ("JAK1", 1, 10)
    assert (second.symbol, second.rank, second.total_score) == ("IL4R", 2, 8)
    literature = next(s for s in second.scores if s.criterion == "literature")
    assert literature.score == 0
    assert "literature: score 2 set to 0 (no citation)" in second.flags
    assert first.flags == []


# Fake: candidate whose mechanism cites only an Open Targets record. Check: flagged, but no score changes.

def test_enforce_citations_flags_mechanism_without_pmid():
    il4r = _candidate("IL4R", rank=1, score=2)
    il4r.mechanism_citations = [Citation(source="opentargets_association", id="x", note="test")]
    result = RankingResult(condition="atopic dermatitis", disease_id="MONDO_0004980", candidates=[il4r])

    result = enforce_citations(result)

    assert "mechanism: no PMID cited" in result.candidates[0].flags
    assert result.candidates[0].total_score == 10


# --- A4e: tool registry ---
# Fake: monkeypatch swaps one entry in FUNCTIONS for a stub, so no API is called.
# Check: a result comes back as compact JSON; a raised error comes back as "Error: ..." text, not an exception;
#        every tool in TOOLS except submit_ranking has a function.

def test_run_tool_returns_compact_json(monkeypatch):
    monkeypatch.setitem(FUNCTIONS, "check_pmids", lambda pmids: {p: True for p in pmids})
    assert run_tool("check_pmids", {"pmids": ["1"]}) == '{"1":true}'

def test_run_tool_returns_errors_as_text(monkeypatch):
    def fail(ensembl_id):
        raise ValueError(f"Unknown target ID: {ensembl_id}")
    monkeypatch.setitem(FUNCTIONS, "get_target_info", fail)
    assert run_tool("get_target_info", {"ensembl_id": "X"}) == "Error: ValueError: Unknown target ID: X"
    assert run_tool("nope", {}) == "Error: unknown tool nope"

def test_every_tool_has_a_function():
    names = {t["function"]["name"] for t in TOOLS} - {"submit_ranking"}
    assert names == set(FUNCTIONS)