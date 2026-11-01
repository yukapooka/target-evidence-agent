"""Evidence rubric: the five scoring criteria, and the rule that every non-zero score must be cited.

Objective:
    Define one shared scale (0-3 per criterion) for the ranking agent, and enforce the citation rule in code
    rather than trusting the LLM to follow it.

Functions:
    CRITERIA             criterion name -> its source and what each score 0-3 means
    rubric_text()        CRITERIA as plain text, pasted into the ranking agent's system prompt
    enforce_citations()  RankingResult -> same result with uncited non-zero scores set to 0 and flagged,
                         totals recomputed and candidates re-ranked

Rationale:
    - Criterion names match CriterionScore.criterion in schemas.py exactly.
    - The rubric lives in data (a dict), so the prompt and any later checks read from one place.
    - Rules in code, not trust: the LLM is told to cite, and enforce_citations catches it when it doesn't.
"""

from agent.schemas import RankingResult


# --- A3a: criteria ---
# One entry per rubric criterion: where the evidence comes from, and what each score means.
# In:  CRITERIA["clinical_status"]["levels"][3]
# Out: "Approved, or Phase 3, for this disease"

CRITERIA = {
    "genetic_association": {
        "source": "Open Targets association scores (get_associated_targets)",
        "levels": {
            0: "No association with this disease",
            1: "Weak association (overall score below 0.3)",
            2: "Moderate association (overall 0.3-0.6), or strong in only one evidence type",
            3: "Strong association (overall above 0.6) across several evidence types",
        },
    },
    "skin_expression": {
        "source": "GTEx skin expression (get_skin_expression)",
        "levels": {
            0: "Not detected in skin (below 1 TPM)",
            1: "Low (1-10 TPM)",
            2: "Moderate (10-50 TPM)",
            3: "High (above 50 TPM), or skin among the top 10 tissues",
        },
        "note": "Secreted cytokines (e.g. IL13) often show very low bulk skin TPM even when central "
                "to the disease. Score the TPM as measured, and say in the reason that it is not informative.",
    },
    "literature": {
        "source": "PubMed (search_pubmed, fetch_abstracts)",
        "levels": {
            0: "No papers linking the target to this disease",
            1: "One or two papers, or indirect evidence only",
            2: "Several papers in the disease context",
            3: "Clinical trial results or multiple strong human studies in this disease",
        },
    },
    "clinical_status": {
        "source": "Open Targets drugs and clinical candidates (get_known_drugs)",
        "levels": {
            0: "No drug tested for this disease",
            1: "Early phase (Phase 1) for this disease",
            2: "Phase 2 for this disease",
            3: "Approved, or Phase 3, for this disease, with a drug whose only target is this one "
               "(a drug that also acts on other targets counts as 2)",
        },
    },
    "mechanism": {
        "source": "PubMed (must cite a PMID)",
        "levels": {
            0: "No plausible mechanism",
            1: "Plausible pathway link, without disease-specific evidence",
            2: "Mechanism supported by disease-context evidence (human or animal)",
            3: "Well-established mechanism with human evidence in this disease",
        },
    },
}

# --- A3b: rubric as prompt text ---
# In:  rubric_text()
# Out: "genetic_association (source: Open Targets association scores)\n  0: ...\n  1: ...\n..."

def rubric_text() -> str:
    """CRITERIA as plain text for the system prompt."""
    lines = []
    for name, spec in CRITERIA.items():
        lines.append(f"{name} (source: {spec['source']})")
        for score, meaning in spec["levels"].items():
            lines.append(f"  {score}: {meaning}")
        if "note" in spec:
            lines.append(f"  Note: {spec['note']}")
    return "\n".join(lines)

# --- A3c: citation rule ---
# Any score > 0 with no citations -> score 0, plus a flag on the candidate. Then totals and ranks are recomputed.
# In:  candidate IL4R with CriterionScore(criterion="literature", score=2, citations=[])
# Out: same candidate, literature score 0, flags=["literature: score 2 set to 0 (no citation)"],
#      total_score lowered by 2, candidates re-sorted by total_score
def enforce_citations(result: RankingResult) -> RankingResult:
    """Zero and flag uncited scores, then recompute totals and re-rank."""
    for candidate in result.candidates:
        for s in candidate.scores:
            if s.score > 0 and not s.citations:
                candidate.flags.append(f"{s.criterion}: score {s.score} set to 0 (no citation)")
                s.score = 0
        no_pmid = not any(c.source == "pmid" for c in candidate.mechanism_citations)
        if no_pmid and "mechanism: no PMID cited" not in candidate.flags:   # safe to run twice
            candidate.flags.append("mechanism: no PMID cited")
        candidate.total_score = sum(s.score for s in candidate.scores)

    result.candidates.sort(key=lambda c: c.total_score, reverse=True)
    for i, candidate in enumerate(result.candidates, start=1):
        candidate.rank = i
    return result