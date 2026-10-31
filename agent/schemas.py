"""Data shapes the agents exchange: citations, rubric scores, candidates and the final ranking.

Objective:
    Give the ranking agent (and later the critic) one fixed, validated output format, so results can be
    checked in code, saved to runs/ and compared between runs.

Models:
    Citation        one piece of tool evidence: where it came from and its ID
    CriterionScore  a 0-3 score on one rubric criterion, with a reason and citations
    Candidate       one ranked target: 5 criterion scores, mechanism, novelty, confidence, rationale, flags
    RankingResult   the whole answer for one condition: candidates in rank order, plus a "partial" flag

Rationale:
    - Pydantic validates the LLM's output: a score of 5 or a confidence of "very high" is rejected,
      and the error goes back to the model to fix.
    - RankingResult.model_json_schema() becomes the parameters of the submit_ranking tool, so the
      schema is written once and used both for the LLM and for validation.
    - Literal types limit fields to fixed choices (e.g. confidence is only "high", "medium" or "low").
"""

from typing import Literal

from pydantic import BaseModel, Field


# --- A2a: Citation ---
# In:  Citation(source="pmid", id="26454361", note="Phase 3 trial of dupilumab in AD")
#      Citation(source="opentargets_association", id="ENSG00000077238|MONDO_0004980", note="overall 0.82")

# --- A2b: CriterionScore ---
# In:  CriterionScore(criterion="literature", score=3, reason="Multiple AD trials", citations=[Citation(...)])
#      CriterionScore(criterion="literature", score=5, ...)  -> ValidationError (score must be 0-3)

# --- A2c: Candidate ---
# In:  Candidate(rank=1, symbol="IL4R", ensembl_id="ENSG00000077238", scores=[5 x CriterionScore],
#                total_score=13, mechanism="IL-4Ra mediates IL-4/IL-13 Th2 signalling ...",
#                mechanism_citations=[Citation(source="pmid", ...)], novelty="known",
#                confidence="high", rationale="...", flags=[])

# --- A2d: RankingResult ---
# In:  RankingResult(condition="atopic dermatitis", disease_id="MONDO_0004980", candidates=[...], partial=False)

class Citation(BaseModel):
    source: Literal["pmid", "opentargets_association", "opentargets_drug", "expression"]
    id: str
    note: str

class CriterionScore(BaseModel):
    criterion: Literal["genetic_association", "skin_expression", "literature", "clinical_status", "mechanism"]
    score: int = Field(ge=0, le=3)
    reason: str
    citations: list[Citation]

class Candidate(BaseModel):
    rank: int
    symbol: str
    ensembl_id: str
    scores: list[CriterionScore]
    total_score: int
    mechanism: str
    mechanism_citations: list[Citation]
    novelty: Literal["known", "less_explored"]
    confidence: Literal["high", "medium", "low"]
    rationale: str
    flags: list[str] = []

class RankingResult(BaseModel):
    condition: str
    disease_id: str
    candidates: list[Candidate]
    partial: bool = False

