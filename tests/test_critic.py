"""Offline tests for the critic: deterministic checks (and, from C3/C4, applying them and the critic agent).

Objective:
    Check that run_checks catches missing PMIDs, scores above their record and mismatched targets,
    without calling any API.

How:
    - The fake_tools fixture replaces every wrapper that critic.py imports with a stub returning one small,
      consistent IL4R record (27.3 TPM, skin rank 28, association 0.7, dupilumab approved).
    - _result() builds a one-candidate RankingResult that matches those records; each test changes one thing.

Rationale:
    - Patching the names inside agent.critic (not the wrapper modules) is what makes run_checks use the stubs.
    - One planted error per test makes each failure easy to read.
"""

import pytest

from agent import critic
from agent.schemas import Candidate, Citation, CriterionScore, RankingResult
from agent.tools.expression import SkinExpression, TissueTPM
from agent.tools.opentargets import KnownDrug, TargetAssociation, TargetInfo
from agent.tools.pubmed import Article


# --- C2e: deterministic checks ---
# Fake: all wrappers in agent.critic -> fixed IL4R records; PMID "99999999" doesn't exist; every abstract
#       title says "Dupilumab in atopic dermatitis" (so the mention check passes via the drug name).
# Check: (1) a consistent result passes every check; (2) a missing PMID and a skin score of 3 (record supports 2)
#        fail, and nothing else does; (3) a symbol that doesn't match its Ensembl ID stops further checks.


IL4R = "ENSG00000077238"


@pytest.fixture
def fake_tools(monkeypatch):
    monkeypatch.setattr(critic, "get_target_info", lambda ensembl_id: TargetInfo(
        ensembl_id=IL4R, symbol="IL4R", name="interleukin 4 receptor", synonyms=["CD124"]))
    monkeypatch.setattr(critic, "get_known_drugs", lambda ensembl_id, disease_id: [KnownDrug(
        drug_id="CHEMBL1", drug_name="DUPILUMAB", stage_for_disease="APPROVAL",
        max_stage_any_disease="APPROVAL", mechanisms=[])])
    monkeypatch.setattr(critic, "get_skin_expression", lambda ensembl_id: SkinExpression(
        ensembl_id=IL4R, symbol="IL4R", skin_rank=28, gtex_tissue_count=51,
        skin_tissues=[TissueTPM(tissue="Skin", uberon_id="UBERON_1", median_tpm=27.3)]))
    monkeypatch.setattr(critic, "get_associated_targets", lambda disease_id, size: [TargetAssociation(
        ensembl_id=IL4R, symbol="IL4R", name="x", overall_score=0.7, datatype_scores={})])
    monkeypatch.setattr(critic, "check_pmids", lambda pmids: {p: p != "99999999" for p in pmids})
    monkeypatch.setattr(critic, "fetch_abstracts", lambda pmids: [
        Article(pmid=p, title="Dupilumab in atopic dermatitis", abstract="", journal="J") for p in pmids])


def _result(skin_score=2, literature_pmid="111"):
    def score(criterion, value, citation):
        return CriterionScore(criterion=criterion, score=value, reason="test", citations=[citation])
    scores = [
        score("genetic_association", 3, Citation(source="opentargets_association", id=IL4R, note="")),
        score("skin_expression", skin_score, Citation(source="expression", id=IL4R, note="")),
        score("literature", 2, Citation(source="pmid", id=literature_pmid, note="")),
        score("clinical_status", 3, Citation(source="opentargets_drug", id="CHEMBL1", note="")),
        score("mechanism", 2, Citation(source="pmid", id="111", note="")),
    ]
    candidate = Candidate(rank=1, symbol="IL4R", ensembl_id=IL4R, scores=scores, total_score=12,
                          mechanism="test", mechanism_citations=[Citation(source="pmid", id="111", note="")],
                          novelty="known", confidence="high", rationale="test")
    return RankingResult(condition="atopic dermatitis", disease_id="MONDO_0004980", candidates=[candidate])


def _failed(checks):
    return {(c.check, c.criterion) for c in checks if not c.passed}


def test_all_checks_pass_on_consistent_result(fake_tools):
    assert _failed(critic.run_checks(_result())) == set()


def test_missing_pmid_and_score_above_record_fail(fake_tools):
    checks = critic.run_checks(_result(skin_score=3, literature_pmid="99999999"))
    assert _failed(checks) == {("pmid_exists", "literature"), ("score_within_record", "skin_expression")}


def test_mismatched_target_is_not_checked_further(fake_tools):
    result = _result()
    result.candidates[0].symbol = "IL13"
    checks = critic.run_checks(result)
    assert [(c.check, c.passed) for c in checks] == [("target_matches", False)]