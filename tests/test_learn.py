"""Offline tests for the Learn step: review queue order and reviewer feedback.

Objective:
    Check that the review queue puts the least certain candidates first, for the stated reasons.

How:
    - A small run dict is built by hand: three candidates, a critic report and checks, as saved by run_pipeline.
"""

import json

from agent import learn
from agent.learn import review_queue


# --- D1: review queue ---
# Fake: A is high confidence with nothing flagged; B is low confidence with a partial verdict and a multi-target note;
#       C is medium confidence, less explored, with one failed check.
# Check: order B (2 + 1 + 0.5 = 3.5), C (1 + 1 + 1 = 3), A (0); every candidate stays in the queue.

def _candidate(rank, symbol, confidence, novelty="known"):
    return {"rank": rank, "symbol": symbol, "confidence": confidence, "novelty": novelty}


def test_review_queue_orders_least_certain_first():
    run = {"result": {"candidates": [_candidate(1, "A", "high"), _candidate(2, "B", "low"),
                                     _candidate(3, "C", "medium", novelty="less_explored")]},
           "critic": {"report": {"verdicts": [{"symbol": "B", "criterion": "clinical_status", "verdict": "partial"},
                                              {"symbol": "A", "criterion": "literature", "verdict": "supported"}],
                                 "confidence_flags": []},
                      "checks": [{"symbol": "B", "criterion": "clinical_status", "citation_id": "CHEMBL1",
                                  "check": "drug_multi_target", "passed": True, "annotation": True},
                                 {"symbol": "C", "criterion": "literature", "citation_id": "99999999",
                                  "check": "pmid_exists", "passed": False}]}}

    queue = review_queue(run)

    assert [(q["symbol"], q["score"]) for q in queue] == [("B", 3.5), ("C", 3), ("A", 0)]
    assert [text for _, text in queue[1]["reasons"]] == [
        "medium confidence", "failed check: pmid_exists (literature)", "less explored target"]


# --- D2: reviewer feedback ---
# Fake: a decisions file with one entry for this condition (naming the run it reviewed) and one for another
#       condition; a saved run where IL31RA's genetic_association was 1. Files live in tmp_path.
# Check: only this condition's entry is used, the edit shows "1 -> 2", and an unknown condition gives None.

def test_summarize_feedback_shows_edits_for_this_condition(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "run1.json").write_text(json.dumps({"result": {"candidates": [
        {"symbol": "IL31RA", "scores": [{"criterion": "genetic_association", "score": 1}]}]}}))
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps([
        {"condition": "Atopic dermatitis", "run_id": "run1", "symbol": "IL31RA", "decision": "approve",
         "edited_scores": {"genetic_association": 2}, "comment": "overall 0.56 is a 2"},
        {"condition": "psoriasis", "symbol": "IL17A", "decision": "reject"}]))
    monkeypatch.setattr(learn, "DECISIONS_PATH", decisions)

    assert learn.summarize_feedback("atopic dermatitis") == (
        "IL31RA: approve; genetic_association edited 1 -> 2; comment: overall 0.56 is a 2")
    assert learn.summarize_feedback("alopecia areata") is None


# --- D3: run comparison ---
# Fake: run 1 ranks IL4R, FLG, IL31RA (IL31RA genetic_association 1); run 2 ranks IL4R, IL31RA, TYK2 (IL31RA 2);
#       the reviewer edited IL31RA genetic_association to 2. A run for another condition is also saved.
# Check: previous_run picks run 1 (not the other condition); added TYK2, dropped FLG, IL31RA moved 3 -> 2,
#        and the edit shows before 1, reviewer 2, now 2.

def _run(run_id, condition, ranked):
    return {"run_id": run_id, "condition": condition, "result": {"candidates": [
        {"symbol": s, "rank": i, "scores": [{"criterion": "genetic_association", "score": g}]}
        for i, (s, g) in enumerate(ranked, start=1)]}}


def test_compare_runs_reports_changes_and_reviewer_edits(tmp_path):
    run1 = _run("20261101-100000-atopic-dermatitis", "atopic dermatitis", [("IL4R", 3), ("FLG", 3), ("IL31RA", 1)])
    other = _run("20261101-110000-psoriasis", "psoriasis", [("IL17A", 3)])
    run2 = _run("20261101-120000-atopic-dermatitis", "atopic dermatitis", [("IL4R", 3), ("IL31RA", 2), ("TYK2", 3)])
    for run in (run1, other, run2):
        (tmp_path / f'{run["run_id"]}.json').write_text(json.dumps(run))
    decisions = [{"condition": "atopic dermatitis", "symbol": "IL31RA", "decision": "keep",
                  "edited_scores": {"genetic_association": 2}}]

    previous = learn.previous_run(run2, runs_dir=str(tmp_path))
    diff = learn.compare_runs(previous, run2, decisions)

    assert diff["previous_run_id"] == run1["run_id"]
    assert (diff["added"], diff["dropped"], diff["rank_changes"]) == (["TYK2"], ["FLG"], [["IL31RA", 3, 2]])
    assert diff["edits"] == [{"symbol": "IL31RA", "criterion": "genetic_association",
                              "reviewer": 2, "before": 1, "now": 2}]

