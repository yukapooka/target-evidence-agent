"""Offline test for the tab-delimited export.

Objective:
    Check that a run becomes two Excel-readable .tsv files with the expected rows and columns.

How:
    - A run dict is built from the one-candidate IL4R result used in the critic tests, with one critic verdict,
      one failed check, and a lower final literature score than the ranker gave.
    - export_tsv writes to tmp_path; the files are read back with the csv module.
"""

import csv

from agent.export import export_tsv
from tests.test_critic import _result


# --- Export ---
# Fake: run with ranker literature 2 -> final 0, critic verdict "unsupported" and a failed pmid_exists check.
# Check: 1 candidate row and 6 claim rows (5 criteria + mechanism statement); before/after scores, verdict and
#        failed check land in the right columns; a text with a tab and line break stays on one row.

def test_export_tsv_writes_candidate_and_claim_files(tmp_path):
    before = _result().model_dump()
    after = _result().model_dump()
    after["candidates"][0]["scores"][2]["score"] = 0
    after["candidates"][0]["total_score"] = 10
    after["candidates"][0]["rationale"] = "line one\nline\ttwo"
    run = {"run_id": "run1", "ranker_result": before, "result": after, "critic": {
        "checks": [{"symbol": "IL4R", "criterion": "literature", "check": "pmid_exists", "passed": False,
                    "detail": "PMID not found in PubMed"}],
        "report": {"verdicts": [{"symbol": "IL4R", "criterion": "literature", "verdict": "unsupported",
                                 "reason": "off topic"}], "confidence_flags": []}}}

    candidates_path, claims_path = export_tsv(run, out_dir=str(tmp_path))

    with open(candidates_path, encoding="utf-8-sig") as f:
        candidates = list(csv.DictReader(f, delimiter="\t"))
    with open(claims_path, encoding="utf-8-sig") as f:
        claims = list(csv.DictReader(f, delimiter="\t"))

    assert len(candidates) == 1 and len(claims) == 6
    assert (candidates[0]["total_score"], candidates[0]["ranker_total_score"]) == ("10", "12")
    assert candidates[0]["rationale"] == "line one line two"
    literature = next(row for row in claims if row["claim"] == "literature")
    assert (literature["ranker_score"], literature["final_score"]) == ("2", "0")
    assert (literature["critic_verdict"], literature["critic_reason"]) == ("unsupported", "off topic")
    assert literature["check_notes"] == "pmid_exists: PMID not found in PubMed"
