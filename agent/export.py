"""Tab-delimited export of a saved run, for reading in Excel.

Objective:
    Turn a run (runs/<run_id>.json) into two .tsv files that scientists can open in Excel:
      <run_id>_candidates.tsv  one row per candidate: rank, scores by criterion, confidence, mechanism, flags
      <run_id>_claims.tsv      one row per claim (candidate x criterion, plus the mechanism statement):
                               score before and after review, citations, reason, critic verdict, failed checks and notes

Contents:
    export_tsv()   run dict -> writes the two files next to the run file and returns their paths
    command line:  python -m agent.export runs/<run_id>.json   (for runs saved earlier)

Rationale:
    - Standard library csv module with a tab delimiter: no new dependency.
    - Files are written as UTF-8 with a byte-order mark ("utf-8-sig"), so Excel shows characters like "α" correctly.
    - Tabs and line breaks inside text are replaced by spaces, so every record stays on one row.
    - Works on runs without a critic section (older runs, or a ranker that failed validation); those columns stay empty.
"""

import argparse
import csv
import json
from pathlib import Path

CRITERIA = ["genetic_association", "skin_expression", "literature", "clinical_status", "mechanism"]


# --- Helpers: clean text and look-ups ---
# _clean() keeps one record per row. _citations() writes citations compactly.
# In:  _clean("line one\nline\ttwo")                                  -> "line one line two"
# In:  _citations([{"source": "pmid", "id": "33761207", ...}, {"source": "expression", "id": "ENSG..."}])
# Out: "pmid:33761207; expression:ENSG..."

def _clean(value) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _citations(citations: list[dict]) -> str:
    return "; ".join(f'{c["source"]}:{c["id"]}' for c in citations)


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows([[_clean(v) for v in row] for row in rows])


# --- Export a run ---
# In:  export_tsv(run)   (run as saved by run_pipeline, or loaded from runs/<run_id>.json)
# Out: [Path("runs/20261101-035703-atopic-dermatitis_candidates.tsv"),
#       Path("runs/20261101-035703-atopic-dermatitis_claims.tsv")]   ([] if the run has no result)

def export_tsv(run: dict, out_dir: str = "runs") -> list[Path]:
    final = run.get("result")
    if not final:
        return []
    before = {c["symbol"]: c for c in (run.get("ranker_result") or final)["candidates"]}
    critic = run.get("critic") or {}
    report = critic.get("report") or {}
    verdicts = {(v["symbol"], v["criterion"]): v for v in report.get("verdicts", [])}
    failed = {}
    for check in critic.get("checks", []):
        if check.get("annotation"):
            failed.setdefault((check["symbol"], check["criterion"]), []).append(f'note: {check["detail"]}')
        elif not check["passed"]:
            failed.setdefault((check["symbol"], check["criterion"]), []).append(f'{check["check"]}: {check["detail"]}')

    candidate_rows, claim_rows = [], []
    for c in final["candidates"]:
        scores = {s["criterion"]: s for s in c["scores"]}
        original = before.get(c["symbol"], c)
        original_scores = {s["criterion"]: s["score"] for s in original["scores"]}
        candidate_rows.append(
            [c["rank"], c["symbol"], c["ensembl_id"], c["total_score"], original["total_score"],
             c["confidence"], c["novelty"]]
            + [scores[k]["score"] if k in scores else "" for k in CRITERIA]
            + [c["mechanism"], _citations(c["mechanism_citations"]), c["rationale"], " | ".join(c["flags"])])

        for k in CRITERIA:
            s = scores.get(k, {"score": "", "citations": [], "reason": ""})
            v = verdicts.get((c["symbol"], k), {})
            claim_rows.append([c["rank"], c["symbol"], k, original_scores.get(k, ""), s["score"],
                               _citations(s["citations"]), s["reason"], v.get("verdict", ""),
                               v.get("reason", ""), "; ".join(failed.get((c["symbol"], k), []))])
        v = verdicts.get((c["symbol"], "mechanism_statement"), {})
        claim_rows.append([c["rank"], c["symbol"], "mechanism_statement", "", "",
                           _citations(c["mechanism_citations"]), c["mechanism"], v.get("verdict", ""),
                           v.get("reason", ""), "; ".join(failed.get((c["symbol"], "mechanism_statement"), []))])

    Path(out_dir).mkdir(exist_ok=True)
    paths = [Path(out_dir) / f'{run["run_id"]}_candidates.tsv', Path(out_dir) / f'{run["run_id"]}_claims.tsv']
    _write(paths[0], ["rank", "symbol", "ensembl_id", "total_score", "ranker_total_score", "confidence",
                      "novelty"] + CRITERIA + ["mechanism_statement", "mechanism_citations", "rationale", "flags"],
           candidate_rows)
    _write(paths[1], ["rank", "symbol", "claim", "ranker_score", "final_score", "citations", "reason",
                      "critic_verdict", "critic_reason", "check_notes"], claim_rows)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a saved run as two tab-delimited files.")
    parser.add_argument("run_file", help="e.g. runs/20261101-035703-atopic-dermatitis.json")
    args = parser.parse_args()
    for path in export_tsv(json.loads(Path(args.run_file).read_text()), out_dir=str(Path(args.run_file).parent)):
        print(f"Saved {path}")
