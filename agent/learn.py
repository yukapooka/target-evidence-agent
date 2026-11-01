"""Learn step: an uncertainty-ordered review queue, reviewer feedback into the next run, and (D3) run comparison.

Objective:
    Put the candidates the system is least sure about in front of the reviewer first, with the reason each one is
    near the top, so reviewer time goes where it can change the outcome most.

Contents:
    REVIEW_RULE           the scoring rule, printed with the queue so the order is never a black box
    review_queue()        run -> candidates sorted least certain first, each with its points and reasons
    print_review_queue()  prints the rule and the queue
    summarize_feedback()  condition -> short text of past reviewer decisions (review/decisions.json), or None
    compare_runs()        previous run, current run, decisions -> added, dropped, rank changes, edited scores
    print_comparison()    finds the previous run for the same condition and prints the comparison

Rationale:
    - A simple, stated points rule rather than a model: easy to check, easy to change. In the README this is described
      as a simple form of active learning (uncertainty sampling for human review), nothing more.
    - Less explored targets get a point (agreed 31 Oct), so novel candidates are not left at the bottom.
    - Multi-target drug notes are annotations, not failures, so they get half a point.
    - Nothing is filtered out: every candidate is in the queue, only the order changes.
    - Feedback is context for the next run, not evidence: the ranker still needs a tool citation for every score.
"""

import json
from pathlib import Path

DECISIONS_PATH = Path("review/decisions.json")
CONFIDENCE_POINTS = {"low": 2, "medium": 1, "high": 0}

REVIEW_RULE = ("low confidence +2, medium +1; +1 per partial or unsupported critic verdict; +1 per failed check; "
               "+1 per critic confidence flag; +0.5 per multi-target drug note; +1 if less explored")


# --- D1: review queue ---
# Scores each candidate of the final result with REVIEW_RULE, using the critic's verdicts and the deterministic checks
# saved in the run. Highest score first; ties keep the ranking order.
# In:  review_queue(run)
# Out: [{"rank": 5, "symbol": "TYK2", "score": 3.5,
#        "reasons": [[2, "low confidence"], [1, "critic partial: clinical_status"], [0.5, "note: multi-target drug CHEMBL4297477"]]},
#       {"rank": 4, "symbol": "FLG", "score": 2, "reasons": [[1, "medium confidence"], [1, "less explored target"]]}, ...]

def review_queue(run: dict) -> list[dict]:
    result = run.get("result")
    if not result:
        return []
    critic = run.get("critic") or {}
    report = critic.get("report") or {}
    checks = critic.get("checks", [])

    queue = []
    for c in result["candidates"]:
        symbol = c["symbol"]
        reasons = []
        if CONFIDENCE_POINTS[c["confidence"]]:
            reasons.append([CONFIDENCE_POINTS[c["confidence"]], f'{c["confidence"]} confidence'])
        for v in report.get("verdicts", []):
            if v["symbol"] == symbol and v["verdict"] != "supported":
                reasons.append([1, f'critic {v["verdict"]}: {v["criterion"]}'])
        for check in checks:
            if check["symbol"] != symbol:
                continue
            if check.get("annotation"):
                reasons.append([0.5, f'note: multi-target drug {check["citation_id"]}'])
            elif not check["passed"]:
                reasons.append([1, f'failed check: {check["check"]} ({check["criterion"] or "target"})'])
        for flag in report.get("confidence_flags", []):
            if flag["symbol"] == symbol:
                reasons.append([1, "critic: confidence too high"])
        if c["novelty"] == "less_explored":
            reasons.append([1, "less explored target"])
        queue.append({"rank": c["rank"], "symbol": symbol, "score": sum(p for p, _ in reasons), "reasons": reasons})
    return sorted(queue, key=lambda q: (-q["score"], q["rank"]))


# In:  print_review_queue(run)
# Out: Review queue (least certain first)
#        Rule: low confidence +2, medium +1; ...
#        1. TYK2      3.5  low confidence (+2); critic partial: clinical_status (+1); note: multi-target drug ... (+0.5)
#        2. FLG       2    medium confidence (+1); less explored target (+1)
#        5. IL4R      0    -

def print_review_queue(run: dict) -> None:
    queue = review_queue(run)
    if not queue:
        return
    print("\nReview queue (least certain first)")
    print(f"  Rule: {REVIEW_RULE}")
    for i, q in enumerate(queue, start=1):
        reasons = "; ".join(f"{text} (+{points:g})" for points, text in q["reasons"]) or "-"
        print(f'  {i}. {q["symbol"]:<9} {q["score"]:<4g} {reasons}')


# --- D2: reviewer feedback for the next run ---
# Reads review/decisions.json (written by the reviewer from the queue) and keeps the entries for this condition,
# one line each. If an entry names the run it reviewed, edited scores show the change ("1 -> 2"), using that run's
# final scores in runs/<run_id>.json.
# File format: [{"condition": "atopic dermatitis", "run_id": "20261101-121557-atopic-dermatitis", "symbol": "IL31RA",
#                "decision": "approve", "edited_scores": {"genetic_association": 2}, "comment": "overall 0.56 is a 2"}]
# decision: "approve" (take forward to validation), "keep" (keep ranking it, not yet approved) or "reject"
# (not for validation now). None of them removes a target: they are context for the next run.
# In:  summarize_feedback("atopic dermatitis")
# Out: "IL31RA: approve; genetic_association edited 1 -> 2; comment: overall 0.56 is a 2"
# In:  summarize_feedback("psoriasis")   -> None   (no file, or no entries for this condition)

def _run_scores(run_id: str | None) -> dict:
    path = Path("runs") / f"{run_id}.json"
    if not run_id or not path.exists():
        return {}
    result = json.loads(path.read_text()).get("result") or {"candidates": []}
    return {c["symbol"]: {s["criterion"]: s["score"] for s in c["scores"]} for c in result["candidates"]}


def summarize_feedback(condition: str) -> str | None:
    if not DECISIONS_PATH.exists():
        return None
    lines = []
    for d in json.loads(DECISIONS_PATH.read_text()):
        if d["condition"].lower() != condition.lower():
            continue
        before = _run_scores(d.get("run_id")).get(d["symbol"], {})
        line = f'{d["symbol"]}: {d["decision"]}'
        for criterion, score in (d.get("edited_scores") or {}).items():
            was = f"{before[criterion]} -> " if criterion in before else ""
            line += f"; {criterion} edited {was}{score}"
        if d.get("comment"):
            line += f'; comment: {d["comment"]}'
        lines.append(line)
    return "\n".join(lines) or None



# --- D3: comparison with the previous run ---
# Finds the latest earlier run for the same condition that has a result, and reports what changed: targets added
# or dropped, rank changes, and, for every score the reviewer edited, the score before, the reviewer's score and
# the score now. Uses final scores (after checks and the critic). Only decisions made on earlier runs count.
# In:  compare_runs(previous_run, current_run, decisions)
# Out: {"previous_run_id": "20261101-121557-atopic-dermatitis", "added": ["TYK2"], "dropped": ["FLG"],
#       "rank_changes": [["IL31RA", 5, 4]],
#       "edits": [{"symbol": "IL31RA", "criterion": "genetic_association", "before": 1, "reviewer": 2, "now": 2}]}

def previous_run(current: dict, runs_dir: str = "runs") -> dict | None:
    earlier = []
    for path in Path(runs_dir).glob("*.json"):
        run = json.loads(path.read_text())
        if (run.get("run_id", "") < current["run_id"] and run.get("result")
                and run.get("condition", "").lower() == current["condition"].lower()):
            earlier.append(run)
    return max(earlier, key=lambda r: r["run_id"], default=None)


def compare_runs(previous: dict, current: dict, decisions: list[dict]) -> dict:
    def ranks(run):
        return {c["symbol"]: c["rank"] for c in run["result"]["candidates"]}

    def scores(run):
        return {c["symbol"]: {s["criterion"]: s["score"] for s in c["scores"]} for c in run["result"]["candidates"]}

    before, now = ranks(previous), ranks(current)
    before_scores, now_scores = scores(previous), scores(current)
    edits = []
    for d in decisions:
        if d["condition"].lower() != current["condition"].lower():
            continue
        if d.get("run_id") and d["run_id"] >= current["run_id"]:   # decided after this run: not feedback for it
            continue
        for criterion, score in (d.get("edited_scores") or {}).items():
            edits.append({"symbol": d["symbol"], "criterion": criterion, "reviewer": score,
                          "before": before_scores.get(d["symbol"], {}).get(criterion),
                          "now": now_scores.get(d["symbol"], {}).get(criterion)})
    return {"previous_run_id": previous["run_id"],
            "added": [s for s in now if s not in before],
            "dropped": [s for s in before if s not in now],
            "rank_changes": [[s, before[s], now[s]] for s in now if s in before and before[s] != now[s]],
            "edits": edits}


# In:  print_comparison(current_run)
# Out: Changes since run 20261101-121557-atopic-dermatitis
#        Added: TYK2 | Dropped: FLG
#        Rank changes: IL31RA 5 -> 4
#        Reviewer edits: IL31RA genetic_association: before 1, reviewer 2, now 2 (matches reviewer)

def print_comparison(current: dict) -> None:
    previous = previous_run(current) if current.get("result") else None
    if previous is None:
        return
    decisions = json.loads(DECISIONS_PATH.read_text()) if DECISIONS_PATH.exists() else []
    diff = compare_runs(previous, current, decisions)
    print(f'\nChanges since run {diff["previous_run_id"]}')
    print(f'  Added: {", ".join(diff["added"]) or "none"} | Dropped: {", ".join(diff["dropped"]) or "none"}')
    print("  Rank changes: " + (", ".join(f"{s} {a} -> {b}" for s, a, b in diff["rank_changes"]) or "none"))
    for e in diff["edits"]:
        before, now = ("not ranked" if e[k] is None else e[k] for k in ("before", "now"))
        match = " (matches reviewer)" if e["now"] == e["reviewer"] else ""
        print(f'  Reviewer edit: {e["symbol"]} {e["criterion"]}: before {before}, reviewer {e["reviewer"]}, '
              f'now {now}{match}')
