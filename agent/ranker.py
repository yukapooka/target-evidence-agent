"""Ranking agent: runs the shared loop with the ranker's prompt and tools, then checks, saves and prints the result.

Objective:
    One call (or one command) ranks candidate targets for a condition:
    python -m agent.ranker "atopic dermatitis" --n 5

Contents:
    load_feedback()  condition -> short text summary of past reviewer decisions from review/decisions.json, or None
    rank_targets()   condition, n -> runs the agent, validates its answer, applies the citation rule,
                     saves runs/<run_id>.json and returns the saved record
    print_run()      prints the ranked table plus tool calls, cost, runtime and stop reason

Rationale:
    - The LLM chooses the tools (via loop.run_agent); this module only sets up the run and checks the answer.
    - The answer is validated with RankingResult, and enforce_citations is applied in code before saving,
      so a saved run never contains an uncited non-zero score.
    - Each run is saved with its model, prompt version and usage, so runs can be compared later (Day 3).
    - Feedback is a simple summary for now; Day 3's learn.py replaces it with a fuller one.
"""

import json
import os
import re
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from agent.loop import make_client, run_agent
from agent.prompts import RANKER_PROMPT_VERSION, RANKER_SYSTEM_PROMPT, ranker_user_prompt
from agent.rubric import enforce_citations
from agent.schemas import RankingResult
from agent.tools.registry import TOOLS, run_tool


DECISIONS_PATH = Path("review/decisions.json")


# --- B3a: past reviewer decisions ---
# Reads review/decisions.json (written by hand, Day 3) and keeps the entries for this condition, one line each.
# File format: [{"condition": "atopic dermatitis", "symbol": "IL31RA", "decision": "approve",
#                "edited_scores": {"clinical_status": 3}, "comment": "approved drug nemolizumab"}, ...]
# In:  load_feedback("atopic dermatitis")
# Out: "IL31RA: approve; edited scores clinical_status=3; comment: approved drug nemolizumab"
# In:  load_feedback("psoriasis")  -> None   (no file, or no entries for this condition)

def load_feedback(condition: str) -> str | None:
    if not DECISIONS_PATH.exists():
        return None
    decisions = json.loads(DECISIONS_PATH.read_text())
    lines = []
    for d in decisions:
        if d["condition"].lower() != condition.lower():
            continue
        line = f'{d["symbol"]}: {d["decision"]}'
        if d.get("edited_scores"):
            edits = ", ".join(f"{k}={v}" for k, v in d["edited_scores"].items())
            line += f"; edited scores {edits}"
        if d.get("comment"):
            line += f'; comment: {d["comment"]}'
        lines.append(line)
    return "\n".join(lines) or None

# --- B3b: run the ranking agent ---
# Builds the prompts, runs the loop, validates the submitted answer, applies the citation rule, and saves the run.
# If the answer is missing or fails validation, the run is still saved, with "error" set and no candidates.
# In:  rank_targets("atopic dermatitis", n=5)
# Out: {"run_id": "20261101-103000-atopic-dermatitis", "condition": "atopic dermatitis", "model": "...",
#       "prompt_version": "ranker-v1", "stop_reason": "submitted", "partial": False, "error": None,
#       "usage": {"tool_calls": 23, "cost": 0.08, ...}, "result": {"disease_id": ..., "candidates": [...]}}
#      and files runs/<run_id>.json, traces/<run_id>.jsonl

USAGE_FIELDS = ("tool_calls", "prompt_tokens", "completion_tokens", "cost", "runtime_s")


def rank_targets(condition: str, n: int = 5, max_tool_calls: int = 50, timeout_s: float = 900) -> dict:
    model = os.environ["RANKER_MODEL"]
    slug = re.sub(r"[^a-z0-9]+", "-", condition.lower()).strip("-")
    run_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{slug}"
    feedback = load_feedback(condition)

    agent_result = run_agent(
        client=make_client(), agent="ranker", model=model,
        system_prompt=RANKER_SYSTEM_PROMPT, user_prompt=ranker_user_prompt(condition, n, feedback),
        tools=TOOLS, run_tool=run_tool, submit_tool="submit_ranking",
        max_tool_calls=max_tool_calls, max_cost=float(os.getenv("MAX_COST_PER_RUN_USD", "0.50")),
        timeout_s=timeout_s, trace_path=f"traces/{run_id}.jsonl", prompt_version=RANKER_PROMPT_VERSION)

    result, error = None, None
    if agent_result.final_args is None:
        error = "no answer submitted"
    else:
        try:
            ranking = RankingResult.model_validate(agent_result.final_args)
            ranking.partial = agent_result.partial
            result = enforce_citations(ranking).model_dump()
        except ValidationError as e:
            error = f"answer failed validation: {e}"

    usage = {k: v for k, v in asdict(agent_result).items() if k in USAGE_FIELDS}
    run = {"run_id": run_id, "condition": condition, "n": n, "model": model,
           "prompt_version": RANKER_PROMPT_VERSION, "feedback": feedback,
           "stop_reason": agent_result.stop_reason, "partial": agent_result.partial, "error": error,
           "usage": usage, "result": result, "raw_answer": agent_result.final_args if error else None}
    Path("runs").mkdir(exist_ok=True)
    Path(f"runs/{run_id}.json").write_text(json.dumps(run, indent=2))
    return run

# --- B3c: print the run and command line ---
# print_run() shows the ranked table, then usage and how the run ended. Flags (e.g. uncited scores zeroed) are
# listed under each candidate. The command line runs rank_targets() and prints the result.
# In:  python -m agent.ranker "atopic dermatitis" --n 5
# Out: Run 20261101-103000-atopic-dermatitis  (ranker-v1, anthropic/claude-haiku-5.5)
#      rank  target    total  confidence  novelty
#         1  IL4R         14  high        known
#      ...
#      Tool calls 23 | cost $0.081 | 95.2 s | stop: submitted

def print_run(run: dict) -> None:
    print(f'\nRun {run["run_id"]}  ({run["prompt_version"]}, {run["model"]})')
    if run["feedback"]:
        print("Feedback used:\n  " + run["feedback"].replace("\n", "\n  "))
    if run["error"]:
        print(f'ERROR: {run["error"]}')
    elif run["result"]:
        print(f'{"rank":>4}  {"target":<9} {"total":>5}  {"confidence":<10}  novelty')
        for c in run["result"]["candidates"]:
            print(f'{c["rank"]:>4}  {c["symbol"]:<9} {c["total_score"]:>5}  {c["confidence"]:<10}  {c["novelty"]}')
            for flag in c["flags"]:
                print(f"        flag: {flag}")
    u = run["usage"]
    print(f'Tool calls {u["tool_calls"]} | cost ${u["cost"]:.3f} | {u["runtime_s"]} s | stop: {run["stop_reason"]}')
    if run["partial"]:
        print("PARTIAL: a limit was hit before the agent finished; results use the evidence gathered so far.")
    print(f'Saved runs/{run["run_id"]}.json and traces/{run["run_id"]}.jsonl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank candidate drug targets for a condition.")
    parser.add_argument("condition", help='e.g. "atopic dermatitis"')
    parser.add_argument("--n", type=int, default=5, help="number of candidates (default 5)")
    parser.add_argument("--max-tool-calls", type=int, default=50)
    args = parser.parse_args()
    print_run(rank_targets(args.condition, n=args.n, max_tool_calls=args.max_tool_calls))