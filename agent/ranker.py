"""Ranking agent: runs the shared loop with the ranker's prompt and tools, then checks, saves and prints the result.

Objective:
    One call (or one command) ranks candidate targets for a condition:
    python -m agent.ranker "atopic dermatitis" --n 5

Contents:
    rank_targets()   condition, n -> runs the agent, validates its answer, applies the citation rule,
                     saves runs/<run_id>.json and returns the saved record
    print_run()      prints the ranked table plus tool calls, cost, runtime and stop reason

Rationale:
    - The LLM chooses the tools (via loop.run_agent); this module only sets up the run and checks the answer.
    - The answer is validated with RankingResult, and enforce_citations is applied in code before saving,
      so a saved run never contains an uncited non-zero score.
    - Each run is saved with its model, prompt version and usage, so runs can be compared later (Day 3).
    - Past reviewer decisions come from learn.summarize_feedback() and are passed as context, not evidence.
"""

import json
import os
import re
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from agent.critic import critique
from agent.export import export_tsv
from agent.learn import print_comparison, print_review_queue, summarize_feedback
from agent.loop import make_client, run_agent, schema_validator
from agent.prompts import RANKER_PROMPT_VERSION, RANKER_SYSTEM_PROMPT, ranker_user_prompt
from agent.rubric import enforce_citations
from agent.schemas import RankingResult
from agent.tools.registry import TOOLS, run_tool


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
    feedback = summarize_feedback(condition)

    agent_result = run_agent(
        client=make_client(), agent="ranker", model=model,
        system_prompt=RANKER_SYSTEM_PROMPT, user_prompt=ranker_user_prompt(condition, n, feedback),
        tools=TOOLS, run_tool=run_tool, submit_tool="submit_ranking", validate=schema_validator(RankingResult),
        reasoning_effort=os.getenv("RANKER_REASONING_EFFORT") or None,
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
    _save_run(run)
    return run


# --- C5a: ranker then critic, one saved run ---
# _save_run() writes runs/<run_id>.json (also used by rank_targets, so a ranker-only result is on disk even if
# the critic fails). run_pipeline() runs the ranker, then the critic on its result, and saves both:
#   ranker_result = the ranker's answer after the citation rule;  result = after checks and critic verdicts
# In:  run_pipeline("atopic dermatitis", n=5)
# Out: {...rank_targets fields..., "ranker_result": {...}, "result": {...final...},
#       "critic": {"model": ..., "checks": [...], "report": {...}, "usage": {...}, ...}}

def _save_run(run: dict) -> None:
    Path("runs").mkdir(exist_ok=True)
    Path(f'runs/{run["run_id"]}.json').write_text(json.dumps(run, indent=2))


def run_pipeline(condition: str, n: int = 5, max_tool_calls: int = 50) -> dict:
    run = rank_targets(condition, n=n, max_tool_calls=max_tool_calls)
    run["ranker_result"] = run["result"]
    run["critic"] = None
    if run["result"]:
        ranking = RankingResult.model_validate(run["result"])
        run["critic"] = critique(ranking, run["run_id"])
        run["result"] = run["critic"].pop("result")
    _save_run(run)
    return run


# --- C5b: provenance summary ---
# Printed under the ranked table: how many deterministic checks passed (grouped by kind), the critic's verdict
# counts, and tool calls, cost and runtime per agent plus the total. Runs without a critic say so.
# In:  _print_provenance(run)
# Out: Provenance summary
#        Deterministic checks: targets 5/5, PMID citations 19/19, record citations 16/16, scores within record 13/13
#        Annotations: 3 multi-target drug citation(s) (see flags)
#        Critic verdicts: 24 supported, 5 partial, 1 unsupported; 1 confidence flag(s)
#        ranker   20 tool calls | $0.010 |  43.2 s | stop: submitted (anthropic/claude-haiku-5.5, ranker-v1)
#        critic    6 tool calls | $0.004 |  25.0 s | stop: submitted (google/gemini-3.8-flash, critic-v1)
#        total    26 tool calls | $0.014 |  68.2 s

CHECK_GROUPS = {"target_matches": "targets", "pmid_exists": "PMID citations", "pmid_mentions_target": "PMID citations",
                "record_matches": "record citations", "score_within_record": "scores within record"}
NOTE_LABELS = {"drug_multi_target": "multi-target drug citation(s)"}


def _usage_line(name: str, usage: dict, extra: str = "") -> str:
    return (f'  {name:<7}{usage["tool_calls"]:>3} tool calls | ${usage["cost"]:.3f} | '
            f'{usage["runtime_s"]:>5} s{extra}')


def _print_provenance(run: dict) -> None:
    critic = run.get("critic")
    print("\nProvenance summary")
    if critic:
        groups = {}
        for check in [c for c in critic["checks"] if not c.get("annotation")]:
            counts = groups.setdefault(CHECK_GROUPS.get(check["check"], check["check"]), [0, 0])
            counts[0] += check["passed"]
            counts[1] += 1
        print("  Deterministic checks: " + ", ".join(f"{name} {p}/{t}" for name, (p, t) in groups.items()))
        notes = Counter(c["check"] for c in critic["checks"] if c.get("annotation"))
        if notes:
            print("  Annotations: " + ", ".join(f"{n} {NOTE_LABELS.get(k, k)}" for k, n in notes.items())
                  + " (see flags)")
        if critic["report"]:
            verdicts = Counter(v["verdict"] for v in critic["report"]["verdicts"])
            print(f'  Critic verdicts: {verdicts["supported"]} supported, {verdicts["partial"]} partial, '
                  f'{verdicts["unsupported"]} unsupported; '
                  f'{len(critic["report"]["confidence_flags"])} confidence flag(s)')
        else:
            print(f'  Critic verdicts: none ({critic["error"]})')
    else:
        print("  Critic: not run" + (" (no valid ranking to review)" if not run["result"] else ""))

    agents = [("ranker", run["usage"], f' | stop: {run["stop_reason"]} ({run["model"]}, {run["prompt_version"]})')]
    if critic:
        agents.append(("critic", critic["usage"],
                       f' | stop: {critic["stop_reason"]} ({critic["model"]}, {critic["prompt_version"]})'))
    for name, usage, extra in agents:
        print(_usage_line(name, usage, extra))
    total = {k: sum(u[k] for _, u, _ in agents) for k in ("tool_calls", "cost", "runtime_s")}
    total["runtime_s"] = round(total["runtime_s"], 1)
    print(_usage_line("total", total))


# --- B3c: print the run and command line ---
# print_run() shows the ranked table, then usage and how the run ended. Flags (e.g. uncited scores zeroed) are
# listed under each candidate. The command line runs run_pipeline() (ranker, then critic) and prints the result.
# In:  python -m agent.ranker "atopic dermatitis" --n 5
# Out: Run 20261101-103000-atopic-dermatitis  (ranker-v1, anthropic/claude-haiku-5.5)
#      rank  target    total  confidence  novelty
#         1  IL4R         14  high        known
#      ...
#      Provenance summary ... (see C5b)

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
    _print_provenance(run)
    if run["partial"]:
        print("PARTIAL: a limit was hit before the agent finished; results use the evidence gathered so far.")
    print(f'Saved runs/{run["run_id"]}.json and traces/{run["run_id"]}.jsonl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank candidate drug targets for a condition.")
    parser.add_argument("condition", help='e.g. "atopic dermatitis"')
    parser.add_argument("--n", type=int, default=5, help="number of candidates (default 5)")
    parser.add_argument("--max-tool-calls", type=int, default=50)
    args = parser.parse_args()
    run = run_pipeline(args.condition, n=args.n, max_tool_calls=args.max_tool_calls)
    print_run(run)
    print_review_queue(run)
    print_comparison(run)
    for path in export_tsv(run):
        print(f"Saved {path} (tab-delimited, opens in Excel)")