"""Shared tool-calling loop: runs one agent (ranker or critic) until it submits a final answer or hits a limit.

Objective:
    Send the conversation to the LLM, run whichever tools it asks for, feed the results back, and repeat
    until the LLM calls its final "submit" tool. Both agents use this same loop with their own settings.

Contents:
    make_client()   OpenAI SDK client pointed at OpenRouter
    AgentResult     what a run returns: the submitted arguments (or None), partial flag, stop reason, usage stats
    run_agent()     the loop itself, with tool-call, cost and time limits; writes one JSONL trace line per step

Rationale:
    - The LLM, not this code, chooses which tools to call and in what order.
    - Limits stop the run cleanly with partial=True instead of raising, so a long run still reports what it found.
    - The client is a parameter, so tests can pass a fake client and run offline.
"""

import json
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from openai import OpenAI
from pydantic import ValidationError
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# --- B1a: client and result ---
# make_client() reads the key from .env (via load_dotenv) or, when deployed, the environment.
# In:  make_client()
# Out: OpenAI(base_url="https://openrouter.ai/api/v1", api_key=...)
# AgentResult is a plain dataclass: the loop fills it in and returns it.
# In:  AgentResult(final_args={"candidates": [...]}, partial=False, stop_reason="submitted", tool_calls=12, ...)

def make_client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


@dataclass
class AgentResult:
    final_args: dict | None      # arguments of the submit call; None if a limit stopped the run first
    partial: bool                # True if stopped by a limit
    stop_reason: str             # "submitted", "max_tool_calls", "max_cost", "timeout" or "no_tool_call"
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0            # USD, from OpenRouter's usage.cost
    runtime_s: float = 0.0


# --- B1b: trace writer ---
# Appends one JSON line per step to the run's trace file, with a UTC timestamp added. Long results are cut
# to a short summary so the trace stays readable.
# In:  _trace("traces/run1.jsonl", {"agent": "ranker", "type": "tool", "tool": "search_disease",
#                                   "args": {"query": "atopic dermatitis"}, "result": "[{...}]"})
# Out: line added: {"ts": "2026-11-01T10:00:00+00:00", "agent": "ranker", "type": "tool", ...}

def _trace(trace_path: str, record: dict) -> None:
    if "result" in record:
        record["result"] = str(record["result"])[:300]
    line = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "a") as f:
        f.write(json.dumps(line) + "\n")



# --- B1c: the loop ---
# One round: send the messages to the LLM, run every tool it asks for, append the results, repeat.
# Stops when the LLM calls the submit tool (final answer) or a limit is hit (partial=True, final_args=None).
# In:  run_agent(client=make_client(), agent="ranker", model="anthropic/claude-haiku-5.5",
#                system_prompt="...", user_prompt="Rank 5 targets for atopic dermatitis", tools=TOOLS,
#                run_tool=run_tool, submit_tool="submit_ranking", max_tool_calls=50, max_cost=0.50,
#                timeout_s=600, trace_path="traces/run1.jsonl", prompt_version="ranker-v1")
# Out: AgentResult(final_args={"condition": ..., "candidates": [...]}, partial=False, stop_reason="submitted",
#                  tool_calls=23, prompt_tokens=..., completion_tokens=..., cost=0.08, runtime_s=95.2)
# validate (optional): args -> error text or None. An invalid submit is sent back to the LLM to fix, up to
# max_rejections times; after that the last answer is returned as is (the caller reports the error).
# In:  run_agent(..., validate=schema_validator(RankingResult))
# reasoning_effort (optional): "low" | "medium" | "high", sent to OpenRouter as {"reasoning": {"effort": ...}} on
# every request of this agent; reasoning (thinking) tokens are billed as output, so this controls cost.
# Each LLM trace line records reasoning_tokens.
#      LLM submits {"candidates": "[{...}]"}  -> tool reply "Submission rejected: candidates: Input should be a
#      valid list ... Fix it and call submit_ranking again."  -> LLM resubmits a real list -> stop_reason="submitted"

def run_agent(*, client, agent: str, model: str, system_prompt: str, user_prompt: str, tools: list[dict],
              run_tool, submit_tool: str, max_tool_calls: int, max_cost: float, timeout_s: float,
              trace_path: str, prompt_version: str, validate=None, max_rejections: int = 2,
              reasoning_effort: str | None = None) -> AgentResult:
    result = AgentResult(final_args=None, partial=True, stop_reason="")
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    step = {"agent": agent, "model": model, "prompt_version": prompt_version}
    if reasoning_effort:
        step["reasoning_effort"] = reasoning_effort
    start = time.monotonic()
    rejections = 0
    _trace(trace_path, {**step, "type": "start", "prompt": user_prompt})   # full task message, incl. any feedback

    while not result.stop_reason:
        msg = _ask_llm(client, model, messages, tools, result, step, trace_path)

        if not msg.tool_calls:
            result.stop_reason = "no_tool_call"

        for call in msg.tool_calls or []:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None
            if name == submit_tool and args is not None:
                error = validate(args) if validate else None
                if error and rejections < max_rejections:
                    rejections += 1
                    output = f"Submission rejected: {error} Fix it and call {submit_tool} again."
                    _trace(trace_path, {**step, "type": "submit_rejected", "tool": name, "result": output})
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                    continue
                _trace(trace_path, {**step, "type": "submit", "tool": name})
                result.final_args, result.partial, result.stop_reason = args, False, "submitted"
                break
            if result.tool_calls >= max_tool_calls:
                output = "Skipped: tool-call limit reached"
            elif args is None:
                output = "Error: arguments were not valid JSON"
            else:
                output = run_tool(name, args)
                result.tool_calls += 1
            _trace(trace_path, {**step, "type": "tool", "tool": name, "args": args, "result": output})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

        if not result.stop_reason:
            if result.tool_calls >= max_tool_calls:
                result.stop_reason = "max_tool_calls"
            elif result.cost >= max_cost:
                result.stop_reason = "max_cost"
            elif time.monotonic() - start >= timeout_s:
                result.stop_reason = "timeout"

    if result.final_args is None:
        _force_submit(client, model, messages, tools, result, step, trace_path, submit_tool)
    result.runtime_s = round(time.monotonic() - start, 1)
    return result

# --- B1d: LLM call and forced final submit ---
# _ask_llm(): one LLM request; adds tokens and cost to the result and writes the trace line. Used by the loop
# and by _force_submit, so usage is counted the same way everywhere.
# In:  _ask_llm(client, model, messages, tools, result, step, trace_path)
# Out: the assistant message (msg.content text and/or msg.tool_calls)
# _force_submit(): if the run stopped without a submit (limit hit, or the LLM replied in plain text), make one
# last request that must call the submit tool, so the run still returns its best answer. partial stays True.
# In:  _force_submit(..., result=AgentResult(final_args=None, partial=True, stop_reason="max_tool_calls"))
# Out: result.final_args = {"candidates": [...]} (or still None if the model could not comply)

def _ask_llm(client, model, messages, tools, result, step, trace_path, tool_choice="auto"):
    effort = step.get("reasoning_effort")
    response = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice=tool_choice,
                                              extra_body={"reasoning": {"effort": effort}} if effort else None)
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)
    cost = getattr(usage, "cost", 0.0) or 0.0
    result.prompt_tokens += usage.prompt_tokens
    result.completion_tokens += usage.completion_tokens
    result.cost += cost
    msg = response.choices[0].message
    _trace(trace_path, {**step, "type": "llm", "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "reasoning_tokens": getattr(details, "reasoning_tokens", None), "cost": cost,
                        "result": msg.content or ""})
    messages.append(msg.model_dump(exclude_none=True))
    return msg


def _force_submit(client, model, messages, tools, result, step, trace_path, submit_tool):
    messages.append({"role": "user", "content": f"Stopped: {result.stop_reason}. Call {submit_tool} now "
                     "with your best answer from the evidence gathered so far. Do not call other tools."})
    msg = _ask_llm(client, model, messages, tools, result, step, trace_path,
                   tool_choice={"type": "function", "function": {"name": submit_tool}})
    for call in msg.tool_calls or []:
        if call.function.name == submit_tool:
            try:
                result.final_args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                return
            _trace(trace_path, {**step, "type": "submit", "tool": submit_tool, "forced": True})
            return



# --- C5c: submit validation ---
# Builds a validate function for run_agent from a Pydantic model. Errors are kept short (field path + message, at
# most 5) because the LLM's broken answer can be thousands of characters long and shouldn't be echoed back.
# In:  schema_validator(RankingResult)({"condition": "ad", "disease_id": "x", "candidates": "[...]"})
# Out: "candidates: Input should be a valid list. Send nested fields as JSON arrays and objects, not as strings."
# In:  schema_validator(RankingResult)(valid_args)   -> None

def schema_validator(model):
    def validate(args: dict) -> str | None:
        try:
            model.model_validate(args)
            return None
        except ValidationError as e:
            problems = "; ".join(f'{".".join(str(p) for p in err["loc"])}: {err["msg"]}' for err in e.errors()[:5])
            return f"{problems}. Send nested fields as JSON arrays and objects, not as strings."
    return validate
