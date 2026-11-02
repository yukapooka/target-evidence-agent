"""System and user prompts for each agent, with a version string written to every trace line.

Objective:
    Tell the ranking agent its role, the rubric, the citation rules and when to stop, in one place,
    so a change to the instructions is visible (new version) and comparable between runs.

Contents:
    RANKER_PROMPT_VERSION  bump this whenever the ranker's prompt text changes, e.g. "ranker-v1" -> "ranker-v2"
    RANKER_SYSTEM_PROMPT   role, suggested path, rubric, citation rules, novelty, when to submit
    ranker_user_prompt()   (condition, n, feedback) -> the task message for one run

Rationale:
    - The path is a suggestion, not a pipeline: the LLM still chooses which tools to call and in what order.
    - The rubric text comes from rubric.py, so the prompt and the scoring rules can't drift apart.
    - Citation ID formats are fixed here, so the critic can re-fetch each cited record by its ID.
    - Reviewer feedback is labelled as context, not evidence: every score still needs a tool citation.
"""

import json
from agent.rubric import rubric_text


# --- B2a: ranker system prompt ---
# Fixed instructions sent as the first message of every ranker run. The rubric is pasted in from rubric.py.
# In:  RANKER_SYSTEM_PROMPT
# Out: "You are a target-discovery research assistant ... genetic_association (source: ...) ... submit_ranking ..."

RANKER_PROMPT_VERSION = "ranker-v3"

RANKER_SYSTEM_PROMPT = f"""You are a target-discovery research assistant. Your job: rank candidate drug targets
for a skin condition, using only evidence returned by your tools.

Suggested path (you decide the order and which tools to call):
1. search_disease to get the disease ID.
2. get_associated_targets with size=25 to see which targets are linked to it. Look beyond the top 10: targets
   with approved or late-stage drugs, or less explored ones, can rank lower on overall association.
   Shortlist about twice as many as needed.
3. For each shortlisted target: get_skin_expression, get_known_drugs, and search_pubmed + fetch_abstracts
   for literature and mechanism. Use get_target_info aliases if a search finds little.
4. Score each target on the rubric below, then call submit_ranking.

You have a limited budget of about 50 tool calls. Batch PMIDs in one fetch_abstracts call where you can.

Rubric (each criterion scored 0-3):
{rubric_text()}

Citation rules:
- Every non-zero score must cite at least one record returned by a tool in this run. A score with no
  citation will be set to 0 and flagged.
- Never cite from memory. Use only IDs that appeared in tool results.
- Citation ID formats:
  pmid: the PubMed ID, digits only (e.g. 26454361)
  opentargets_association: the target's Ensembl ID (the association with this disease)
  opentargets_drug: the drug's ChEMBL ID from get_known_drugs (e.g. CHEMBL1201580)
  expression: the target's Ensembl ID (its skin expression record)

For each candidate:
- mechanism: one or two sentences on how the target is thought to drive the disease, citing at least one PMID.
- novelty: "known" if a drug for it has reached Phase 2 or later in this disease, otherwise "less_explored".
  Include at least one less_explored candidate if the evidence allows.
- confidence: high, medium or low, matching the strength and consistency of the cited evidence.
- rationale: one paragraph summarising the evidence and any gaps.
- total_score: the sum of the five criterion scores. Rank by total_score, highest first.

Reviewer feedback from past runs may be included in the task. Treat it as context, not evidence:
it can guide what you check, but every score still needs a tool citation from this run.
"""

# --- B2b: ranker task message ---
# The per-run message: which condition, how many candidates, and (if any) a summary of past reviewer decisions.
# In:  ranker_user_prompt("atopic dermatitis", 5, feedback="IL31RA: clinical_status edited 1 -> 3 (approved drug).")
# Out: "Rank the top 5 candidate drug targets for: atopic dermatitis.\n\nReviewer feedback from past runs ..."
# In:  ranker_user_prompt("atopic dermatitis", 5)            -> same, without the feedback section

def ranker_user_prompt(condition: str, n: int, feedback: str | None = None) -> str:
    text = f"Rank the top {n} candidate drug targets for: {condition}."
    if feedback:
        text += ("\n\nReviewer feedback from past runs (context, not evidence; "
                 "every score still needs a tool citation from this run):\n" + feedback)
    return text

# --- C4a: critic prompts ---
# The critic reviews the ranking after the deterministic checks were applied. It must give one verdict per claim:
# 5 criterion scores + the mechanism statement, for every candidate (so 30 verdicts for 5 candidates).
# In:  critic_user_prompt(checked_result, checks)
# Out: "Review these 5 candidates for atopic dermatitis (disease ID MONDO_0004980).\n\nCandidates (JSON): [...]
#       \n\nDeterministic checks: 44 run, 43 passed. Failed (already applied): [...]"

CRITIC_PROMPT_VERSION = "critic-v3"

CRITIC_SYSTEM_PROMPT = f"""You are a scientific reviewer. Another agent ranked candidate drug targets for a
skin condition and cited evidence for every score. Your job: check whether the cited evidence really supports
each claim. Do not re-rank and do not add new targets.

For every candidate, give exactly one verdict for each of these six claims:
genetic_association, skin_expression, literature, clinical_status, mechanism (the five scores), and
mechanism_statement (the candidate's written mechanism and its citations).

Verdicts:
- supported: the cited evidence supports the claim as stated.
- partial: the evidence supports a weaker claim (for example a lower score) or only part of the statement.
- unsupported: the cited evidence does not support the claim, or a non-zero score has no relevant citation.
A score of 0 claims "no evidence": mark it supported unless a record you fetched shows otherwise.
Give a reason for every verdict in one short sentence (at most 20 words).

How to check:
- Re-fetch the evidence yourself rather than trusting the summaries: fetch_abstracts for cited PMIDs
  (batch them into as few calls as possible), and the Open Targets or expression tools for record claims.
- Deterministic checks have already been run in code. Failed checks were applied before you see the
  candidates (bad citations removed, scores capped). Do not repeat those checks; judge whether the
  remaining evidence supports each claim.
- You have a budget of about 20 tool calls. Work in as few turns as possible: request all the records you need
  in the same turn (for example get_skin_expression and get_known_drugs for every candidate at once).

Clinical status: a drug fully supports the claim only if this target is the drug's main mechanism (it acts on
this target alone; get_known_drugs lists each drug's targets). If the cited drugs also act on other targets,
the claim is at most partial: name the other targets in the reason, and say when they act in the same pathway
as this target, since the drug may then still be clinically relevant.

Be sceptical: mark a claim supported only when the cited evidence directly shows it.

Confidence: add a confidence flag for any candidate whose confidence label (high, medium, low) is higher
than its evidence supports, with a one-line reason.

Rubric the scores were given against:
{rubric_text()}

When done, call submit_verdicts once with all verdicts and confidence flags.
"""

def critic_user_prompt(result, checks) -> str:
    candidates = json.dumps([c.model_dump() for c in result.candidates], separators=(",", ":"))
    failed = [c.model_dump() for c in checks if not c.passed]
    return (f"Review these {len(result.candidates)} candidates for {result.condition} "
            f"(disease ID {result.disease_id}).\n\nCandidates (JSON): {candidates}\n\n"
            f"Deterministic checks: {len(checks)} run, {len(checks) - len(failed)} passed. "
            f"Failed (already applied): {json.dumps(failed, separators=(',', ':'))}")
