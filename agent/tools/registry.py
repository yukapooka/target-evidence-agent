"""Tool registry: what the LLM is told about each tool, and how a tool call becomes a Python call.

Objective:
    Expose the Day 1 wrappers to the ranking agent in OpenAI tool-calling format, plus a submit_ranking
    tool for the final answer, and run whichever tool the LLM picks.

Contents:
    TOOLS        tool definitions sent to the LLM on every request (name, description, JSON-schema parameters)
    FUNCTIONS    tool name -> Python wrapper function
    run_tool()   (name, args) -> compact JSON text for the LLM; errors come back as text, not exceptions

Rationale:
    - Descriptions are what the LLM reads to choose a tool, so each says when to use it and what it returns.
    - "maximum" limits on sizes keep tool results small, which keeps cost and context down.
    - submit_ranking's parameters come from RankingResult.model_json_schema(), so the schema lives in one place.
"""
import json
from agent.schemas import CriticReport, RankingResult
from agent.tools import expression, opentargets, pubmed
from pydantic_core import to_jsonable_python

# --- A4a: tool definitions ---
# _tool() wraps the parts that differ into the OpenAI format; every tool takes an object of named arguments.
# In:  _tool("get_target_info", "Approved symbol, name and aliases ...", {"ensembl_id": ENSEMBL_ID}, ["ensembl_id"])
# Out: {"type": "function", "function": {"name": "get_target_info", "description": "...",
#       "parameters": {"type": "object", "properties": {"ensembl_id": {...}}, "required": ["ensembl_id"]}}}

def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


# Pydantic writes nested models as "$ref" links; some models then send nested lists as JSON strings,
# so the submit schema is given with every "$ref" replaced by the definition it points to.
# In:  {"$defs": {"Citation": {...}}, "properties": {"citations": {"items": {"$ref": "#/$defs/Citation"}}}}
# Out: {"properties": {"citations": {"items": {...Citation schema...}}}}

def _inline_refs(schema: dict) -> dict:
    defs = schema.get("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


ENSEMBL_ID ={"type": "string", "description": "Ensembl gene ID from a tool result, e.g. ENSG00000077238"}
DISEASE_ID = {"type": "string", "description": "Open Targets disease ID from search_disease, e.g. MONDO_0004980"}
PMID_LIST = {"type": "array", "items": {"type": "string"}, "maxItems": 10,
             "description": "PubMed IDs (digits only), at most 10 per call"}

TOOLS = [
    _tool("search_disease",
          "Find a disease by name. Returns matching diseases with their Open Targets IDs. Call this first.",
          {"query": {"type": "string", "description": "Disease name, e.g. 'atopic dermatitis'"},
           "size": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}},
          ["query"]),
    _tool("get_associated_targets",
          "Top targets associated with a disease, sorted by overall score, with a score per evidence type "
          "(genetic_association, literature, clinical, ...). Use for the genetic_association criterion.",
          {"disease_id": DISEASE_ID,
           "size": {"type": "integer", "minimum": 1, "maximum": 50, "default": 25}},
          ["disease_id"]),
    _tool("get_known_drugs",
          "Drugs acting on a target that have been tested in this disease, with the highest clinical stage "
          "reached for this disease. Empty list = no drug tested. Use for the clinical_status criterion.",
          {"ensembl_id": ENSEMBL_ID, "disease_id": DISEASE_ID},
          ["ensembl_id", "disease_id"]),
    _tool("get_target_info",
          "Approved symbol, full name and aliases of a target. Use aliases to build PubMed searches.",
          {"ensembl_id": ENSEMBL_ID},
          ["ensembl_id"]),
    _tool("get_skin_expression",
          "GTEx expression (median TPM) in the two normal skin tissues, and where skin ranks among all GTEx "
          "tissues (1 = highest). Use for the skin_expression criterion.",
          {"ensembl_id": ENSEMBL_ID},
          ["ensembl_id"]),
    _tool("search_pubmed",
          "Search PubMed. Returns PMIDs, most relevant first. Follow with fetch_abstracts to read them.",
          {"query": {"type": "string", "description": "PubMed query, e.g. 'IL4R atopic dermatitis'"},
           "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}},
          ["query"]),
    _tool("fetch_abstracts",
          "Title, journal, year and abstract (trimmed) for each PMID. Use for literature and mechanism evidence.",
          {"pmids": PMID_LIST},
          ["pmids"]),
    _tool("check_pmids",
          "Check that PMIDs exist in PubMed. Returns {pmid: true/false}.",
          {"pmids": PMID_LIST},
          ["pmids"]),
    {
        "type": "function",
        "function": {
            "name": "submit_ranking",
            "description": "Submit the final ranked candidates. Call once, when all scoring is done. "
                           "Every non-zero score must cite a record returned by a tool.",
            "parameters": _inline_refs(RankingResult.model_json_schema()),
        },
    },
]


# --- C4b: critic tool set ---
# The critic gets read-only tools to re-fetch cited evidence (no searching for new evidence), plus its own
# submit tool, whose parameters come from CriticReport.
# In:  [t["function"]["name"] for t in CRITIC_TOOLS]
# Out: ["get_associated_targets", "get_known_drugs", "get_target_info", "get_skin_expression",
#       "fetch_abstracts", "submit_verdicts"]

CRITIC_TOOL_NAMES = {"get_associated_targets", "get_known_drugs", "get_target_info",
                     "get_skin_expression", "fetch_abstracts"}

CRITIC_TOOLS = [t for t in TOOLS if t["function"]["name"] in CRITIC_TOOL_NAMES] + [
    {
        "type": "function",
        "function": {
            "name": "submit_verdicts",
            "description": "Submit all verdicts (one per claim: five criteria plus mechanism_statement, for every "
                           "candidate) and any confidence flags. Call once, when the review is done.",
            "parameters": _inline_refs(CriticReport.model_json_schema()),
        },
    },
]


# --- A4c: name -> function ---
# The LLM only sends a tool name as text; this maps it to the Python function to call.
# submit_ranking has no function: the loop stops when the LLM calls it.
# In:  FUNCTIONS["get_target_info"]("ENSG00000077238")
# Out: TargetInfo(ensembl_id="ENSG00000077238", symbol="IL4R", ...)

FUNCTIONS = {
    "search_disease": opentargets.search_disease,
    "get_associated_targets": opentargets.get_associated_targets,
    "get_target_info": opentargets.get_target_info,
    "get_known_drugs": opentargets.get_known_drugs,
    "get_skin_expression": expression.get_skin_expression,
    "search_pubmed": pubmed.search_pubmed,
    "fetch_abstracts": pubmed.fetch_abstracts,
    "check_pmids": pubmed.check_pmids,
}

# --- A4d: run a tool call ---
# Calls the wrapper the LLM picked and returns compact JSON text. Errors come back as text so the LLM can
# read them and try again (for example after a bad ID) instead of crashing the run.
# In:  run_tool("check_pmids", {"pmids": ["31227409"]})
# Out: '{"31227409":true}'
# In:  run_tool("get_target_info", {"ensembl_id": "ENSG_BAD"})
# Out: 'Error: ValueError: Unknown target ID: ENSG_BAD'

def run_tool(name: str, args: dict) -> str:
    if name not in FUNCTIONS:
        return f"Error: unknown tool {name}"
    try:
        result = FUNCTIONS[name](**args)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    return json.dumps(to_jsonable_python(result), separators=(",", ":"))