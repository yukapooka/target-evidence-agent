"""Shared HTTP layer: the only module in the project that talks to the internet.

Objective:
    Give every tool wrapper one simple, cached way to call external APIs
    (Open Targets GraphQL and NCBI E-utilities for PubMed).

Functions:
    make_session()  builds a requests-cache session (GET and POST cached for 7 days)
    graphql()       sends a query to Open Targets; returns the "data" part or raises a readable error
    get_json()      GET returning JSON (PubMed esearch, esummary)
    get_text()      GET returning text (PubMed efetch, which only returns XML)

Rationale:
    - Caching makes reruns fast and free, and lets agent runs be repeated cheaply.
      POST is cached too, because GraphQL uses POST; the request body is part of the cache key.
    - Open Targets can answer HTTP 200 with an "errors" key, so a filter keeps those out of the cache.
    - api_key/email/tool are excluded from the cache, so the NCBI key is never written to disk.
    - Wrappers call only these helpers, so caching, timeouts and error handling live in one place.
"""

from datetime import timedelta

import requests_cache
from dotenv import load_dotenv

# Read .env (NCBI_API_KEY etc.) once, so any module can use os.getenv() afterwards.
load_dotenv()


# --- Step 1b: cache filter ---
# requests-cache calls this after every real network response: True = save it, False = don't.
# Open Targets can return HTTP 200 with an "errors" key; those must not be saved.
# In:  response with body {"data": {...}}                -> Out: True  (save)
# In:  response with body {"data": null, "errors": [...]} -> Out: False (don't save)
# In:  PubMed XML response (not JSON)                     -> Out: True  (save)

def _is_cacheable(response) -> bool:
    """Skip caching GraphQL error bodies: Open Targets returns HTTP 200 with an "errors" key."""
    if "application/json" not in response.headers.get("Content-Type", ""):
        return True
    try:
        body = response.json()
    except ValueError:
        return True
    return not (isinstance(body, dict) and "errors" in body)


# --- Step 1a: cached session ---
# Builds a requests session that stores answers (GET and POST) for 7 days.
# api_key/email/tool are left out of the cache key and the saved file, so the NCBI key is never stored.
# In:  make_session()                  -> Out: session saving to .cache/http_cache.sqlite (real runs)
# In:  make_session(backend="memory")  -> Out: same settings, kept in RAM (tests)

def make_session(cache_name: str = ".cache/http_cache", backend: str ="sqlite") -> requests_cache.CachedSession:
    """Build an HTTP session that caches responses (including POST) so reruns are fast and offline-friendly."""
    return requests_cache.CachedSession(
        cache_name=cache_name,
        backend=backend,
        allowable_methods=("GET", "POST"),
        expire_after=timedelta(days=7),
        ignored_parameters=["api_key", "email", "tool"],
        filter_fn=_is_cacheable,
    )

# One shared session for the whole app, so all wrappers share one cache.
session = make_session()

OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
TIMEOUT = 30  # seconds; requests has no default timeout, so a stalled server would hang forever


# --- Step 1c: request helpers (the only functions the wrappers call) ---

# Sends a GraphQL query to Open Targets and returns just the "data" part.
# In:  graphql("{ meta { apiVersion { x } } }")
# Out: {"meta": {"apiVersion": {"x": "26"}}}
# On error: raises RuntimeError("Open Targets GraphQL error: Cannot query field ...")

def graphql(query: str, variables: dict | None = None) -> dict:
    """POST a query to Open Targets and return the "data" part, raising on GraphQL errors."""
    response = session.post(OT_GRAPHQL_URL, json={"query": query, "variables": variables or {}}, timeout=TIMEOUT)
    # 5xx bodies are often HTML, so fail on status; 4xx bodies are JSON with a useful message, read below.
    if response.status_code >= 500:
        response.raise_for_status()
    body = response.json()
    if body.get("errors"):
        raise RuntimeError("Open Targets GraphQL error: " + "; ".join(e["message"] for e in body["errors"]))
    return body["data"]


# GET request that returns parsed JSON (PubMed esearch, esummary).
# In:  get_json(".../esearch.fcgi", {"db": "pubmed", "term": "IL4R", "retmode": "json"})
# Out: {"esearchresult": {"idlist": ["38186219", ...], ...}}

def get_json(url: str, params: dict) -> dict:
    """GET a JSON endpoint (used for PubMed esearch and esummary)."""
    response = session.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


# GET request that returns raw text (PubMed efetch, which only returns XML).
# In:  get_text(".../efetch.fcgi", {"db": "pubmed", "id": "31112383", "retmode": "xml"})
# Out: "<PubmedArticleSet><PubmedArticle>...</PubmedArticle></PubmedArticleSet>"

def get_text(url: str, params: dict) -> str:
    """GET a text endpoint (used for PubMed efetch, which returns XML)."""
    response = session.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.text
