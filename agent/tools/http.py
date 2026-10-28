from datetime import timedelta

import requests_cache
from dotenv import load_dotenv

load_dotenv()

def _is_cacheable(response) -> bool:
    """Skip caching GraphQL error bodies: Open Targets returns HTTP 200 with an "errors" key."""
    if "application/json" not in response.headers.get("Content-Type", ""):
        return True
    try:
        body = response.json()
    except ValueError:
        return True
    return not (isinstance(body, dict) and "errors" in body)

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

session = make_session()

OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
TIMEOUT = 30


def graphql(query: str, variables: dict | None = None) -> dict:
    """POST a query to Open Targets and return the "data" part, raising on GraphQL errors."""
    response = session.post(OT_GRAPHQL_URL, json={"query": query, "variables": variables or {}}, timeout=TIMEOUT)
    if response.status_code >= 500:
        response.raise_for_status()
    body = response.json()
    if body.get("errors"):
        raise RuntimeError("Open Targets GraphQL error: " + "; ".join(e["message"] for e in body["errors"]))
    return body["data"]


def get_json(url: str, params: dict) -> dict:
    """GET a JSON endpoint (used for PubMed esearch and esummary)."""
    response = session.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def get_text(url: str, params: dict) -> str:
    """GET a text endpoint (used for PubMed efetch, which returns XML)."""
    response = session.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.text
