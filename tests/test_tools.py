"""Offline tests for the HTTP layer and the tool wrappers.

Objective:
    Check that caching and each wrapper's parsing work, without any internet access.

How:
    - Cache tests: a FakeAdapter replaces the network, and counts how often it is called.
    - Wrapper tests: monkeypatch replaces graphql() with a function returning a saved real
      API answer from tests/fixtures/ (recorded once from the live API).

Rationale:
    - Tests run fast, cost nothing and give the same result every time.
    - Each test fakes the layer just below the code it checks, so the real logic still runs.
"""

import json
from pathlib import Path

import pytest
import requests
from requests.adapters import BaseAdapter
from urllib3 import HTTPResponse

from agent.tools import expression, opentargets
from agent.tools.http import make_session


FAKE_URL = "https://fake.test/graphql"


# --- Step 1d: cache tests ---
# FakeAdapter replaces the network: every request gets the same 200 JSON answer, and it counts calls.
# In:  FakeAdapter({"data": {...}})  -> each session.post(...) returns that body with status 200

class FakeAdapter(BaseAdapter):
    """Stands in for the network: every request gets the same 200 JSON response."""

    def __init__(self, body: dict):
        super().__init__()
        self.body = body
        self.calls = 0

    def send(self, request, **kwargs):
        self.calls += 1
        response = requests.Response()
        response.status_code = 200
        response.reason = "OK"
        response.headers["Content-Type"] = "application/json"
        response._content = json.dumps(self.body).encode()
        # requests-cache copies metadata from the underlying urllib3 response when storing
        response.raw = HTTPResponse(body=b"", status=200, headers=response.headers, request_url=request.url)
        response.url = request.url
        response.request = request
        return response

    def close(self):
        pass


# Sends the same query twice through a fresh in-memory cached session.
# In:  _post_twice({"data": {...}})
# Out: (second response, adapter)  -> check second.from_cache and adapter.calls

def _post_twice(body: dict):
    session = make_session(backend="memory")
    adapter = FakeAdapter(body)
    session.mount("https://", adapter)
    query = {"query": "{ meta { apiVersion { x } } }"}
    session.post(FAKE_URL, json=query)
    second = session.post(FAKE_URL, json=query)
    return second, adapter


# Error body -> not saved, so the fake network is hit both times (calls == 2).
def test_graphql_error_response_is_not_cached():
    second, adapter = _post_twice({"data": None, "errors": [{"message": "bad argument"}]})
    assert not second.from_cache
    assert adapter.calls == 2

# Good body -> saved, so the second call comes from the cache (calls == 1).
def test_graphql_success_response_is_cached():
    second, adapter = _post_twice({"data": {"meta": {"apiVersion": {"x": "26"}}}})
    assert second.from_cache
    assert adapter.calls == 1


# --- Fixture loader ---
# Fixtures are real API answers saved to tests/fixtures/ (the "data" part returned by graphql()).
# In:  load_fixture("ot_search_atopic_dermatitis.json")  -> Out: {"search": {"hits": [...]}}

FIXTURES = Path(__file__).parent / "fixtures"

def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# --- Step 2: disease search ---
# Fake: graphql() returns the saved search answer. Check: atopic eczema is first, 5 hits.
def test_search_disease_returns_atopic_eczema_first(monkeypatch):
    fixture = load_fixture("ot_search_atopic_dermatitis.json")
    monkeypatch.setattr(opentargets, "graphql", lambda query, variables: fixture)

    hits = opentargets.search_disease("atopic dermatitis")

    assert hits[0].id == "MONDO_0004980"
    assert len(hits) == 5


# --- Step 3: associated targets ---
# Fake: graphql() returns the saved top-25 targets. Check: sorted by score; IL13 in top 10 with a clinical score.
def test_associated_targets_parses_scores(monkeypatch):
    fixture = load_fixture("ot_associations_atopic_eczema.json")
    monkeypatch.setattr(opentargets, "graphql", lambda query, variables: fixture)

    targets = opentargets.get_associated_targets("MONDO_0004980")

    scores = [t.overall_score for t in targets]
    assert scores == sorted(scores, reverse=True)
    il13 = next(t for t in targets[:10] if t.symbol == "IL13")
    assert il13.datatype_scores["clinical"] > 0

# Fake: graphql() returns {"disease": None}, as the API does for an unknown ID. Check: ValueError.
def test_associated_targets_unknown_disease_raises(monkeypatch):
    monkeypatch.setattr(opentargets, "graphql", lambda query, variables: {"disease": None})

    with pytest.raises(ValueError):
        opentargets.get_associated_targets("NOT_A_DISEASE")


# --- Step 4: known drugs ---
# Fake: graphql() returns the saved IL13 drug list (7 drugs, 3 tested in atopic eczema).
# Check: only atopic eczema drugs kept, approved ones first, and cendakimab gets its AD stage (PHASE_2),
# not its overall stage (PHASE_3).
def test_known_drugs_filters_to_disease_and_uses_disease_stage(monkeypatch):
    fixture = load_fixture("ot_known_drugs_il13.json")
    monkeypatch.setattr(opentargets, "graphql", lambda query, variables: fixture)

    drugs = opentargets.get_known_drugs("ENSG00000169194", "MONDO_0004980")

    by_name = {d.drug_name: d for d in drugs}
    assert set(by_name) == {"LEBRIKIZUMAB", "TRALOKINUMAB", "CENDAKIMAB"}
    assert drugs[0].stage_for_disease == "APPROVAL"
    assert by_name["CENDAKIMAB"].stage_for_disease == "PHASE_2"
    assert by_name["CENDAKIMAB"].max_stage_any_disease == "PHASE_3"


# --- Step 5: target info and aliases ---
# Fake: graphql() returns the saved IL4R record. Check: CD124 is an alias; the official symbol isn't repeated.
def test_target_info_includes_aliases(monkeypatch):
    fixture = load_fixture("ot_target_info_il4r.json")
    monkeypatch.setattr(opentargets, "graphql", lambda query, variables: fixture)

    info = opentargets.get_target_info("ENSG00000077238")

    assert info.symbol == "IL4R"
    assert "CD124" in info.synonyms
    assert "IL4R" not in info.synonyms


# --- Step 6: skin expression ---
# Fake: graphql() returns the saved IL4R expression (all 51 GTEx tissues + 20 single-cell rows to ignore).
# Check: 2 skin tissues with TPM > 0; skin ranks 28th of 51 GTEx tissues.
def test_skin_expression(monkeypatch):
    fixture = load_fixture("ot_expression_il4r.json")
    monkeypatch.setattr(expression, "graphql", lambda query, variables: fixture)

    result = expression.get_skin_expression("ENSG00000077238")

    assert len(result.skin_tissues) == 2
    for tissue in result.skin_tissues:
        assert tissue.median_tpm > 0
    assert result.skin_rank == 28
    assert result.gtex_tissue_count == 51