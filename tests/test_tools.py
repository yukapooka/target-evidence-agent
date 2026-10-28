import json
import requests
from requests.adapters import BaseAdapter
from urllib3 import HTTPResponse
from agent.tools.http import make_session
from pathlib import Path
from agent.tools import opentargets

FAKE_URL = "https://fake.test/graphql"


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


def _post_twice(body: dict):
    session = make_session(backend="memory")
    adapter = FakeAdapter(body)
    session.mount("https://", adapter)
    query = {"query": "{ meta { apiVersion { x } } }"}
    session.post(FAKE_URL, json=query)
    second = session.post(FAKE_URL, json=query)
    return second, adapter


def test_graphql_error_response_is_not_cached():
    second, adapter = _post_twice({"data": None, "errors": [{"message": "bad argument"}]})
    assert not second.from_cache
    assert adapter.calls == 2

def test_graphql_success_response_is_cached():
    second, adapter = _post_twice({"data": {"meta": {"apiVersion": {"x": "26"}}}})
    assert second.from_cache
    assert adapter.calls == 1

FIXTURES = Path(__file__).parent / "fixtures"

def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())

def test_search_disease_returns_atopic_eczema_first(monkeypatch):
    fixture = load_fixture("ot_search_atopic_dermatitis.json")
    monkeypatch.setattr(opentargets, "graphql", lambda query, variables: fixture)

    hits = opentargets.search_disease("atopic dermatitis")

    assert hits[0].id == "MONDO_0004980"
    assert len(hits) == 5
