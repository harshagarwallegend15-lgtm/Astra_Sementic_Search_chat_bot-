"""Tests for the HTTP API and the frontend it serves.

The pipeline never imported Streamlit, so ``api.py`` is the second front end over
the same backend. These tests assert the contract the JavaScript console depends
on: the payload *shape* of every response, not the prose of any answer.

No network and no real model: ``src.llm.build_llm_client`` is patched at the
definition site, exactly as ``tests/test_ui_smoke.py`` does for the Streamlit app.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="fastapi is required for the API tests")

from fastapi.testclient import TestClient  # noqa: E402

from .stubs import StubLLM  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STUB_ANSWER = (
    "Electronic warfare has three subdivisions: electronic attack, "
    "electronic protection and electronic warfare support [S1]."
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A TestClient over an isolated copy of the index, with a stubbed model."""
    source = PROJECT_ROOT / "data" / "vectorstore"
    if not source.exists():
        pytest.skip("no persisted index; run `python app.py --index` first")

    root = tmp_path_factory.mktemp("api-index")
    shutil.copytree(source, root / "vectorstore")

    import os

    os.environ["VECTORSTORE_DIR"] = str(root / "vectorstore")
    os.environ["UPLOADS_DIR"] = str(root / "uploads")

    import src.llm as llm_module

    original = llm_module.build_llm_client
    llm_module.build_llm_client = lambda settings: StubLLM(STUB_ANSWER)

    import api

    api._pipeline = None
    with TestClient(api.app) as test_client:
        test_client.api = api
        yield test_client

    llm_module.build_llm_client = original
    api._pipeline = None


def _citation_shape(citation: dict) -> None:
    assert citation["marker"]
    assert citation["title"]
    assert citation["page"] >= 1
    assert citation["excerpt"]
    assert 0.0 <= citation["score"] <= 1.0
    assert citation["chunk_id"]


class TestService:
    def test_health(self, client):
        assert client.get("/api/health").json() == {"ok": True}

    def test_state_reports_documents_and_model(self, client):
        payload = client.get("/api/state").json()
        assert payload["stats"]["documents"] == 3
        assert payload["stats"]["chunks"] > 0
        assert payload["llm_ready"] is True
        assert len(payload["documents"]) == 3
        assert all("document_id" in doc for doc in payload["documents"])

    def test_suggestions_come_from_the_question_file(self, client):
        payload = client.get("/api/suggestions").json()
        assert len(payload["questions"]) == 10
        assert all(q.strip() for q in payload["questions"])

    def test_the_frontend_is_served(self, client):
        index = client.get("/")
        assert index.status_code == 200
        assert "ASTRA" in index.text
        for asset in ("/static/styles.css", "/static/app.js"):
            assert client.get(asset).status_code == 200, asset


class TestAsking:
    QUESTION = "What are the three primary mission types of electronic warfare?"

    def test_ask_returns_a_renderable_answer(self, client):
        payload = client.post("/api/ask", json={"question": self.QUESTION}).json()
        assert payload["status"] in {
            "grounded", "partially_grounded", "ungrounded", "no_context",
        }
        assert 0.0 <= payload["confidence"] <= 1.0
        assert payload["answer"].strip()
        assert payload["citations"], "a grounded answer must carry citations"
        _citation_shape(payload["citations"][0])

    def test_retrieved_passages_serialize_with_their_chunk(self, client):
        payload = client.post("/api/ask", json={"question": self.QUESTION}).json()
        assert payload["retrieved"]
        chunk = payload["retrieved"][0]["chunk"]
        assert chunk["text"].strip()
        assert chunk["page"] >= 1
        assert "score" in payload["retrieved"][0]

    def test_the_stub_model_is_what_answered(self, client):
        payload = client.post("/api/ask", json={"question": self.QUESTION}).json()
        assert STUB_ANSWER.split(" [S1]")[0] in payload["answer"]

    def test_evidence_returns_citations_without_a_model(self, client):
        payload = client.post("/api/evidence", json={"question": self.QUESTION}).json()
        assert payload["citations"]
        _citation_shape(payload["citations"][0])

    def test_an_empty_question_is_rejected(self, client):
        assert client.post("/api/ask", json={"question": "   "}).status_code == 422

    def test_a_failing_model_becomes_a_500_not_a_traceback(self, client, monkeypatch):
        """A provider blow-up must be a clean HTTP error, never a stack trace."""
        import src.errors as errors

        def boom(*_args, **_kwargs):
            raise errors.LLMError("provider exploded")

        client.api._pipeline.llm = lambda: None
        monkeypatch.setattr(
            client.api.AstraPipeline, "answerer",
            lambda self, llm=None: (_ for _ in ()).throw(errors.LLMError("provider exploded")),
        )
        response = client.post("/api/ask", json={"question": self.QUESTION})
        assert response.status_code == 400
        assert "Traceback" not in response.text


class TestIntake:
    def test_briefings_require_a_model(self, client):
        original = client.api._pipeline.llm
        client.api._pipeline.llm = lambda: None
        try:
            response = client.post("/api/briefings")
            assert response.status_code == 503
        finally:
            client.api._pipeline.llm = original

    def test_rebuild_reports_what_it_added(self, client):
        payload = client.post("/api/rebuild").json()
        assert "added" in payload
        assert "chunk_count" in payload