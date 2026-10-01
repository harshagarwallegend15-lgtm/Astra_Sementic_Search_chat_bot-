"""Tests for the HTTP API and the frontend it serves.

The pipeline never imported Streamlit, so ``api.py`` is the second front end over
the same backend. These tests assert the contract the JavaScript console depends
on: the payload *shape* of every response, not the prose of any answer.

No network and no real model: ``src.llm.build_llm_client`` is patched at the
definition site, exactly as ``tests/test_ui_smoke.py`` does for the Streamlit app.
"""

from __future__ import annotations

import io
import shutil
import zlib
from contextlib import contextmanager
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


def _tiny_pdf() -> bytes:
    """A valid one-page PDF with a real text layer.

    Built by hand with zlib rather than pymupdf so the helper needs no heavy
    import at module scope. The body deliberately clears ``MIN_DOCUMENT_CHARS``
    (200): a shorter file is rejected as too small to index, which is correct
    behaviour and would make this an upload test that never uploads anything.
    """
    def obj(number: int, body: str) -> bytes:
        return f"{number} 0 obj\n{body}\nendobj\n".encode()

    lines = [
        "This document was uploaded by an operator during a test run.",
        "It exists to prove that an index survives a restart without being "
        "replaced by the bundled starter set, which would silently discard "
        "anything the operator had added.",
        "Unmanned ground vehicles are platforms that carry out a mission with "
        "no human crew onboard, under remote supervision from a control station.",
    ]
    text = "BT /F1 11 Tf 72 720 Td 14 TL\n"
    for line in lines:
        text += f"({line}) Tj T*\n"
    text += "ET"
    compressed = zlib.compress(text.encode())
    objects = [
        obj(1, "<< /Type /Catalog /Pages 2 0 R >>"),
        obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        obj(
            3,
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        ),
        (
            f"4 0 obj\n<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode()
            + compressed
            + b"\nendstream\nendobj\n"
        ),
        obj(5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for chunk in objects:
        offsets.append(out.tell())
        out.write(chunk)
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF".encode()
    )
    return out.getvalue()


def _citation_shape(citation: dict) -> None:
    assert citation["marker"]
    assert citation["title"]
    assert citation["page"] >= 1
    assert citation["excerpt"]
    assert 0.0 <= citation["score"] <= 1.0
    assert citation["chunk_id"]


class TestStartupSeed:
    """A deployed container starts with an empty index.

    ``data/vectorstore`` is gitignored, so a fresh container has nothing in it.
    Without a seed the console opens on an empty corpus and /api/health returns
    503, which makes the platform's own health check fail the deployment. These
    tests boot the app against an isolated empty directory and assert the seed
    happens exactly once and only when needed.
    """

    @staticmethod
    @contextmanager
    def _booted(root, monkeypatch, **extra_env):
        """Boot the app against an isolated directory, then tear it down.

        ``api._pipeline`` is a module-level cache, so it is cleared on both
        sides of every boot. Without that, a second boot would reuse the first
        boot's pipeline and read the first boot's index.
        """
        import api

        monkeypatch.setenv("VECTORSTORE_DIR", str(root / "vectorstore"))
        monkeypatch.setenv("UPLOADS_DIR", str(root / "uploads"))
        # The hashing backend keeps these hermetic: no model download.
        monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
        for key, value in extra_env.items():
            monkeypatch.setenv(key, value)

        # Settings are a process-wide singleton, so setting the environment is
        # not enough: without this refresh every boot would reuse the paths
        # resolved by whichever test ran first.
        import src.config as config_module

        monkeypatch.setattr(config_module, "_settings", None)

        api._pipeline = None
        try:
            with TestClient(api.app) as test_client:
                yield test_client
        finally:
            api._pipeline = None
            config_module._settings = None

    def test_an_empty_index_is_seeded_at_startup(self, tmp_path, monkeypatch):
        with self._booted(tmp_path, monkeypatch) as test_client:
            response = test_client.get("/api/health")
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["index_ready"] is True
            assert payload["checks"]["index"]["documents"] == 3

            state = test_client.get("/api/state").json()
            assert len(state["documents"]) == 3
            assert sum(d["chunk_count"] for d in state["documents"]) > 0

    def test_an_existing_index_is_not_reseeded(self, tmp_path, monkeypatch):
        """A restart must never discard what an operator uploaded.

        Reseeding unconditionally would replace their documents with the bundled
        starter set, so the upload would silently vanish across a deploy. The
        marker document below is the tripwire: it is not in sample-documents/,
        so if a reseed happened it would be gone.
        """
        with self._booted(tmp_path, monkeypatch) as test_client:
            assert test_client.get("/api/health").status_code == 200
            marker = _tiny_pdf()
            upload = test_client.post(
                "/api/upload",
                files={"files": ("operator-upload.pdf", marker, "application/pdf")},
            )
            assert upload.status_code == 200, upload.text
            assert len(upload.json()["added"]) == 1, upload.text
            before = len(test_client.get("/api/state").json()["documents"])

        with self._booted(tmp_path, monkeypatch) as restarted:
            after = restarted.get("/api/state").json()["documents"]
            assert len(after) == before, "the restart reseeded the index"
            assert any(d["filename"] == "operator-upload.pdf" for d in after)
            assert restarted.get("/api/health").status_code == 200

    def test_the_seed_can_be_disabled(self, tmp_path, monkeypatch):
        with self._booted(tmp_path, monkeypatch, ASTRA_SKIP_SEED="1") as test_client:
            # Still 503: nothing was indexed and nothing was forced.
            assert test_client.get("/api/health").status_code == 503

    def test_a_failed_seed_does_not_stop_the_service(self, tmp_path, monkeypatch):
        """Boot must survive an absent starter set.

        The console still has to load so the operator can see the problem and
        use the Maintenance controls. Refusing to start would leave a deploy
        with nothing at all.
        """
        import api
        import src.pipeline as pipeline_module

        def explode(self, rebuild=False):
            raise RuntimeError("starter documents are missing")

        monkeypatch.setattr(
            pipeline_module.AstraPipeline, "index_starter_documents", explode
        )
        with self._booted(tmp_path, monkeypatch) as test_client:
            assert test_client.get("/").status_code == 200
            assert test_client.get("/api/health").status_code == 503


class TestService:
    def test_health_probes_the_index_and_the_model(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["index_ready"] is True
        assert payload["llm_ready"] is True
        # A real probe, not a hardcoded liveness flag.
        assert payload["checks"]["index"]["chunks"] > 0
        assert payload["checks"]["index"]["documents"] == 3
        assert "error" not in payload["checks"]["index"]

    def test_health_reports_503_when_the_index_is_unusable(self, client, monkeypatch):
        import api

        monkeypatch.setattr(
            type(api.get_pipeline()),
            "stats",
            lambda self: {"chunks": 0, "documents": 0},
        )
        try:
            response = client.get("/api/health")
            assert response.status_code == 503
            payload = response.json()
            assert payload["index_ready"] is False
            assert payload["status"] == "degraded"
            # The payload keeps its shape when it fails, so a probe can read
            # what is broken without unwrapping `detail`.
            assert "detail" not in payload
        finally:
            monkeypatch.undo()

    def test_health_never_raises(self, client, monkeypatch):
        import api

        def explode(self):
            raise RuntimeError("index directory is not readable")

        monkeypatch.setattr(type(api.get_pipeline()), "stats", explode)
        try:
            response = client.get("/api/health")
            assert response.status_code == 503
            assert response.json()["status"] == "degraded"
        finally:
            monkeypatch.undo()

    def test_removing_an_unknown_document_is_a_404(self, client):
        response = client.delete("/api/documents/does-not-exist")
        assert response.status_code == 404
        assert "does-not-exist" in response.json()["detail"]

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

    def test_an_expected_provider_error_becomes_a_400_not_a_traceback(self, client, monkeypatch):
        """A user-facing LLMError is a client-visible 400 with no stack trace."""
        import src.errors as errors

        original = client.api._pipeline.llm
        client.api._pipeline.llm = lambda: None
        monkeypatch.setattr(
            client.api.AstraPipeline, "answerer",
            lambda self, llm=None: (_ for _ in ()).throw(errors.LLMError("provider exploded")),
        )
        try:
            response = client.post("/api/ask", json={"question": self.QUESTION})
            assert response.status_code == 400
            assert "Traceback" not in response.text
        finally:
            client.api._pipeline.llm = original

    def test_an_unexpected_error_is_not_leaked_to_the_client(self, client, monkeypatch):
        """An unexpected exception must not stringify its internals into the body.

        The handlers catch bare ``Exception``, so forwarding ``str(exc)`` can
        expose provider request metadata, filesystem paths and missing-key
        names. The cause belongs in the log; the client gets a correlation id.
        """
        original = client.api._pipeline.llm

        def boom(*_args, **_kwargs):
            raise RuntimeError("api_key=hunter2 at C:\\secrets\\vault.json")

        client.api._pipeline.llm = lambda: None
        monkeypatch.setattr(
            client.api.AstraPipeline, "answerer",
            lambda self, llm=None: (_ for _ in ()).throw(RuntimeError("api_key=hunter2 at C:\\secrets\\vault.json")),
        )
        try:
            response = client.post("/api/ask", json={"question": self.QUESTION})
            assert response.status_code == 500
            assert "Traceback" not in response.text
            assert "hunter2" not in response.text
            assert "vault.json" not in response.text
            # The client is given something it can quote in a bug report.
            assert "Reference:" in response.json()["detail"]
        finally:
            client.api._pipeline.llm = original


class TestIntake:
    def test_briefings_require_a_model(self, client):
        original = client.api._pipeline.llm
        client.api._pipeline.llm = lambda: None
        try:
            response = client.post("/api/briefings")
            assert response.status_code == 503
        finally:
            client.api._pipeline.llm = original

    def test_briefings_can_target_a_single_document(self, client, monkeypatch):
        """Briefing one row must not regenerate the whole set.

        The console briefs from the register row that was clicked, so an
        operator wanting one summary pays for one summary.
        """
        from src.models import DocumentSummary

        asked: list[str | None] = []

        def fake(self, llm=None, document_id=None):
            asked.append(document_id)
            return [DocumentSummary(
                document_id=document_id or "all",
                title="Stub",
                summary="Stub summary.",
                topics=["stubs"],
                llm_generated=True,
                model="stub-model",
            )]

        monkeypatch.setattr(client.api.AstraPipeline, "summaries", fake)

        document_id = client.get("/api/state").json()["documents"][0]["document_id"]
        payload = client.post(f"/api/briefings?document_id={document_id}").json()

        assert asked == [document_id]
        assert payload["summaries"][0]["document_id"] == document_id

    def test_a_short_pdf_reports_its_shortfall_not_a_scan(self, client, tmp_path):
        """A too-short PDF needs different advice than an image-only one.

        Both are `EmptyDocumentError`, so they previously shared one message
        that told the user their file was a scanned image. It extracted text
        fine; there was just too little of it. The message now names the
        actual shortfall.
        """
        import pymupdf

        doc = pymupdf.open()
        doc.new_page().insert_text((72, 720), "Too short.", fontsize=11)
        path = tmp_path / "stub.pdf"
        doc.save(str(path))

        response = client.post(
            "/api/upload",
            files={"files": ("stub.pdf", path.read_bytes(), "application/pdf")},
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "characters of text" in detail
        assert "scan" not in detail.lower()

    def test_briefing_an_unknown_document_is_a_404(self, client, monkeypatch):
        monkeypatch.setattr(
            client.api.AstraPipeline, "summaries",
            lambda self, llm=None, document_id=None: [],
        )
        response = client.post("/api/briefings?document_id=nope")
        assert response.status_code == 404
        assert "nope" in response.json()["detail"]

    def test_rebuild_is_refused_without_explicit_confirmation(self, client):
        """Rebuild discards every indexed document, so it must be deliberate.

        Without the token the endpoint used to be a single unauthenticated POST
        away from wiping the corpus, and the browser confirm() dialog was the
        only thing standing in the way.
        """
        before = client.get("/api/state").json()["stats"]["chunks"]
        response = client.post("/api/rebuild")
        assert response.status_code == 400
        assert "confirm=rebuild" in response.json()["detail"]
        # And it genuinely did not touch the index.
        assert client.get("/api/state").json()["stats"]["chunks"] == before

    def test_rebuild_reports_what_it_added(self, client):
        payload = client.post("/api/rebuild?confirm=rebuild").json()
        assert "added" in payload
        assert "chunk_count" in payload

    def test_clear_is_refused_without_explicit_confirmation(self, client):
        before = client.get("/api/state").json()["stats"]["chunks"]
        response = client.post("/api/clear")
        assert response.status_code == 400
        assert "confirm=clear" in response.json()["detail"]
        assert client.get("/api/state").json()["stats"]["chunks"] == before


class TestUploadGuards:
    """The upload endpoint has to reject bad batches before doing real work."""

    @staticmethod
    def _files(count: int) -> list[tuple[str, tuple[str, bytes, str]]]:
        return [
            ("files", (f"doc{i}.pdf", b"%PDF-1.4 stub", "application/pdf"))
            for i in range(count)
        ]

    def test_too_many_files_is_rejected(self, client):
        limit = client.api.MAX_UPLOAD_FILES
        response = client.post(
            "/api/upload",
            files=self._files(limit + 1),
        )
        assert response.status_code == 400
        assert str(limit) in response.json()["detail"]

    def test_an_oversized_batch_is_rejected_with_413(self, client, monkeypatch):
        """The limit is enforced while streaming, not after buffering the lot.

        Every UploadFile spools to a temp file first, so an unbounded batch is
        read into memory before any size check runs.
        """
        import dataclasses

        import api

        settings = dataclasses.replace(api.get_settings(), max_upload_mb=0)
        monkeypatch.setattr(api, "get_settings", lambda: settings)
        try:
            response = client.post("/api/upload", files=self._files(1))
            assert response.status_code == 413
            assert "MB limit" in response.json()["detail"]
        finally:
            monkeypatch.undo()

    def test_an_empty_batch_is_rejected(self, client):
        assert client.post("/api/upload", files=[]).status_code in {400, 422}

    def test_a_real_pdf_is_ingested(self, client):
        """The happy path still works end to end through the thread pool."""
        sources = sorted(
            (client.api.PROJECT_ROOT / "sample-documents").glob("*.pdf")
        )
        assert sources, "the starter corpus is missing"
        with sources[0].open("rb") as handle:
            response = client.post(
                "/api/upload",
                files=[
                    (
                        "files",
                        (sources[0].name, handle.read(), "application/pdf"),
                    )
                ],
            )
        assert response.status_code == 200, response.text
        assert client.get("/api/health").json()["index_ready"] is True