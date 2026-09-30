"""Headless smoke test for the Streamlit app.

``streamlit.testing.v1.AppTest`` runs ``app.py`` the way a browser does - full
script execution, widgets, session state - and surfaces exceptions that a plain
import check cannot. Importing ``app.py`` proves almost nothing, because
Streamlit only executes the script when a client connects. This is how the
``getattr(st, <function>)`` crash in ``_render_result`` was found: the app
raised on every single answer.

Two isolation details matter:

* ``app.get_llm`` and ``app.get_pipeline`` are ``@st.cache_resource``, so their
  values survive between ``AppTest`` instances in one process. The caches are
  cleared per test, otherwise a client built with (or without) credentials in an
  earlier test leaks into a later one.
* The app indexes into ``data/vectorstore``. The tests point it at a private
  copy so they neither depend on nor damage the developer's index.

No network and no real model: ``app.build_llm_client`` is replaced with a stub,
so these tests assert the app's own plumbing rather than the model's prose.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="streamlit is required for the UI smoke test")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from .stubs import StubLLM  # noqa: E402

APP = "app.py"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUB_ANSWER = (
    "Electronic warfare has three subdivisions: electronic attack, "
    "electronic protection and electronic warfare support [S1]."
)


@pytest.fixture(scope="module")
def isolated_index(tmp_path_factory):
    """A private copy of the persisted index, so tests are hermetic."""
    source = PROJECT_ROOT / "data" / "vectorstore"
    if not source.exists():
        pytest.skip("no persisted index; run `python app.py --index` first")
    root = tmp_path_factory.mktemp("ui-index")
    shutil.copytree(source, root / "vectorstore")
    return root


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    """Stop a cached pipeline or LLM client leaking between tests."""
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _launch(root: Path, monkeypatch, llm=StubLLM(STUB_ANSWER)) -> AppTest:
    """Run the app against an isolated index and a stubbed model.

    The patch is installed on ``src.llm``, not on ``app``, because
    ``AppTest.from_file`` executes ``app.py`` in its own namespace: the
    ``from src.llm import build_llm_client`` line binds at script-run time, so
    only the definition site is reachable. Patching the ``app`` module attribute
    silently does nothing and the test quietly calls the real provider.
    """
    import src.llm as llm_module

    monkeypatch.setenv("VECTORSTORE_DIR", str(root / "vectorstore"))
    monkeypatch.setenv("UPLOADS_DIR", str(root / "uploads"))
    monkeypatch.setattr(llm_module, "build_llm_client", lambda settings: llm)
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    return at


def _blob(at: AppTest) -> str:
    """Everything the app rendered, as one searchable string."""
    parts = [str(md.value) for md in at.markdown]
    parts += [str(cm.value) for cm in at.chat_message]
    parts += [str(cap.value) for cap in at.caption]
    parts += [str(e.value) for e in at.error]
    parts += [str(w.value) for w in at.warning]
    parts += [str(s.value) for s in at.success]
    parts += [str(i.value) for i in at.info]
    return "\n".join(parts)


def _ask(at: AppTest, question: str, label: str = "Ask") -> AppTest:
    """Type a question and click a button.

    The returned tree is the state *immediately after* the click's script run.
    Do not add "just in case" ``at.run()`` calls here: a bare rerun replays the
    script without the click, so ``_render_result`` stops emitting the live
    status banner and citations, and the assertions below fail against a
    correct app. Streamlit's own run queue is already drained once the click
    resolves (``default_timeout`` above), so the tree is complete.
    """
    next(t for t in at.text_input if t.label == "Your question").set_value(question)
    at.run()
    next(b for b in at.button if b.label == label).click().run()
    return at


class TestAppBoots:
    def test_the_script_runs_without_exceptions(self, isolated_index, monkeypatch):
        at = _launch(isolated_index, monkeypatch)
        assert not at.exception, [str(e) for e in at.exception]

    def test_the_title_is_rendered(self, isolated_index, monkeypatch):
        at = _launch(isolated_index, monkeypatch)
        assert any("ASTRA" in str(h.value) for h in at.title)

    def test_the_indexed_documents_are_listed(self, isolated_index, monkeypatch):
        at = _launch(isolated_index, monkeypatch)
        blob = _blob(at)
        assert "Electronic warfare" in blob
        assert "Unmanned ground vehicle" in blob

    def test_the_question_box_starts_empty(self, isolated_index, monkeypatch):
        at = _launch(isolated_index, monkeypatch)
        boxes = [t for t in at.text_input if t.label == "Your question"]
        assert len(boxes) == 1
        assert not boxes[0].value

    def test_it_boots_with_an_empty_index(self, tmp_path, monkeypatch):
        """No documents must render an instruction, not crash."""
        at = _launch(tmp_path, monkeypatch)
        assert not at.exception, [str(e) for e in at.exception]

    def test_the_offers_controls(self, isolated_index, monkeypatch):
        at = _launch(isolated_index, monkeypatch)
        labels = {b.label for b in at.button}
        assert {"Ask", "Show evidence only"} <= labels


class TestAnswering:
    QUESTION = "What are the three primary mission types of electronic warfare?"

    @pytest.fixture
    def answered(self, isolated_index, monkeypatch):
        return _ask(_launch(isolated_index, monkeypatch), self.QUESTION)

    def test_asking_does_not_raise(self, answered):
        assert not answered.exception, [str(e) for e in answered.exception]

    def test_the_question_is_echoed_back(self, answered):
        assert "primary mission types" in _blob(answered)

    def test_the_answer_is_shown(self, answered):
        # The stub's answer is rendered verbatim by the chat pane, so this
        # doubles as proof the stub model was actually injected.
        assert STUB_ANSWER.split(" [S1]")[0] in _blob(answered)

    def test_a_citation_is_rendered_with_a_page(self, answered):
        """Assert the citation's shape, not a specific page.

        Which page wins is a retrieval decision that legitimately changes when
        thresholds are recalibrated, so pinning a page number here would fail
        on a correct system.
        """
        blob = _blob(answered)
        assert "S1" in blob
        assert "Electronic warfare" in blob
        citation = re.search(r"\[S\d+\][^\n]*?page \d+", blob, re.IGNORECASE)
        assert citation, f"no '[S#] ... page N' citation rendered:\n{blob[-800:]}"

    def test_the_status_banner_is_rendered(self, answered):
        """This is the assertion that catches a crash in the result renderer."""
        blob = _blob(answered).lower()
        assert "confidence" in blob

    def test_asking_with_no_llm_degrades_to_evidence(self, isolated_index, monkeypatch):
        """A dead provider must show evidence, never a traceback.

        The banner is deliberately *styled* as an error for a refusal, so the
        assertion is on the absence of an uncaught exception and of a Python
        traceback, not on the absence of ``st.error``.
        """
        at = _ask(_launch(isolated_index, monkeypatch, llm=None), self.QUESTION)
        assert not at.exception, [str(e) for e in at.exception]
        blob = _blob(at)
        assert "Traceback" not in blob
        assert at.error or at.warning, "expected a refusal/evidence banner"
        assert "S1" in blob

    def test_evidence_only_mode_needs_no_model(self, isolated_index, monkeypatch):
        at = _ask(
            _launch(isolated_index, monkeypatch, llm=None),
            self.QUESTION,
            label="Show evidence only",
        )
        assert not at.exception, [str(e) for e in at.exception]
        assert "S1" in _blob(at)

    def test_an_empty_question_is_rejected_gracefully(self, isolated_index, monkeypatch):
        at = _ask(_launch(isolated_index, monkeypatch), "   ")
        assert not at.exception, [str(e) for e in at.exception]
        assert any("Enter a question" in str(w.value) for w in at.warning)