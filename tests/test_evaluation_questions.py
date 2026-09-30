"""The evaluation question loader.

This exists because of a bug that invalidated a full round of results: the
question file is prose markdown and every question wraps onto a second indented
line, but the loader read one line per numbered item. All ten questions were
truncated to fragments - question 8 reached the model as "Compare unmanned
aerial and unmanned ground vehicles as presented in the" - so the model was
answering sentence fragments and every score described the wrong task.

The test pins the *shape* of the real file rather than a hand-written fixture,
because the failure mode only appears when continuation lines exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

evaluate = pytest.importorskip("scripts.evaluate", reason="evaluation harness")
load_questions = evaluate.load_questions


@pytest.fixture(scope="module")
def questions() -> list[str]:
    loaded = load_questions()
    assert loaded, "example-questions.md produced no questions"
    return loaded


def test_all_ten_questions_are_loaded(questions):
    assert len(questions) == 10


def test_every_question_is_a_complete_sentence(questions):
    """A truncated question loses its verb phrase and its real ask."""
    for number, question in enumerate(questions, start=1):
        # Question 10 is quoted in the source file, so it closes with ?" rather
        # than a bare ?.
        assert question.rstrip('"').endswith(("?", ".")), (
            f"Q{number} does not end as a question: {question!r}"
        )


def test_wrapped_continuation_lines_are_joined(questions):
    # Each of these continuations lives on the second line of the markdown.
    assert questions[0].endswith("*Unmanned aerial vehicle* document?")
    assert questions[3].endswith("Point me to them.")
    assert questions[4].endswith("surveillance UAVs?")
    assert questions[5].endswith("distinguish jamming from deception?")
    assert questions[6].endswith("and its developer.")


def test_the_comparison_question_is_complete(questions):
    """The question whose truncation made a weak answer look like a model fault."""
    assert questions[7].startswith("Compare unmanned aerial and unmanned ground vehicles")
    assert "shared operational roles" in questions[7]
    assert questions[7].endswith("what shared operational roles do they have?")


def test_the_honesty_question_keeps_its_instruction(questions):
    assert questions[8].endswith("Answer only with statements actually present.")


def test_annotation_notes_are_not_swallowed_into_a_question(questions):
    """Question 10 is followed by bullets describing good and bad behaviour."""
    assert questions[9].endswith('documents?"')
    assert "Grounded behaviour" not in questions[9]
    assert "60%" not in questions[9]


def test_no_question_absorbed_a_bullet_heading_or_prose_after_the_list(questions):
    for number, question in enumerate(questions, start=1):
        for leaked in ("Answer **+ source pointer**", "Minimum requirement", "Wikipedia content"):
            assert leaked not in question, f"Q{number} leaked prose: {question!r}"


def test_questions_are_stripped_of_surrounding_whitespace(questions):
    for number, question in enumerate(questions, start=1):
        assert question == question.strip(), f"Q{number} has stray whitespace"
        assert "  " not in question, f"Q{number} has a double space after joining"


def test_loader_is_deterministic(questions):
    assert load_questions() == questions