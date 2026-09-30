"""End-to-end evaluation against the starter question set.

Runs every question in ``example-questions.md`` through the real pipeline with a
real LLM and checks the behaviour that matters:

* whether the expected document was cited;
* whether the answer carries resolvable citations;
* whether any number in the answer is absent from the retrieved pages;
* for questions the documents genuinely do not answer, whether the system
  refuses instead of inventing content.

Usage::

    python scripts/evaluate.py                       # starter documents + Ollama
    python scripts/evaluate.py --no-llm              # retrieval gate only
    python scripts/evaluate.py --question 10         # single question
    python scripts/evaluate.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.citations import build_citations  # noqa: E402
from src.grounding import detect_refusal, verify  # noqa: E402
from src.llm import build_llm_client  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402
from src.pipeline import AstraPipeline  # noqa: E402

QUESTION_FILE = PROJECT_ROOT / "example-questions.md"

#: Questions whose honest answer is that the documents do not state it.
EXPECTED_REFUSAL = {10}

#: Document title fragments each question should surface, where checkable.
EXPECTED_DOCUMENT = {
    1: "unmanned aerial vehicle",
    2: "electronic warfare",
    3: "unmanned ground vehicle",
    4: None,
    5: "unmanned aerial vehicle",
    6: None,
    7: "unmanned ground vehicle",
    8: None,
    9: None,
    10: None,
}

#: Substrings that must NOT appear in an answer, per question.
FORBIDDEN = {
    10: ("mission types",),
}

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


_QUESTION_ITEM_RE = re.compile(r"^(\d+)[\.\)]\s+(.*\S)\s*$")

#: Lines that end an item rather than continuing it.
_QUESTION_BREAK_RE = re.compile(r"^\s*(?:[-*>]\s|#{1,6}\s|\d+[\.\)]\s)")


def load_questions() -> list[str]:
    """Read the numbered questions, joining wrapped continuation lines.

    The question file is prose markdown, so every one of the ten questions wraps
    onto a second indented line. Reading one line per item silently truncated
    all ten - "Compare UAVs and UGVs as presented in the" - so the model was
    answering fragments and the scores described the wrong task. Continuation is
    any indented, non-blank line that does not itself start a new item, list
    bullet, heading or blockquote (the notes under question 10 must not be
    swallowed).
    """
    if not QUESTION_FILE.is_file():
        return []

    questions: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            questions.append(" ".join(current).strip())
            current.clear()

    for raw in QUESTION_FILE.read_text(encoding="utf-8").splitlines():
        item = _QUESTION_ITEM_RE.match(raw.strip())
        if item:
            flush()
            current.append(item.group(2).strip())
            continue
        if not current:
            continue
        # A blank line or a new block ends the item - but only after recording it.
        if not raw.strip() or _QUESTION_BREAK_RE.match(raw):
            flush()
        elif raw[:1].isspace():
            current.append(raw.strip())
    flush()
    return questions


def evaluate_question(
    pipeline: AstraPipeline,
    llm,
    number: int,
    question: str,
) -> dict:
    started = time.perf_counter()
    answerer = pipeline.answerer(llm)
    result = answerer.answer(question)
    elapsed = time.perf_counter() - started

    report = verify(result.answer, result.retrieved, require_citations=False)
    is_refusal = detect_refusal(result.answer)
    titles = [c.title.lower() for c in result.citations]
    expected_doc = EXPECTED_DOCUMENT.get(number)
    doc_ok = expected_doc is None or any(expected_doc in title for title in titles)

    checks = {
        "has_citations": bool(result.citations),
        "document_match": doc_ok,
        "no_unsupported_numbers": not report.unsupported_numbers,
        "markers_resolve": not report.unknown_markers,
    }
    if number in EXPECTED_REFUSAL:
        checks["refused"] = is_refusal

    return {
        "number": number,
        "question": question,
        "answer": result.answer,
        "status": result.status,
        "confidence": result.confidence,
        "citations": [c.to_dict() for c in result.citations],
        "retrieved_count": len(result.retrieved),
        "unsupported_numbers": report.unsupported_numbers,
        "unknown_markers": report.unknown_markers,
        "is_refusal": is_refusal,
        "notes": result.notes,
        "latency_ms": result.latency_ms,
        "elapsed_s": round(elapsed, 1),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-llm", action="store_true", help="Retrieval gate only.")
    parser.add_argument("--question", type=int, help="Run a single question by number.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index first.")
    parser.add_argument("--json", type=Path, help="Write full results to this file.")
    args = parser.parse_args()

    setup_logging()
    questions = load_questions()
    if not questions:
        print(f"Could not read questions from {QUESTION_FILE}")
        return 2
    if args.question:
        questions = [questions[args.question - 1]]
        number_offset = args.question - 1
    else:
        number_offset = 0

    print("Loading / building index ...")
    pipeline = AstraPipeline()
    if args.rebuild or not pipeline.load():
        pipeline.index_starter_documents(rebuild=args.rebuild)
    stats = pipeline.stats()
    print(
        f"  {stats.get('documents')} documents, "
        f"{stats.get('chunks')} chunks, model={stats.get('embedding_model')}"
    )

    llm = None
    if not args.no_llm:
        try:
            llm = build_llm_client(pipeline.settings)
        except Exception as exc:  # noqa: BLE001
            print(f"  LLM unavailable ({exc}); running retrieval-only mode.")

    results: list[dict] = []
    for offset, question in enumerate(questions):
        number = number_offset + offset + 1
        print(f"\n{'=' * 72}\nQ{number}: {question}")
        try:
            entry = evaluate_question(pipeline, llm, number, question)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            results.append({"number": number, "question": question, "passed": False,
                            "error": str(exc)})
            continue
        results.append(entry)

        mark = f"{GREEN}PASS{RESET}" if entry["passed"] else f"{RED}FAIL{RESET}"
        print(f"  {mark}  status={entry['status']} conf={entry['confidence']} "
              f"chunks={entry['retrieved_count']} {entry['elapsed_s']}s")
        for name, ok in entry["checks"].items():
            if not ok:
                print(f"    {RED}failed check: {name}{RESET}")
        if entry.get("unsupported_numbers"):
            print(f"    {RED}unsupported numbers: {entry['unsupported_numbers']}{RESET}")
        print(f"  --- answer ---\n{_indent(entry['answer'])}")
        print("  --- citations ---")
        for citation in entry["citations"][:4]:
            print(f"    [{citation['marker']}] {citation['title']} p.{citation['page']} "
                  f"({citation['section']})")
        for text in FORBIDDEN.get(entry["number"], ()):
            if text.lower() in entry["answer"].lower():
                print(f"    {RED}forbidden term present: {text!r}{RESET}")
                entry["passed"] = False

    passed = sum(1 for r in results if r.get("passed"))
    print(f"\n{'=' * 72}")
    colour = GREEN if passed == len(results) else YELLOW
    print(f"{colour}{passed}/{len(results)} checks passed{RESET}")

    if args.json:
        args.json.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote {args.json}")
    return 0 if passed == len(results) else 1


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in (text or "").splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
