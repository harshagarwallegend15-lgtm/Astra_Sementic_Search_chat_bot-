"""Calibrate retrieval thresholds against a candidate embedding model.

Retrieval thresholds are *not* transferable between embedding models. Similarity
is not comparable across encoders: ``all-MiniLM-L6-v2`` puts unrelated defence
text around 0.1-0.3, while ``bge-small`` pushes related text as high as 0.8 and
leaves unrelated text higher too. Reusing a threshold tuned for one model on
another silently returns too much (flooding the context with noise) or too
little (false refusals).

This script answers three questions for a given model:

1. What does the score distribution actually look like - where does relevant
   evidence sit, and where does off-topic text sit?
2. Which ``MIN_DENSE_SIMILARITY`` separates them without discarding the
   passages that carry the answer?
3. Which ``MIN_RETRIEVAL_SCORE`` maximises document-level recall without letting
   unrelated documents into the context?

The objective is document recall: for each question, does the retrieved context
contain at least one passage from the document that should answer it? That is the
signal that decides whether the model *can* answer, so optimising it directly
reduces false refusals. Per-question ``--detail`` output shows which passages were
found, which is how the Q7 UGV-programme failure was diagnosed.

No LLM calls, so a full sweep takes seconds.

Usage
-----
    python scripts/calibrate.py
    python scripts/calibrate.py --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
    python scripts/calibrate.py --detail
    python scripts/calibrate.py --apply BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings, get_settings  # noqa: E402
from src.pipeline import AstraPipeline  # noqa: E402
from src.retriever import Retriever  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate import load_questions  # noqa: E402

# Which document must appear in the context for the question to be answerable.
# Derived from example-questions.md, which names the source document per
# question. Questions that are explicitly cross-document are handled below.
EXPECTED_SUBSTRINGS: dict[int, tuple[str, ...]] = {
    1: ("unmanned aerial vehicle",),
    2: ("electronic warfare",),
    3: ("unmanned ground vehicle",),
    4: ("unmanned aerial vehicle",),
    5: ("unmanned aerial vehicle",),
    6: ("electronic warfare",),
    7: ("unmanned ground vehicle",),
    8: ("unmanned aerial vehicle", "unmanned ground vehicle"),
    9: ("unmanned aerial vehicle", "unmanned ground vehicle", "electronic warfare"),
    10: ("unmanned aerial vehicle",),
}

DENSE_GRID = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
SCORE_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
RATIO_GRID = (0.0, 0.30, 0.45, 0.60, 0.75)


def build_settings(model: str, root: Path) -> Settings:
    """Settings for a throwaway index using ``model``."""
    base = get_settings()
    return replace(
        base,
        embedding_model=model,
        embedding_backend="semantic",
        vectorstore_dir=root / "vectorstore",
        uploads_dir=root / "uploads",
    )


def build_index(settings: Settings) -> AstraPipeline:
    pipeline = AstraPipeline(settings)
    pipeline.index_starter_documents(rebuild=True)
    return pipeline


def recall(retriever: Retriever, question: str, expected: tuple[str, ...]) -> float:
    """Fraction of the expected documents present in the retrieved context."""
    titles = {
        rc.chunk.title.lower() for rc in retriever.search(question)
    }
    if not expected:
        return 1.0
    return sum(1 for want in expected if any(want in t for t in titles)) / len(expected)


def question_rows(pipeline: AstraPipeline, questions, settings: Settings):
    """Yield ``(number, text, expected_titles)`` for each evaluation question."""
    for number, text in enumerate(questions, start=1):
        yield number, text, EXPECTED_SUBSTRINGS.get(number, ())


def sweep(pipeline: AstraPipeline, questions, settings: Settings) -> dict:
    """Grid-search the thresholds, maximising recall then penalising noise."""
    titles_of = {d.document_id: d.title.lower() for d in pipeline.documents()}
    best = None
    for dense in DENSE_GRID:
        for score in SCORE_GRID:
            for ratio in RATIO_GRID:
                probe = replace(
                    settings,
                    min_dense_similarity=dense,
                    min_retrieval_score=score,
                    document_diversity_ratio=ratio,
                )
                retriever = Retriever(pipeline.store, probe)
                scores = [recall(retriever, q, e) for _, q, e in question_rows(pipeline, questions, probe)]
                total = sum(scores) / len(scores)
                # Reward recall, penalise a wide net: a threshold that drags in
                # irrelevant documents is how the LLM ends up hedging.
                noise = 0.0
                for _, q, expected in question_rows(pipeline, questions, probe):
                    for rc in retriever.search(q):
                        title = titles_of.get(rc.chunk.document_id, "")
                        if not any(w in title for w in expected):
                            noise += 1
                objective = total - 0.02 * (noise / max(1, len(questions)))
                if best is None or objective > best[0]:
                    best = (objective, dense, score, ratio, total, noise)
    objective, dense, score, ratio, total, noise = best
    return {
        "min_dense_similarity": dense,
        "min_retrieval_score": score,
        "document_diversity_ratio": ratio,
        "recall": total,
        "off_topic_chunks": noise,
    }


def score_distribution(pipeline: AstraPipeline, questions, settings: Settings) -> None:
    """Print where relevant and off-topic evidence actually lands."""
    probe = replace(settings, min_dense_similarity=0.0, min_retrieval_score=0.0)
    retriever = Retriever(pipeline.store, probe)
    titles_of = {d.document_id: d.title.lower() for d in pipeline.documents()}
    print(f"\n  {'Q':>3}  {'top':>7} {'top':>7}  {'mean':>7} {'p95':>7}   context")
    for number, question, expected in question_rows(pipeline, questions, probe):
        found = retriever.search(question)
        if not found:
            print(f"  {number:>3}  (nothing)")
            continue
        dense = [rc.dense_score for rc in found if rc.dense_score is not None]
        on_topic = [
            rc.dense_score
            for rc in found
            if rc.dense_score is not None
            and any(w in titles_of.get(rc.chunk.document_id, "") for w in expected)
        ]
        off = [d for d in dense if d not in on_topic]
        top = max(dense) if dense else 0.0
        mean = sum(dense) / len(dense) if dense else 0.0
        ordered = sorted(dense)
        p95 = ordered[int(len(ordered) * 0.95)] if ordered else 0.0
        print(
            f"  {number:>3}  {top:>7.3f} {max(on_topic) if on_topic else 0.0:>7.3f} "
            f"{mean:>7.3f} {p95:>7.3f}   {len(found)} chunks"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
    )
    parser.add_argument("--detail", action="store_true", help="Print score distributions.")
    parser.add_argument("--apply", metavar="MODEL", help="Write results into .env.")
    args = parser.parse_args()

    questions = load_questions()
    if not questions:
        print("No questions found in example-questions.md", file=sys.stderr)
        return 1

    results: dict[str, dict] = {}
    for model in args.models:
        root = Path(tempfile.mkdtemp(prefix="astra-cal-"))
        try:
            settings = build_settings(model, root)
            print(f"\n=== {model}")
            pipeline = build_index(settings)
            if args.detail:
                score_distribution(pipeline, questions, settings)
            results[model] = sweep(pipeline, questions, settings)
            best = results[model]
            print(
                f"  recall {best['recall']:.2f} | MIN_DENSE_SIMILARITY="
                f"{best['min_dense_similarity']:.2f} MIN_RETRIEVAL_SCORE="
                f"{best['min_retrieval_score']:.2f} RATIO={best['document_diversity_ratio']:.2f}"
                f" | off-topic chunks {best['off_topic_chunks']}"
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    if len(results) > 1:
        print("\n" + "=" * 72)
        print(f"{'model':<44} {'recall':>7} {'dense':>7} {'score':>7} {'ratio':>7}")
        for model, best in results.items():
            print(
                f"{model:<44} {best['recall']:>7.2f} "
                f"{best['min_dense_similarity']:>7.2f} {best['min_retrieval_score']:>7.2f} "
                f"{best['document_diversity_ratio']:>7.2f}"
            )

    if args.apply:
        best = results.get(args.apply)
        if not best:
            print(f"\nNo results for {args.apply!r}; nothing applied.", file=sys.stderr)
            return 1
        env = PROJECT_ROOT / ".env"
        if not env.exists():
            print("\nNo .env to update.", file=sys.stderr)
            return 1
        text = env.read_text(encoding="utf-8")
        for key, value in (
            ("EMBEDDING_MODEL", args.apply),
            ("MIN_DENSE_SIMILARITY", f"{best['min_dense_similarity']:.2f}"),
            ("MIN_RETRIEVAL_SCORE", f"{best['min_retrieval_score']:.2f}"),
            ("DOCUMENT_DIVERSITY_RATIO", f"{best['document_diversity_ratio']:.2f}"),
        ):
            import re

            pattern = re.compile(rf"^{key}=.*$", re.MULTILINE)
            if pattern.search(text):
                text = pattern.sub(f"{key}={value}", text)
            else:
                text += f"\n{key}={value}\n"
        env.write_text(text, encoding="utf-8")
        print(f"\nUpdated .env for {args.apply}.")
        print("Now rebuild the index:  python app.py --index --rebuild")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())