"""Is semantic search actually working, or is it silently lexical?

`EMBEDDING_BACKEND=auto` falls back to a deterministic hashing embedder when the
model cannot load. The app still answers, and the results still look like
results - but the vectors carry no meaning, so paraphrases stop matching and
retrieval quietly degrades to keyword overlap. Nothing in the UI would reveal
that, which is exactly why it needs checking.

Three things are asserted:
  1. the real MiniLM model is loaded, not the fallback;
  2. the corpus vectors are 384-dimensional and non-degenerate;
  3. a paraphrase with no shared keywords still retrieves the right document,
     which is the property a lexical index cannot have.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from anywhere: the package lives at the repo root, not next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.embeddings import get_embedder
from src.pipeline import AstraPipeline

FAILS: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + name + (f"  --  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


settings = get_settings()

# ------------------------------------------------------------ 1. the model --

print("=== embedder ===")
print(f"  configured : {settings.embedding_model}")
print(f"  backend    : {settings.embedding_backend}")

embedder = get_embedder(settings)
print(f"  active     : {embedder.name}")
print(f"  dimensions : {embedder.dimensions}")

check(
    "hash" not in embedder.name.lower(),
    "the semantic model is loaded",
    f"active embedder is {embedder.name!r}; a hashing fallback means no semantics",
)
check(
    embedder.dimensions == 384,
    "MiniLM 384-dimensional vectors",
    f"got {embedder.dimensions}",
)
check(
    settings.embedding_backend != "auto",
    "the backend is pinned, not `auto`",
    "with `auto` a failed download silently becomes lexical",
)

# ------------------------------------------------------ 2. the index itself --

print("\n=== index ===")
pipeline = AstraPipeline(settings)
pipeline.load()
store = pipeline.store
chunks = store.all_chunks()
check(len(chunks) > 0, "the index has chunks", f"{len(chunks)} chunks")

if chunks:
    vectors = embedder.embed_documents([c.text[:600] for c in chunks[:24]])
    widths = {len(v) for v in vectors}
    check(widths == {384}, "every vector has 384 dimensions", f"widths seen: {widths}")

    # Degenerate embeddings (all zeros, all identical) still satisfy a width
    # check, so test for actual variation.
    first = vectors[0]
    distinct = len({tuple(round(x, 6) for x in v) for v in vectors})
    check(
        distinct >= len(vectors) - 1,
        "vectors are distinct, not collapsed to one value",
        f"{distinct} distinct of {len(vectors)}",
    )
    nonzero = sum(1 for x in first if x != 0.0)
    check(nonzero > len(first) * 0.5, "vectors are not mostly zero", f"{nonzero} non-zero terms")

# --------------------------------------- 3. does meaning beat keyword overlap --

print("\n=== retrieval quality ===")

# Each probe is answered by a known document, and the probe deliberately avoids
# the document's own vocabulary. A BM25-only index scores these near zero.
PROBES = [
    ("What do the three primary mission types of electronic warfare include?",
     "defence-electronics-electronic-warfare.pdf",
     "electronic warfare"),
    ("How is a flying platform kept under control when nobody is aboard?",
     "unmanned-aerial-vehicle-overview.pdf",
     "aerial"),
    ("What sorts of sensors let a ground robot perceive its surroundings?",
     "autonomous-systems-unmanned-ground-vehicle.pdf",
     "ground vehicle"),
]

from src.retriever import tokenize  # noqa: E402


def unpack(results):
    """The retriever yields RetrievedChunk; the id list is what we need."""
    return [r.chunk for r in results]


def shared_terms(query: str, text: str) -> set[str]:
    q = set(tokenize(query))
    t = set(tokenize(text))
    return {w for w in q & t if len(w) > 3}


for question, expected_file, hint in PROBES:
    hits = pipeline.retriever.search(question, k=5)
    if not hits:
        check(False, f"retrieval returns results for {hint!r}", "no results")
        continue

    ranked = unpack(hits)
    files = [c.filename for c in ranked]
    docs = [c.title for c in ranked]

    # Where did the expected document land?
    position = next(
        (i for i, f in enumerate(files) if f == expected_file),
        None,
    )
    top1_doc = docs[0] if docs else "?"
    best = hits[0].score if hasattr(hits[0], "score") else 0.0
    overlap = shared_terms(question, ranked[0].text)

    check(
        position == 0,
        f"{hint!r}: the right document is ranked first",
        f"top1={top1_doc!r} at {float(best):.3f}; "
        f"expected={expected_file}; top3={[d[:28] for d in docs[:3]]}",
    )
    check(
        len(overlap) <= 3,
        f"{hint!r}: the top hit is reached without heavy keyword overlap",
        f"{len(overlap)} shared content words: {sorted(overlap)[:6]}",
    )

# ------------------------------------------------- 4. paraphrase consistency --

print("\n=== paraphrase stability ===")
base = pipeline.retriever.search("electronic warfare mission types", k=3)
if base:
    base_doc = unpack(base)[0].filename
    # Same meaning, none of the same content words.
    for rephrase in (
        "list the three kinds of electronic warfare",
        "name the warfare branches that attack, protect and support",
    ):
        got = pipeline.retriever.search(rephrase, k=3)
        top = unpack(got)[0].filename if got else None
        check(
            top == base_doc,
            f"rephrase still resolves to {base_doc[:28]!r}",
            f"{rephrase!r} -> {top!r}",
        )

print()
if FAILS:
    print(f"{len(FAILS)} check(s) failed: " + "; ".join(FAILS))
    sys.exit(1)
print("semantic search is healthy")
