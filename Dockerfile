# syntax=docker/dockerfile:1
#
# ASTRA INTEL - FastAPI console image for Railway / Render.
#
# This supersedes the Streamlit image: the vanilla console in static/ is the
# front end, served by api.py on one port. The previous CMD started Streamlit,
# which would have deployed the wrong application entirely.
#
# Size note: torch dominates the image. It is pulled from the CPU-only index so
# a CPU host does not download ~2 GB of CUDA libraries it can never use.
#
# Never bake an API key into the image. Railway injects it at run time.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The container installs a trimmed set. The full requirements.txt pulls in
# Streamlit, langchain-ollama and pytest, which a server running api.py never
# imports, at the cost of a much heavier and slower build.
COPY requirements.txt requirements-container.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-container.txt

COPY src/ ./src/
COPY static/ ./static/
COPY api.py ./
COPY sample-documents/ ./sample-documents/
COPY sample-metadata.json ./example-questions.md ./

# Bake the embedding model into the image. Without this the first request
# downloads ~90 MB from HuggingFace, and on a platform that freezes the
# filesystem after the first response that download either fails or silently
# re-runs on every cold start.
RUN python -c "\
from src.config import get_settings;\
from src.embeddings import get_embedder;\
e = get_embedder(get_settings());\
print('embedding backend ready:', e.name, e.dimensions)"

# The FAISS index and uploaded PDFs live on a mounted volume, not here.
RUN mkdir -p data/vectorstore data/uploads

EXPOSE 8502

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -fsS "http://localhost:${PORT:-8502}/api/health" || exit 1

# The index directory is written at run time, so the service cannot run as a
# user that lacks write access to it.
RUN useradd --create-home --uid 10001 astra \
    && chown -R astra:astra /app
USER astra

# PORT is injected by the platform and read in api.py's main block.
CMD ["python", "api.py"]