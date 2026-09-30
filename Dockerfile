# syntax=docker/dockerfile:1
#
# ASTRA INTEL - Streamlit app image.
#
# Notes:
#  * This image is large (torch + sentence-transformers + faiss). If you only
#    need to run the app locally, a virtualenv is lighter and faster.
#  * By default this image runs in "local LLM" mode: it talks to an Ollama
#    server you provide on the host, so no API key is baked in. Run the app
#    container with:
#        docker run --rm -p 8501:8501 \
#            -e LLM_BASE_URL=http://host.docker.internal:11434 \
#            -v "$(pwd)/data:/app/data" astra-intel
#  * Never bake an API key into the image. Pass it at run time with -e.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    OLLAMA_HOST=0.0.0.0:11434

# libGL/libglib are needed by opencv-style transitive deps; build-essential lets
# pip fall back to source builds for wheels that are missing on this platform.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch from the CPU-only index so the image does not pull ~2 GB of
# CUDA libraries that a CPU host will never use.
COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Warm the embedding model into the image so the first query is not slow.
# Skipped automatically when EMBEDDING_BACKEND=hashing.
COPY src/ ./src/
RUN python -c "\
from src.config import get_settings;\
from src.embeddings import get_embedder;\
e = get_embedder(get_settings());\
print('embedding backend ready:', e.name, e.dimensions)"

COPY app.py ./
COPY sample-documents/ ./sample-documents/
COPY sample-metadata.json ./

RUN mkdir -p data/vectorstore data/uploads

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Run as a non-root user.
RUN useradd --create-home --uid 10001 astra \
    && chown -R astra:astra /app
USER astra

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
