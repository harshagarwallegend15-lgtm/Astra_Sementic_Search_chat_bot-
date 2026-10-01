"""The container's dependency set is a subset of the full one.

`requirements.txt` installs Streamlit, pytest and httpx for local development
and the Streamlit front end. The image installs `requirements-container.txt`
instead, which omits them to keep the build fast and the image small.

The risk is drift: someone adds an import to `src/` that only the full
requirements.txt satisfies, and the failure surfaces as a build error on the
deployment platform rather than in a test. So this asserts the trimmed set
still covers everything the API actually imports.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTAINER_REQUIREMENTS = PROJECT_ROOT / "requirements-container.txt"
FULL_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

#: Distribution name -> import name, where they differ.
_DIST_TO_IMPORT = {
    "faiss-cpu": "faiss",
    "python-multipart": "multipart",
    "pymupdf": "pymupdf",
    "rank-bm25": "rank_bm25",
    "python-dotenv": "dotenv",
    "sentence-transformers": "sentence_transformers",
    "langchain-core": "langchain_core",
    "langchain-community": "langchain_community",
    "langchain-openai": "langchain_openai",
    "langchain-ollama": "langchain_ollama",
    "uvicorn[standard]": "uvicorn",
    "pydantic": "pydantic",
    "numpy": "numpy",
    "fastapi": "fastapi",
}


def _parse(path: Path) -> dict[str, str]:
    """Map distribution name to the module that proves it is installed."""
    packages: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!\[]", line, maxsplit=1)[0].strip().lower()
        packages[name] = _DIST_TO_IMPORT.get(name, name.replace("-", "_"))
    return packages


def test_the_trimmed_set_is_a_strict_subset():
    full = _parse(FULL_REQUIREMENTS)
    container = _parse(CONTAINER_REQUIREMENTS)
    unknown = sorted(set(container) - set(full))
    assert not unknown, f"not in requirements.txt: {unknown}"


@pytest.mark.parametrize("module", sorted(_parse(CONTAINER_REQUIREMENTS).values()))
def test_every_container_dependency_imports(module):
    __import__(module)


def test_the_omitted_packages_are_genuinely_unused_by_the_api():
    """Streamlit, pytest and httpx must not be imported by the served app.

    If `src/` or `api.py` ever imports one of these, removing it from the
    container requirements would break the deployment, so this fails loudly at
    test time instead.

    ``src/ui.py`` is exempt: it is the Streamlit front end consumed by app.py
    and is not in the API's import graph, so its streamlit import is never
    executed in the container.
    """
    served = [PROJECT_ROOT / "api.py", *sorted((PROJECT_ROOT / "src").glob("*.py"))]
    exempt = {PROJECT_ROOT / "src" / "ui.py"}
    offenders: list[str] = []
    for path in served:
        if path in exempt:
            continue
        text = path.read_text(encoding="utf-8")
        for banned in ("streamlit", "pytest", "httpx"):
            if re.search(rf"^\s*(import|from)\s+{banned}\b", text, re.MULTILINE):
                offenders.append(f"{path.name} imports {banned}")
    assert not offenders, offenders


def test_the_dockerfile_installs_the_trimmed_set():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "requirements-container.txt" in dockerfile, (
        "the image must install the trimmed set, not the full one"
    )
    # A bare `-r requirements.txt` would silently undo the trim. Line
    # continuations are joined first, so a split install line still matches.
    joined = re.sub(r"\\\s*\n\s*", " ", dockerfile)
    install_lines = [
        line for line in joined.splitlines()
        if re.search(r"\bpip install\b", line)
    ]
    assert install_lines, "no pip install found in the Dockerfile"
    for line in install_lines:
        assert "-r requirements-container.txt" in line, line
        assert re.search(r"-r\s+requirements\.txt\b", line) is None, line


def test_the_container_serves_the_api_not_streamlit():
    """A wrong CMD deploys the wrong application, silently and successfully.

    The previous Dockerfile ran Streamlit, so a deploy would have come up green
    while serving an entirely different front end over the wrong port. The last
    CMD is the entry point: the earlier one belongs to HEALTHCHECK.
    """
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    cmds = re.findall(r"^CMD\s+(.+)$", dockerfile, re.MULTILINE)
    assert cmds, "no CMD instruction in the Dockerfile"
    cmd = cmds[-1]
    assert "api.py" in cmd, f"CMD does not start the API: {cmd!r}"
    assert "streamlit" not in cmd.lower()
    # .dockerignore must not exclude the module the CMD runs.
    ignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert not re.search(r"^api\.py$", ignore, re.MULTILINE), (
        "api.py is excluded by .dockerignore but the CMD runs it"
    )


def test_the_api_binds_to_the_injected_port():
    """Render and Railway both inject PORT and proxy to 0.0.0.0.

    A server bound to 127.0.0.1 is unreachable from the platform's proxy, and
    it looks healthy locally, so the only symptom is a deploy that never serves.
    """
    source = (PROJECT_ROOT / "api.py").read_text(encoding="utf-8")
    main = source.split('if __name__ == "__main__"')[-1]
    assert 'os.environ.get("PORT"' in main, "PORT is not read"
    assert "0.0.0.0" in main, "the server is never told to bind all interfaces"


def test_a_render_blueprint_exists_and_uses_a_persistent_disk():
    """Without the disk the index is lost on every restart and re-seeded.

    Render's `free` plan also evicts the filesystem after inactivity, so a paid
    plan is required for a real corpus.
    """
    blueprint = PROJECT_ROOT / "render.yaml"
    assert blueprint.is_file(), "render.yaml is missing"
    text = blueprint.read_text(encoding="utf-8")
    assert "dockerfilePath" in text
    assert "healthCheckPath: /api/health" in text
    assert "mountPath: /app/data" in text, "no persistent volume for the index"
    assert "LLM_API_KEY" in text and "sync: false" in text, (
        "the API key must be supplied in the dashboard, never committed"
    )
    # A committed key would be the worst possible outcome of this file.
    assert not re.search(r"gsk_[A-Za-z0-9]{20,}", text)


def test_the_blueprint_only_uses_fields_render_actually_accepts():
    """`healthCheckTimeout` failed the Blueprint; every unknown key costs a
    deploy cycle. `tests/test_render_blueprint.py` validates against Render's
    live schema; this is the offline check for the keys that bit us."""
    text = (PROJECT_ROOT / "render.yaml").read_text(encoding="utf-8")
    for invented in (
        "healthCheckTimeout",
        "startupTimeout",
        "healthCheckInterval",
        "healthCheckRetries",
    ):
        assert f"{invented}:" not in text, f"{invented} is not a Render field"
    assert "plan: starter" not in text, "'starter' is not a Render plan id"
