"""End-to-end check of the live console at 127.0.0.1:8502.

Exercises every endpoint app.js calls, in the order the browser would, and
prints a PASS/FAIL line per step. Read-only steps are safe; the mutation steps
(clear, rebuild, upload, delete) restore the index before exiting.
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
import urllib.request

import pymupdf

BASE = "http://127.0.0.1:8502"
results: list[tuple[bool, str, str]] = []


def _decode(content_type: str, body: bytes):
    if not body:
        return None
    if "json" in content_type:
        return json.loads(body)
    return body.decode("utf-8", errors="replace")


def call(path, payload=None, method=None, raw=None, content_type=None, timeout=120):
    data = raw
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(
        BASE + path, data=data, headers=headers, method=method
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            ms = (time.perf_counter() - started) * 1000
            return response.status, _decode(response.headers.get("Content-Type", ""), body), ms
    except urllib.error.HTTPError as err:
        body = err.read()
        ms = (time.perf_counter() - started) * 1000
        return err.code, _decode(err.headers.get("Content-Type", ""), body), ms


def check(name, ok, detail=""):
    results.append((ok, name, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  --  {detail}" if detail else ""))
    return ok


def multipart(payload: bytes, filename: str) -> tuple[bytes, str]:
    boundary = "astra-verify-boundary"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode()
    return head + payload + f"\r\n--{boundary}--\r\n".encode(), (
        f"multipart/form-data; boundary={boundary}"
    )


def one_page_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    paragraphs = [
        "ASTRA console upload verification.",
        "Unmanned ground vehicles are platforms that carry out a mission with "
        "no human crew onboard, under remote supervision.",
        "Sensors on the platform observe the environment and relay the data to "
        "a control station, which either supervises the mission directly or "
        "hands autonomy to an onboard controller.",
        "Because the operator stays at a distance, loss of the communications "
        "link is the single most consequential failure mode, and most designs "
        "therefore hold a pre-planned abort behaviour that engages when the "
        "signal is lost for too long.",
    ]
    y = 740
    for index, text in enumerate(paragraphs):
        page.insert_text((72, y), text, fontsize=11)
        y -= 14 if index == 0 else 26
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ------------------------------------------------------------ static shell --

for path, floor in (("/", 9000), ("/static/app.js", 30000), ("/static/styles.css", 30000)):
    status, body, ms = call(path)
    check(
        f"GET {path}",
        status == 200 and isinstance(body, str) and len(body) > floor,
        f"{status}, {len(body) if isinstance(body, str) else 0} chars, {ms:.0f}ms",
    )

# ------------------------------------------------------------------- state --

status, state, ms = call("/api/state")
check("GET /api/state", status == 200 and "documents" in state, f"{status} in {ms:.0f}ms")

docs = state.get("documents", [])
check(
    "state carries real documents",
    len(docs) == 3,
    f"{len(docs)} docs, {sum(d['chunk_count'] for d in docs)} passages",
)
check(
    "state.runtime exposes the readouts the strip needs",
    all(
        k in state.get("runtime", {})
        for k in ("llm_provider", "llm_model", "embedding_model", "llm_ready")
    ),
    json.dumps(state.get("runtime", {}), default=str)[:120],
)
check("state.llm_ready is a real flag", state.get("llm_ready") is True)

page_rows = all(
    isinstance(d.get("pages"), list) and len(d["pages"]) == d["page_count"]
    for d in docs
)
check("every document carries a per-page density array", page_rows)

# ----------------------------------------------------------------- health --

status, health, ms = call("/api/health")
check(
    "GET /api/health reports ready",
    status == 200 and health.get("status") == "ok" and health.get("llm_ready"),
    f"{health.get('status')}, {ms:.0f}ms",
)

# ------------------------------------------------------------- suggestions --

status, body, _ = call("/api/suggestions")
check(
    "GET /api/suggestions fills the chips",
    status == 200 and len(body.get("questions", [])) >= 3,
    f"{len(body.get('questions', []))} questions",
)

# --------------------------------------------------------------------- ask --

QUESTION = "What are the three primary mission types of electronic warfare?"
ask_resp = None
ask_status = None
ask_elapsed = 0
for attempt in range(3):
    started = time.perf_counter()
    ask_status, ask_resp, _ = call("/api/ask", {"question": QUESTION}, timeout=180)
    ask_elapsed = time.perf_counter() - started
    if ask_status == 200 and ask_resp and ask_resp.get("citations"):
        break

check(
    "POST /api/ask returns a grounded cited answer",
    ask_status == 200
    and ask_resp
    and ask_resp.get("status") == "grounded"
    and ask_resp.get("citations"),
    f"{ask_resp.get('status') if ask_resp else ask_status} conf={ask_resp.get('confidence') if ask_resp else None} "
    f"cites={len(ask_resp.get('citations') or []) if ask_resp else 0} {ask_elapsed:.0f}s",
)
check(
    "answer cites at least one indexed document",
    ask_resp and any(c.get("filename") for c in (ask_resp.get("citations") or [])),
    ", ".join(sorted({c["filename"] for c in (ask_resp.get("citations") or [])}))[:90]
    if (ask_resp and ask_resp.get("citations"))
    else "none",
)
check("ask latency is usable", ask_elapsed < 30, f"{ask_elapsed:.1f}s")

status, off, _ = call("/api/ask", {"question": "   "})
check("POST /api/ask rejects an empty question", status == 422, str(status))

# ---------------------------------------------------------------- evidence --

status, ev, _ = call("/api/evidence", {"question": "How is a UGV controlled at distance?"})
check(
    "POST /api/evidence retrieves without calling the model",
    status == 200 and ev.get("citations"),
    f"{len(ev.get('citations', []))} citations",
)

# ---------------------------------------------------------------- briefings --

status, briefs, ms = call("/api/briefings", {}, timeout=600)
check(
    "POST /api/briefings returns one summary per document",
    status == 200 and len(briefs.get("summaries", [])) == len(docs),
    f"{len(briefs.get('summaries', []))} summaries in {ms / 1000:.0f}s",
)

target = docs[0]["document_id"]
status, single, _ = call(f"/api/briefings?document_id={target}", {}, timeout=600)
check(
    "briefing one document returns only that document",
    status == 200
    and len(single.get("summaries", [])) == 1
    and single["summaries"][0]["document_id"] == target,
    single["summaries"][0]["title"] if status == 200 else str(single),
)

status, missing, _ = call("/api/briefings?document_id=does-not-exist", {})
check("briefing an unknown document is a 404", status == 404, str(status))

# ------------------------------------------------------- destructive guards --

for path in ("/api/rebuild", "/api/clear"):
    status, body, _ = call(path, {})
    check(
        f"POST {path} refuses without a confirmation token",
        status in (400, 422),
        f"{status}: {str(body.get('detail'))[:60]}",
    )

status, _, _ = call("/api/documents/does-not-exist", method="DELETE")
check("DELETE unknown document is a 404", status == 404, str(status))

# ----------------------------------------------------------------- upload --

raw, ctype = multipart(b"%PDF-1.4\nnot really a pdf\n%%EOF", "corrupt.pdf")
status, body, _ = call("/api/upload", raw=raw, content_type=ctype)
check(
    "POST /api/upload rejects a corrupt PDF with a specific message",
    status == 400 and "could not be read" in str(body.get("detail", "")),
    str(body.get("detail"))[:110],
)

pdf = one_page_pdf()
raw, ctype = multipart(pdf, "astra-verification.pdf")
status, uploaded, _ = call("/api/upload", raw=raw, content_type=ctype, timeout=300)
check(
    "POST /api/upload ingests a valid PDF",
    status == 200 and uploaded and uploaded.get("added"),
    f"{status}: {str(uploaded)[:200]}",
)

added = (uploaded or {}).get("added") or []
uploaded_id = added[0]["document_id"] if added else None

status, dup, _ = call("/api/upload", raw=raw, content_type=ctype, timeout=300)
check(
    "re-uploading the same file is skipped as a duplicate",
    status == 200 and dup and len(dup.get("duplicates", [])) == 1,
    f"duplicates={len(dup.get('duplicates', []))}",
)

status, state2, _ = call("/api/state")
check(
    "the register reflects the new document",
    status == 200 and len(state2.get("documents", [])) == len(docs) + 1,
    f"{len(state2.get('documents', []))} docs",
)

if uploaded_id:
    status, _, _ = call(f"/api/documents/{uploaded_id}", method="DELETE")
    check("DELETE the uploaded document", status == 200, str(status))
    status, state3, _ = call("/api/state")
    check(
        "the register returns to its original size",
        len(state3.get("documents", [])) == len(docs),
        f"{len(state3.get('documents', []))} docs",
    )

# ---------------------------------------------------------------- summary --

failed = [name for ok, name, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("failed: " + "; ".join(failed))
sys.exit(1 if failed else 0)