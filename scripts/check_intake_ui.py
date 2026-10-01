"""Drive the real intake UI in headless Edge and confirm a picked file appears.

The complaint was that a selected file never shows up, so asserting the markup
and the JS strings is not enough - the list has to actually render. This loads
the console in a browser, injects a real File into the file input, fires the
change event the picker would fire, and screenshots the result.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf

BROWSER = next(
    (p for p in (
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ) if p.is_file()),
    None,
)
if BROWSER is None:
    print("no Chromium browser found", file=sys.stderr)
    sys.exit(1)

# Loads the page, waits for /api/state, simulates a file selection, then
# prints what the list element contains as JSON into the DOM title.
HARNESS = """
(async () => {
  const log = [];
  window.addEventListener('error', e => log.push('WINDOW ERROR: ' + e.message));
  try {
    const input = document.getElementById('upload-input');
    log.push('input found: ' + !!input);

    // Open the Documents tab the way the operator does.
    document.querySelector('.nav-item[data-view="documents"]').click();
    await new Promise(r => setTimeout(r, 300));

    const list = document.getElementById('upload-list');
    const sub = document.getElementById('dz-sub');
    log.push('list hidden before: ' + list.hidden);

    // Build a real File and hand it to the input, exactly as the picker would.
    const bytes = new Uint8Array(340 * 1024).fill(37);
    const pdf = new File([bytes], 'ukraine-briefing-january-2026.pdf',
                         { type: 'application/pdf' });
    const dt = new DataTransfer();
    dt.items.add(pdf);
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
    await new Promise(r => setTimeout(r, 300));

    log.push('input.files.length: ' + input.files.length);
    log.push('list hidden after: ' + list.hidden);
    log.push('list html: ' + list.innerHTML.replace(/\\s+/g, ' ').trim());
    log.push('list text: ' + list.textContent.replace(/\\s+/g, ' ').trim());
    log.push('sub text: ' + sub.textContent.trim());
    log.push('submit enabled: ' + !document.getElementById('upload-submit').disabled);
    // The listener runs on the element, so an exception there never reaches
    // this try/catch; call it directly to see what it would do.
    try {
      window.__astra_preflight = typeof preflight;
    } catch (_) {}
  } catch (e) {
    log.push('ERROR: ' + e.message);
  }
  document.title = 'RESULT:' + JSON.stringify(log);
})();
"""

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
<script>
window.addEventListener('load', () => {
  const s = document.createElement('script');
  s.textContent = %s;
  document.body.appendChild(s);
});
</script></body></html>
"""


def main() -> int:
    import json
    import re
    import shutil
    import urllib.request

    project = Path(__file__).resolve().parents[1]
    static = project / "static"
    base = "http://127.0.0.1:8502"
    # Must live under static/ because api.py only serves "/" plus /static/*.
    probe = static / "_intake_probe.html"
    probe_url = base + "/static/_intake_probe.html"

# The probe must be served from the app's own origin so the relative
    # /static/app.js resolves and the DOM is the real console, not a mock.
    # Matched by prefix: cache-busting appends ?v=<hash>, and matching the bare
    # tag now fails silently, so the harness never installs and the run reports
    # a failure that has nothing to do with intake.
    index = (static / "index.html").read_text(encoding="utf-8")
    open_tag = index.find('<script src="/static/app.js')
    if open_tag == -1:
        raise SystemExit("app.js is not referenced from index.html")
    close_tag = index.index("</script>", open_tag) + len("</script>")
    probe.write_text(
        index[:close_tag] + f"<script>{HARNESS}</script>" + index[close_tag:],
        encoding="utf-8",
    )

    try:
        urllib.request.urlopen(base + "/api/health", timeout=10).read()
    except Exception as exc:  # noqa: BLE001
        probe.unlink(missing_ok=True)
        print(f"the console is not running on {base}: {exc}", file=sys.stderr)
        print("start it with:  python api.py", file=sys.stderr)
        return 1

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            dom = subprocess.run(
                [
                    str(BROWSER), "--headless=new", "--disable-gpu", "--no-sandbox",
                    "--no-first-run", "--virtual-time-budget=15000",
                    f"--user-data-dir={tmpdir / 'profile'}",
                    "--dump-dom", probe_url,
                ],
                capture_output=True, timeout=240, encoding='utf-8', errors='replace',
            ).stdout
            shot = tmpdir / "shot.pdf"
            subprocess.run(
                [
                    str(BROWSER), "--headless=new", "--disable-gpu", "--no-sandbox",
                    "--no-first-run", "--virtual-time-budget=15000",
                    f"--user-data-dir={tmpdir / 'profile2'}",
                    f"--print-to-pdf={shot}", probe_url,
                ],
                capture_output=True, text=True, timeout=240,
            )
            if shot.is_file():
                out = Path(r"C:\Users\harsh\AppData\Local\Temp\opencode\intake.png")
                out.parent.mkdir(parents=True, exist_ok=True)
                with pymupdf.open(shot) as doc:
                    doc[0].get_pixmap(dpi=110).save(out)
                print(f"screenshot: {out}")
    finally:
        probe.unlink(missing_ok=True)

    m = re.search(r"RESULT:(\[.*?\])</title>", dom, re.S)
    if not m:
        print("harness produced no result", file=sys.stderr)
        print(dom[-1200:], file=sys.stderr)
        return 1

    findings = json.loads(
        m.group(1).replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<")
    )
    for line in findings:
        print(line)

    errors = [f for f in findings if f.startswith("ERROR") or f.startswith("WINDOW ERROR")]
    blob = " ".join(findings)
    checks = {
        "the file input exists": "input found: true" in findings,
        "the picker assigned a file": "input.files.length: 1" in findings,
        "the list is revealed": "list hidden after: false" in findings,
        "the file name is shown": "ukraine-briefing-january-2026.pdf" in blob,
        "the size is shown": "KB" in blob or "MB" in blob,
        "the selection line confirms it": "ready to ingest" in blob,
        "no runtime error": not errors,
    }
    print()
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

