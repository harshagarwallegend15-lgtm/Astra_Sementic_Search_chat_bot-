"""Screenshot the live console's Overview view and confirm the hero renders.

Rendering is the whole deliverable here, so it is checked by loading the real
page in a browser at a desktop viewport, asserting the hero actually painted
pixels rather than sitting black, and capturing an image to look at.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

PROJECT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8502"
OUT = Path(r"C:\Users\harsh\AppData\Local\Temp\opencode\hero.png")

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

PROBE = """
(async () => {
  const out = [];
  window.addEventListener('error', e => out.push('ERROR ' + e.message));
  await new Promise(r => setTimeout(r, 2500));
  const cv = document.getElementById('hero-canvas');
  out.push('canvas: ' + (cv.width + 'x' + cv.height));
  let painted = 0;
  try {
    const ctx = cv.getContext('2d');
    const d = ctx.getImageData(0, 0, cv.width, cv.height).data;
    for (let i = 3; i < d.length; i += 4 * 97) if (d[i] > 0) painted++;
    out.push('painted samples: ' + painted);
  } catch (e) { out.push('readback failed: ' + e.message); }
  out.push('video has is-live: ' + document.getElementById('hero-video').classList.contains('is-live'));
  out.push('docs: ' + document.getElementById('hero-docs').textContent);
  out.push('passages: ' + document.getElementById('hero-passages').textContent);
  out.push('pages: ' + document.getElementById('hero-pages').textContent);
  out.push('model: ' + document.getElementById('hero-model').textContent);
  out.push('home bars rows: ' + document.querySelectorAll('#home-bars .bar-row').length);
  out.push('nav items: ' + document.querySelectorAll('.nav-item').length);
  document.title = 'HERO:' + JSON.stringify(out);
})();
"""

probe = PROJECT / "static" / "_hero_probe.html"
_index = (PROJECT / "static" / "index.html").read_text(encoding="utf-8")
# Matched by prefix: cache-busting appended ?v=<hash> to the script tag, and
# matching the bare tag now fails silently, so the probe never installs.
_open = _index.find('<script src="/static/app.js')
if _open == -1:
    raise SystemExit("app.js is not referenced from index.html")
_close = _index.index("</script>", _open) + len("</script>")
probe.write_text(
    _index[:_close] + f"<script>{PROBE}</script>" + _index[_close:],
    encoding="utf-8",
)


def main() -> int:
    import json
    import re
    import urllib.request

    try:
        urllib.request.urlopen(BASE + "/api/health", timeout=10).read()
    except Exception as exc:  # noqa: BLE001
        probe.unlink(missing_ok=True)
        print(f"console not running on {BASE}: {exc}", file=sys.stderr)
        return 1

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            dom = subprocess.run(
                [
                    str(BROWSER), "--headless=new", "--disable-gpu", "--no-sandbox",
                    "--no-first-run", "--window-size=1600,1000",
                    "--virtual-time-budget=20000",
                    f"--user-data-dir={tmpdir / 'p1'}",
                    "--dump-dom", f"{BASE}/static/_hero_probe.html",
                ],
                capture_output=True, timeout=300,
                encoding="utf-8", errors="replace",
            ).stdout
            shot = tmpdir / "hero.pdf"
            subprocess.run(
                [
                    str(BROWSER), "--headless=new", "--disable-gpu", "--no-sandbox",
                    "--no-first-run", "--window-size=1600,1000",
                    "--virtual-time-budget=20000",
                    f"--user-data-dir={tmpdir / 'p2'}",
                    f"--print-to-pdf={shot}", f"{BASE}/static/_hero_probe.html",
                ],
                capture_output=True, timeout=300,
                encoding="utf-8", errors="replace",
            )
            if shot.is_file():
                OUT.parent.mkdir(parents=True, exist_ok=True)
                with pymupdf.open(shot) as doc:
                    doc[0].get_pixmap(dpi=120).save(OUT)
    finally:
        probe.unlink(missing_ok=True)

    m = re.search(r"HERO:(\[.*?\])</title>", dom, re.S)
    if not m:
        print("no probe result", file=sys.stderr)
        return 1
    findings = json.loads(
        m.group(1).replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<")
    )
    for line in findings:
        print(line)

    blob = " ".join(findings)
    painted = 0
    for line in findings:
        if line.startswith("painted samples:"):
            painted = int(line.split(":")[1].strip())
    checks = {
        "the canvas has real dimensions": "0x0" not in blob,
        "the canvas actually painted": painted > 50,
        "no runtime error": "ERROR " not in blob,
        "documents are shown": "docs: 3" in blob,
        "passages are shown": "515" in blob,
        "the model is shown": "gpt-oss-120b" in blob,
        "three nav destinations": "nav items: 3" in blob,
        "corpus bars rendered": "home bars rows: 3" in blob,
    }
    print()
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    if OUT.is_file():
        print(f"\nscreenshot: {OUT}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
