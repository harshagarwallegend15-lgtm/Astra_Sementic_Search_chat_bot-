"""Screenshot each view of the live console, so a theme change can be eyeballed.

The glass rework touched every panel, tile, badge and button in the app, not
just the hero. Markup assertions confirm the CSS is present; they cannot tell
you the console still looks right. This captures each destination.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

PROJECT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8502"
OUTDIR = Path(r"C:\Users\harsh\AppData\Local\Temp\opencode")

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

VIEWS = ("home", "query", "documents")


def harness(view: str) -> str:
    return f"""
<script>
window.addEventListener('load', () => {{
  // Give the app time to fetch state before switching and capturing.
  setTimeout(() => {{
    const item = document.querySelector('.nav-item[data-view="{view}"]');
    if (item) item.click();
    if ("IntersectionObserver" in window && "{view}" !== "home") {{
      // Force the hero canvas to stop so it does not paint over other views.
      document.getElementById('view-home').classList.remove('is-active');
    }}
    document.title = 'VIEW:{view}';
  }}, 2200);
}});
</script>
"""


def main() -> int:
    import urllib.request

    try:
        urllib.request.urlopen(BASE + "/api/health", timeout=10).read()
    except Exception as exc:  # noqa: BLE001
        print(f"console not running on {BASE}: {exc}", file=sys.stderr)
        return 1

    OUTDIR.mkdir(parents=True, exist_ok=True)
    index = (PROJECT / "static" / "index.html").read_text(encoding="utf-8")
    probe = PROJECT / "static" / "_views_probe.html"

    written = []
    try:
        for view in VIEWS:
            # Matched by prefix: cache-busting appended ?v=<hash>, and a bare
            # tag match now fails silently, leaving the harness uninstalled.
            open_tag = index.find('<script src="/static/app.js')
            if open_tag == -1:
                print("app.js is not referenced from index.html", file=sys.stderr)
                return 1
            close = index.index("</script>", open_tag) + len("</script>")
            probe.write_text(
                index[:close] + harness(view) + index[close:],
                encoding="utf-8",
            )
            with tempfile.TemporaryDirectory() as tmp:
                tmpdir = Path(tmp)
                shot = tmpdir / f"{view}.pdf"
                subprocess.run(
                    [
                        str(BROWSER), "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--no-first-run", "--window-size=1500,1180",
                        "--virtual-time-budget=18000",
                        f"--user-data-dir={tmpdir / 'p'}",
                        f"--print-to-pdf={shot}",
                        f"{BASE}/static/_views_probe.html",
                    ],
                    capture_output=True, timeout=300,
                    encoding="utf-8", errors="replace",
                )
                if not shot.is_file():
                    print(f"no render for {view}", file=sys.stderr)
                    continue
                target = OUTDIR / f"view-{view}.png"
                with pymupdf.open(shot) as doc:
                    doc[0].get_pixmap(dpi=110).save(target)
                written.append(target)
    finally:
        probe.unlink(missing_ok=True)

    for path in written:
        print(f"wrote {path}")
    return 0 if len(written) == len(VIEWS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
