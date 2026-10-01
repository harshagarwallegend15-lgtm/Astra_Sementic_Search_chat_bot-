"""Prove the home-page backdrop is actually moving, rather than merely configured to.

Markup assertions and CSS contracts cannot answer the question that was
reported: "there is no background image motion". Two things can be true at
once — the animation exists in the stylesheet and nothing visibly moves — and
that is what happened the first time: a few percent of travel over two minutes
is roughly one pixel per second.

So this renders the backdrop on its own at two points on the animation
timeline and measures the difference between the frames. A static backdrop
scores ~0; anything above the floor has demonstrably moved. A second pass reads
computed styles out of the PDF text layer, which says *why*.

Timeline control matters. --virtual-time-budget does not advance CSS animations
under --print-to-pdf, so waiting between captures yields two identical frames
and a false failure. Each layer is therefore paused at a chosen negative
animation-delay, which freezes the whole stack at a known instant without
depending on when the page happens to be printed.

Requires a live console on 127.0.0.1:8502 and Microsoft Edge.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops, ImageStat

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

# A frame-to-frame change smaller than this is compression noise, not motion.
MOTION_FLOOR = 2.0

PROBE_SCRIPT = """
<script>
(() => {
  const read = (sel) => {
    const el = document.querySelector(sel);
    if (!el) return { missing: true };
    const s = getComputedStyle(el);
    return { opacity: s.opacity, transform: s.transform };
  };
  const pre = document.createElement('pre');
  pre.id = 'dbg';
  pre.textContent = 'PROBE ' + JSON.stringify({
    animations: document.getAnimations().length,
    frames: [0, 1, 2, 3, 4, 5].map(i => read('.map-photo.p' + i)),
    contours: read('.map-contours'),
  });
  document.body.appendChild(pre);
})();
</script>
"""


def probe_style(sample_ms: int) -> str:
    return f"""
<style>
  /* The console chrome paints over the backdrop and would dominate the diff.
     Removing it leaves the layer stack alone to measure. */
  .shell {{ display: none !important; }}
  /* Freeze every ambient layer at `sample_ms` into the timeline. The reels
     keep their stagger, because six frames frozen at the same instant would
     stack instead of dissolving and the check would measure the wrong thing. */
  .map-photo {{
    animation-play-state: paused !important;
    animation-delay: calc(var(--i) * 20s - {sample_ms}ms) !important;
  }}
  .map-contours, .map-grid, .map-sectors, .map-schematic,
  .orb, .grid-veil, .sweep {{
    animation-play-state: paused !important;
    animation-delay: -{sample_ms}ms !important;
  }}
  #dbg {{ position: fixed; left: 8px; top: 8px; z-index: 99; margin: 0;
          font: 10px/1.3 monospace; color: #9ff; background: #000;
          padding: 6px; width: 320px; white-space: pre-wrap;
          word-break: break-all; }}
</style>
"""


def _with_probe(index: str, extra: str) -> str:
    """Append the probe after app.js, whatever cache-busting suffix it carries.

    Matching the bare tag stopped working the moment ?v=<hash> was added, and
    the failure is silent: the harness simply never runs.
    """
    open_tag = index.find('<script src="/static/app.js')
    if open_tag == -1:
        raise AssertionError("app.js is not referenced from index.html")
    close = index.index("</script>", open_tag) + len("</script>")
    return index[:close] + extra + index[close:]


def render(tmpdir: Path, sample_ms: int, probe: bool) -> Path | None:
    index = (PROJECT / "static" / "index.html").read_text(encoding="utf-8")
    injected = _with_probe(index, probe_style(sample_ms) + (PROBE_SCRIPT if probe else ""))
    # Served over HTTP, not file://: app.js is a root-relative URL, so a file
    # origin resolves it to /static/app.js on disk, 404s, and takes every
    # harness below with it.
    page = PROJECT / "static" / "_motion_probe.html"
    page.write_text(injected, encoding="utf-8")
    shot = Path(tmpdir) / f"shot-{sample_ms}{'-probe' if probe else ''}.pdf"
    try:
        subprocess.run(
            [
                str(BROWSER), "--headless=new", "--disable-gpu", "--no-sandbox",
                "--no-first-run", "--window-size=1500,1180",
                f"--user-data-dir={Path(tmpdir) / f'p{sample_ms}{int(probe)}'}",
                f"--print-to-pdf={shot}",
                f"{BASE}/static/_motion_probe.html",
            ],
            capture_output=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
    finally:
        page.unlink(missing_ok=True)
    return shot if shot.is_file() else None


def to_png(pdf: Path, target: Path) -> None:
    with pymupdf.open(pdf) as doc:
        doc[0].get_pixmap(dpi=96).save(target)


def main() -> int:
    if BROWSER is None:
        print("no Chromium browser found", file=sys.stderr)
        return 1
    try:
        import urllib.request

        urllib.request.urlopen(BASE + "/api/health", timeout=10).read()
    except Exception as exc:  # noqa: BLE001
        print(f"console not running on {BASE}: {exc}", file=sys.stderr)
        return 1

    OUTDIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        # Two points inside one frame's slot prove the push is moving; a third
        # point in the next slot proves the sequence actually changes.
        samples = (4000, 16000, 26000)
        frames = []
        for sample in samples:
            shot = render(tmp, sample, probe=False)
            if shot is None:
                print(f"no render at {sample}ms", file=sys.stderr)
                return 1
            png = OUTDIR / f"backdrop-{sample}.png"
            to_png(shot, png)
            frames.append(Image.open(png).convert("RGB"))

        # Compare the upper two thirds: everything below it is where the shell
        # sits, and its now-empty space says nothing about layer motion.
        width, height = frames[0].size
        box = (0, 0, width, int(height * 0.66))
        cropped = [frame.crop(box) for frame in frames]

        def delta(a: Image.Image, b: Image.Image) -> float:
            return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3

        push = delta(cropped[0], cropped[1])
        handover = delta(cropped[1], cropped[2])
        print(f"push between {samples[0]}ms and {samples[1]}ms: {push:.2f}/255")
        print(f"handover between {samples[1]}ms and {samples[2]}ms: {handover:.2f}/255")
        if push < MOTION_FLOOR:
            failures.append(
                f"the frame moved by only {push:.2f}/255 inside its slot; "
                "it is not visibly pushing"
            )
        if handover < MOTION_FLOOR:
            failures.append(
                f"the sequence changed by only {handover:.2f}/255 between slots; "
                "it is not changing scenes"
            )

        shot = render(tmp, 30000, probe=True)
        if shot is None:
            print("no diagnostic render", file=sys.stderr)
            return 1
        with pymupdf.open(shot) as doc:
            # The pre is wider than the page, so the extractor inserts
            # newlines mid-payload. The JSON is single-line, so flattening
            # the wraps restores it.
            text = doc[0].get_text().replace("\n", "")
        start = text.find("PROBE {")
        data = None
        if start != -1:
            # The payload is nested JSON, so a non-greedy match would stop at
            # the first closing brace; balance them instead.
            depth = 0
            for offset, char in enumerate(text[start + 6 :], start=start + 6):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(text[start + 6 : offset + 1])
                        except json.JSONDecodeError:
                            data = None
                        break
        if data is None:
            failures.append("diagnostics were not printed into the PDF")
        else:
            print(f"animations tracked: {data['animations']}")
            visible = 0
            strongest = 0.0
            for index, frame in enumerate(data["frames"]):
                if frame.get("missing"):
                    failures.append(f"frame {index} is missing from the DOM")
                    continue
                opacity = float(frame["opacity"])
                strongest = max(strongest, opacity)
                visible += opacity > 0.01
                print(f"  frame {index}: opacity={frame['opacity']} {frame['transform']}")
            # Sampled mid-handover, so one frame is handing over to the next;
            # what must never happen is the whole stack going dark in between.
            if strongest < 0.15:
                failures.append(
                    f"the strongest frame is only {strongest:.2f} opaque; "
                    "there is no picture on screen"
                )
            if visible < 2:
                failures.append(
                    f"only {visible} frames overlap mid-handover; the reel cuts"
                )
            if float(data["contours"]["opacity"]) <= 0.01:
                failures.append("the contour layer is transparent")
            if data["animations"] < 10:
                failures.append(f"only {data['animations']} animations are running")

    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if not failures:
        print("backdrop motion verified")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
