"""Render the presenter guide to PDF.

Pipeline: `docs/presenter-guide.md` -> styled HTML -> headless Edge print-to-PDF.

Edge is used rather than PyMuPDF's `Story` because `Story` is a reflowable-flow
engine built for reading screens, not paginated documents: it dropped roughly
80% of this content without raising. A browser already implements the CSS we
want - `@page` margins, background colours, border accents, table layout - and
does the pagination correctly.

Usage:
    python scripts/build_presenter_guide.py
    python scripts/build_presenter_guide.py --open     # also launch the PDF
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "docs" / "presenter-guide.md"
OUTPUT = PROJECT_ROOT / "docs" / "presenter-guide.pdf"

# Candidate locations, newest layout first.
EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
)

CSS = """
@page { size: A4; margin: 16mm 15mm; }

* { box-sizing: border-box; }

html {
  /* Chromium drops background colours when printing unless asked. */
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}

body {
  font-family: "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 9.7pt;
  line-height: 1.52;
  color: #1e293b;
  margin: 0;
  background: #ffffff;
}

h1 {
  font-size: 22pt; color: #0369a1; margin: 0 0 1mm 0; line-height: 1.1;
}
.kicker {
  color: #64748b; font-size: 8pt; text-transform: uppercase;
  letter-spacing: 0.9pt; margin: 0 0 5mm 0; padding-bottom: 3mm;
  border-bottom: 1.5pt solid #bae6fd;
}
h2 {
  font-size: 12.5pt; color: #0284c7; margin: 7mm 0 2.5mm 0;
  padding-bottom: 1.2mm; border-bottom: 0.5pt solid #cbd5e1;
  /* Stop a heading being orphaned at the foot of a page. */
  page-break-after: avoid; break-after: avoid;
}
h3 { font-size: 10.3pt; color: #334155; margin: 4.5mm 0 1.5mm 0;
     page-break-after: avoid; break-after: avoid; }
p { margin: 0 0 2.4mm 0; }
ul, ol { margin: 0 0 2.6mm 0; padding-left: 6.5mm; }
li { margin-bottom: 1.2mm; }
code {
  font-family: Consolas, "Courier New", monospace; font-size: 8.3pt;
  color: #92400e; background: #fef3c7; padding: 0.3mm 1.1mm;
  border-radius: 1mm;
}
blockquote.say {
  background: #eff6ff; border-left: 2.5pt solid #0ea5e9;
  padding: 2.6mm 3.2mm; margin: 0 0 3mm 0; color: #1e3a8a;
}
blockquote.warn {
  background: #fff7ed; border-left: 2.5pt solid #f97316;
  padding: 2.6mm 3.2mm; margin: 0 0 3mm 0; color: #7c2d12;
}
blockquote.tip {
  background: #ecfdf5; border-left: 2.5pt solid #10b981;
  padding: 2.6mm 3.2mm; margin: 0 0 3mm 0; color: #065f46;
}
blockquote.card {
  background: #f8fafc; border: 0.5pt solid #cbd5e1;
  padding: 2.8mm 3.2mm; margin: 0 0 3mm 0; color: #334155;
}
blockquote strong { color: #0f172a; }
table { border-collapse: collapse; width: 100%; margin: 0 0 3.5mm 0;
        page-break-inside: avoid; break-inside: avoid; }
th, td {
  border: 0.4pt solid #cbd5e1; padding: 1.7mm 2.2mm;
  vertical-align: top; text-align: left; font-size: 8.9pt;
}
th { background: #f1f5f9; color: #0369a1; font-weight: 600; }
tbody tr:nth-child(even) { background: #fafafa; }
hr { border: none; border-top: 0.5pt solid #e2e8f0; margin: 5mm 0; }
strong { color: #0f172a; }
em { color: #475569; }
"""


def find_browser() -> Path | None:
    for candidate in EDGE_CANDIDATES:
        if candidate.is_file():
            return candidate
    found = shutil.which("msedge") or shutil.which("chrome")
    return Path(found) if found else None


def _escape(text: str) -> str:
    """Escape first, so nothing in the source can inject markup."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _inline(text: str) -> str:
    escaped = _escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?!\w)", r"<em>\1</em>", escaped)
    return escaped


def markdown_to_html(markdown: str) -> str:
    """Convert the guide's small Markdown vocabulary to HTML.

    Not a general parser, deliberately: the source uses headings, paragraphs,
    lists, tables, blockquotes and callout divs, and a full implementation would
    bring escaping edge cases the document does not need.
    """
    out: list[str] = []
    in_list: str | None = None
    in_table = False
    # A callout accumulates lines until its closing </div>; a say-quote is a
    # single line in this document, so it is emitted whole.
    callout: str | None = None

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append(f"</{in_list}>")
            in_list = None

    def close_table() -> None:
        nonlocal in_table
        if in_table:
            out.append("</tbody></table>")
            in_table = False

    def close_callout() -> None:
        nonlocal callout
        if callout:
            out.append("</blockquote>")
            callout = None

    def close_all() -> None:
        close_list()
        close_table()
        close_callout()

    title = ""
    for raw in markdown.splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            # A blank line ends every open construct. Without this an
            # unterminated blockquote swallows the rest of the document.
            close_all()
            continue

        if re.fullmatch(r"-{3,}", stripped):
            close_all()
            out.append("<hr/>")
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            close_all()
            level = min(len(heading.group(1)), 3)
            if level == 1 and not title:
                title = re.sub(r"^#\s+", "", stripped)
                out.append(f"<h1>{_inline(title)}</h1>")
                out.append(
                    '<p class="kicker">ASTRA INTEL &middot; screen-recorded '
                    "walkthrough &middot; 12&ndash;15 minutes</p>"
                )
            else:
                out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            continue

        # Callout divs carry formatted text a one-line blockquote cannot, so
        # their body is accumulated until the closing tag.
        opening = re.match(r'^<div class="(warn|tip|card)">$', stripped)
        if opening:
            close_all()
            callout = opening.group(1)
            out.append(f'<blockquote class="{callout}">')
            continue
        if stripped == "</div>" and callout:
            close_callout()
            continue
        if callout:
            # A blank-ish line inside the callout starts a fresh paragraph.
            if out[-1].endswith("</blockquote>") or "<p>" in out[-1]:
                out[-1] = out[-1][: -len("</blockquote>")] + "<br/>"
            out[-1] = out[-1][: -len("</blockquote>")] + _inline(stripped) + "</blockquote>"
            continue

        if stripped.startswith(">"):
            close_all()
            out.append(f'<blockquote class="say">{_inline(stripped[1:].lstrip())}</blockquote>')
            continue

        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            close_list()
            close_callout()
            if not in_table:
                out.append("<table><tbody>")
                in_table = True
            if all(c.startswith("**") for c in cells):
                out.append(
                    "<tr>"
                    + "".join(f"<th>{_inline(c.strip('*'))}</th>" for c in cells)
                    + "</tr>"
                )
            else:
                out.append(
                    "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>"
                )
            continue

        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        numbered = re.match(r"^\s*\d+[.)]\s+(.*)$", line)
        if bullet or numbered:
            close_table()
            close_callout()
            kind = "ul" if bullet else "ol"
            if in_list != kind:
                close_list()
                out.append(f"<{kind}>")
                in_list = kind
            out.append(f"<li>{_inline((bullet or numbered).group(1))}</li>")
            continue

        close_all()
        out.append(f"<p>{_inline(stripped)}</p>")

    close_all()
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true", help="open the PDF when done")
    args = parser.parse_args()

    if not SOURCE.is_file():
        print(f"missing source: {SOURCE}", file=sys.stderr)
        return 1

    browser = find_browser()
    if browser is None:
        print(
            "no Chromium browser found; install Edge or Chrome, or set one of:\n"
            + "\n".join(str(c) for c in EDGE_CANDIDATES),
            file=sys.stderr,
        )
        return 1

    body = markdown_to_html(SOURCE.read_text(encoding="utf-8"))
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>ASTRA INTEL - Presenter Guide</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "guide.html"
        page.write_text(html, encoding="utf-8")
        profile = Path(tmp) / "profile"
        result = subprocess.run(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--no-first-run",
                "--no-pdf-header-footer",
                f"--user-data-dir={profile}",
                f"--print-to-pdf={OUTPUT}",
                page.as_uri(),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if not OUTPUT.is_file():
            print(result.stdout[-800:], file=sys.stderr)
            print(result.stderr[-800:], file=sys.stderr)
            print("print-to-pdf produced no file", file=sys.stderr)
            return 1

    import pymupdf

    with pymupdf.open(OUTPUT) as doc:
        pages = doc.page_count
        words = sum(len(page.get_text().split()) for page in doc)

    if words < 400:
        print(f"suspiciously short PDF: {words} words", file=sys.stderr)
        return 1

    size_kb = OUTPUT.stat().st_size / 1024
    print(
        f"wrote {OUTPUT.relative_to(PROJECT_ROOT)} - "
        f"{pages} pages, ~{words} words, {size_kb:.0f} KB"
    )

    if args.open:
        subprocess.Popen(["cmd", "/c", "start", "", str(OUTPUT)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())