"""Procedurally generate layered defence-themed SVG artwork for the backdrop.

No raster images and no stock assets: everything is drawn as SVG, which is a
real image format, costs a few kilobytes, stays crisp at any display density,
and carries no licensing question. Each layer is a separate file so the CSS can
drift them at different speeds and parallax them.

Layers, back to front:
  1. `terrain-contours.svg`  topographic iso-lines, the cartographic base
  2. `coord-grid.svg`        coordinate graticule with tick labels
  3. `sector-arcs.svg`       range arcs and bearing spokes, dashed
  4. `schematic.svg`         technical callout marks and corner brackets

Deterministic: a fixed seed means the artwork is byte-identical on every run,
so a re-render never changes the look of a deployed page.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "static" / "media"
W, H = 1600, 1000
SEED = 20260201


def _value_noise(rng: random.Random, cols: int, rows: int):
    """A smooth random field, bilinearly interpolated from a coarse grid."""
    grid = [[rng.random() for _ in range(cols + 1)] for _ in range(rows + 1)]
    # Wrap the edges so the field is tileable; a seam across the backdrop is
    # exactly the kind of detail that gives a generated texture away.
    for row in grid:
        row[-1] = row[0]
    for x in range(cols + 1):
        grid[-1][x] = grid[0][x]

    def sample(u: float, v: float) -> float:
        fx, fy = u * cols, v * rows
        x0, y0 = int(fx), int(fy)
        x1, y1 = (x0 + 1) % cols, (y0 + 1) % rows
        tx, ty = fx - x0, fy - y0
        # Smoothstep, so the field has no visible linear creases.
        tx = tx * tx * (3 - 2 * tx)
        ty = ty * ty * (3 - 2 * ty)
        a = grid[y0][x0] * (1 - tx) + grid[y0][x1] * tx
        b = grid[y1][x0] * (1 - tx) + grid[y1][x1] * tx
        return a * (1 - ty) + b * ty

    return sample


def terrain_contours() -> str:
    """Topographic iso-lines: nested closed curves over a noise field."""
    rng = random.Random(SEED)
    noise = _value_noise(rng, 6, 4)
    cx, cy = W * 0.5, H * 0.5
    max_r = math.hypot(W, H) * 0.52
    levels = 26

    paths = []
    for i in range(levels):
        # Non-linear spacing gives the characteristic contour clustering.
        t = (i / levels) ** 1.35
        base = 40 + t * max_r
        pts = []
        steps = 220
        for s in range(steps + 1):
            a = (s / steps) * math.tau
            # Sample the noise around the ring to displace its radius.
            u = 0.5 + math.cos(a) * (0.16 + t * 0.30)
            v = 0.5 + math.sin(a) * (0.16 + t * 0.30)
            wobble = (noise(u, v) - 0.5) * (140 + 260 * t)
            r = base + wobble
            pts.append(f"{cx + math.cos(a) * r:.1f},{cy + math.sin(a) * r:.1f}")
        # Every fifth contour is an index line: heavier, like a printed map.
        index = i % 5 == 0
        opacity = 0.30 if index else 0.13
        width = 1.15 if index else 0.6
        paths.append(
            f'<polygon points="{" ".join(pts)}" fill="none" '
            f'stroke="rgb(56,189,248)" stroke-opacity="{opacity}" '
            f'stroke-width="{width}"/>'
        )

    return _svg(
        'xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid slice"',
        "".join(paths),
    )


def coord_grid() -> str:
    """Coordinate graticule with tick labels - the survey-map reference."""
    rng = random.Random(SEED + 7)
    parts = []
    step = 80
    for x in range(0, W + step, step):
        major = (x // step) % 3 == 0
        parts.append(
            f'<line x1="{x}" y1="0" x2="{x}" y2="{H}" '
            f'stroke="rgb(125,211,252)" stroke-opacity="{0.13 if major else 0.06}" '
            f'stroke-width="1"/>'
        )
    for y in range(0, H + step, step):
        major = (y // step) % 3 == 0
        parts.append(
            f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" '
            f'stroke="rgb(125,211,252)" stroke-opacity="{0.13 if major else 0.06}" '
            f'stroke-width="1"/>'
        )
    # Tick crosses along the frame, plus a few coordinate labels.
    for i in range(0, 11):
        x = i * W / 10
        parts.append(
            f'<path d="M{x:.0f} 0 v14 M{x:.0f} 0 v-14" '
            f'stroke="rgb(125,211,252)" stroke-opacity="0.22" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x + 6:.0f}" y="22" fill="rgb(125,211,252)" '
            f'fill-opacity="0.30" font-family="Consolas,monospace" '
            f'font-size="12">{"%03d" % (i * 10)}</text>'
        )
    for i in range(1, 6):
        y = i * H / 6
        parts.append(
            f'<text x="8" y="{y - 6:.0f}" fill="rgb(125,211,252)" '
            f'fill-opacity="0.22" font-family="Consolas,monospace" '
            f'font-size="12">{"%02d" % (48 + i)}</text>'
        )
    return _svg(
        f'xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        'preserveAspectRatio="xMidYMid slice"',
        "".join(parts),
    )


def sector_arcs() -> str:
    """Dashed range arcs and bearing spokes - the tactical overlay."""
    rng = random.Random(SEED + 13)
    parts = []
    cx, cy = W * 0.5, H * 0.5
    for i in range(1, 9):
        r = i * 105
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
            f'stroke="rgb(34,211,238)" stroke-opacity="0.10" stroke-width="1" '
            'stroke-dasharray="14 10"/>'
        )
    for i in range(24):
        a = i * math.tau / 24
        x2, y2 = cx + math.cos(a) * 950, cy + math.sin(a) * 950
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{x2:.0f}" y2="{y2:.0f}" '
            f'stroke="rgb(34,211,238)" stroke-opacity="{0.09 if i % 2 else 0.05}" '
            'stroke-width="1"/>'
        )
    # A few plotted contacts with leader lines, in the manner of a plot overlay.
    for _ in range(9):
        a = rng.random() * math.tau
        r = 120 + rng.random() * 700
        x, y = cx + math.cos(a) * r, cy + math.sin(a) * r
        amber = rng.random() < 0.3
        col = "251,191,36" if amber else "125,211,252"
        parts.append(
            f'<g stroke="rgb({col})" stroke-opacity="0.32" fill="none" '
            'stroke-width="1">'
            f'<circle cx="{x:.0f}" cy="{y:.0f}" r="9"/>'
            f'<line x1="{x - 17:.0f}" y1="{y:.0f}" x2="{x + 17:.0f}" y2="{y:.0f}"/>'
            f'<line x1="{x:.0f}" y1="{y - 17:.0f}" x2="{x:.0f}" y2="{y + 17:.0f}"/>'
            f'<path d="M{x + 12:.0f} {y + 12:.0f} L{x + 46:.0f} {y + 46:.0f}"/>'
            "</g>"
        )
    return _svg(
        f'xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        'preserveAspectRatio="xMidYMid slice"',
        "".join(parts),
    )


def schematic() -> str:
    """Technical callouts: dimension arrows, stencils, corner brackets."""
    rng = random.Random(SEED + 29)
    parts = []
    # Corner brackets, the framing mark used on drawings and map sheets.
    for x, y, dx, dy in ((0, 0, 1, 1), (W, 0, -1, 1), (0, H, 1, -1), (W, H, -1, -1)):
        parts.append(
            f'<path d="M{x + dx * 54} {y} L{x} {y} L{x} {y + dy * 54}" '
            'fill="none" stroke="rgb(34,211,238)" stroke-opacity="0.22" '
            'stroke-width="1.5"/>'
        )
    # Dimension lines with end ticks.
    for _ in range(5):
        y = 120 + rng.random() * (H - 240)
        x1 = 80 + rng.random() * 400
        x2 = x1 + 180 + rng.random() * 460
        parts.append(
            f'<g stroke="rgb(125,211,252)" stroke-opacity="0.16" '
            'stroke-width="1" fill="none">'
            f'<line x1="{x1:.0f}" y1="{y:.0f}" x2="{x2:.0f}" y2="{y:.0f}"/>'
            f'<line x1="{x1:.0f}" y1="{y - 6:.0f}" x2="{x1:.0f}" y2="{y + 6:.0f}"/>'
            f'<line x1="{x2:.0f}" y1="{y - 6:.0f}" x2="{x2:.0f}" y2="{y + 6:.0f}"/>'
            "</g>"
        )
    # Stencilled labels, as on an operational chart.
    for label, x, y in (
        ("SECTOR A", 130, 240),
        ("GRID 44N", W - 250, 190),
        ("CONTOUR INTERVAL 20m", 140, H - 150),
        ("PLOTTED", W - 210, H - 210),
    ):
        parts.append(
            f'<text x="{x}" y="{y}" fill="rgb(125,211,252)" fill-opacity="0.20" '
            'font-family="Consolas,monospace" font-size="17" '
            'letter-spacing="3">' + label + "</text>"
        )
    return _svg(
        f'xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        'preserveAspectRatio="xMidYMid slice"',
        "".join(parts),
    )


def _svg(attrs: str, body: str) -> str:
    return f"<svg {attrs}>{body}</svg>"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    layers = {
        "terrain-contours.svg": terrain_contours(),
        "coord-grid.svg": coord_grid(),
        "sector-arcs.svg": sector_arcs(),
        "schematic.svg": schematic(),
    }
    for name, content in layers.items():
        path = OUT_DIR / name
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.name}  {path.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
