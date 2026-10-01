"""Fetch and grade the defence photography that carries the home page.

The backdrop had no picture on it at all: only generated line work, which read
as instrument chrome rather than as imagery. These frames are US government
photographs released into the public domain, so they can ship inside the repo
with no attribution burden and no CDN to fail during a recorded demo.

Two rules the pipeline enforces rather than assumes:

* Nothing downloads unless its Commons licence is public domain or CC0. The
  licence is read from the API at fetch time, so a retitled or re-licensed
  file fails the build rather than shipping a rights problem.
* Each frame is graded toward the console palette - slightly desaturated,
  slightly cooled, darkened - so six different photographers still read as one
  set. They are colour images; the earlier duotone treatment erased exactly the
  thing worth showing.

Re-running is idempotent: same filenames, same result.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
MEDIA = PROJECT / "static" / "media"

# filename stem -> Commons file title. The stems are hyphenated so the
# filenames stay readable in the browser's network panel.
FILES = {
    "scene-abrams": (
        "File:Left front view M1A1 Abrams Main Battle Tank at U.S. Army Armor "
        "Center, USAARMC DA-SC-94-01698.jpg"
    ),
    "scene-armour-column": (
        "File:Abrams Tanks home to 3rd ACR Scouts, Mosul.jpg"
    ),
    "scene-briefing": (
        "File:American and Somali officers at daily progress briefing during "
        "multinational joint service exercise.jpg"
    ),
    "scene-radar": (
        "File:Flickr - Official U.S. Navy Imagery - A Sailor monitors the "
        "SPA-25G radar console from the combat information center..jpg"
    ),
    "scene-artillery": (
        "File:Battery A, 7th Battalion, 15th Artillery 8-inch howitzer fires "
        "at Firebase Exodus.jpg"
    ),
    "scene-observation": "File:Forward Observation Post (8421327).jpg",
}

WIDTH = 1500
API = "https://commons.wikimedia.org/w/api.php"
UA = "astra-intel-backdrop/1.0 (local asset pipeline)"
ALLOWED = ("public domain", "cc0")


def _get(url: str, timeout: int = 120) -> bytes:
    """Fetch with backoff: the Commons API rate-limits bursts of requests."""
    delay = 3
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            time.sleep(delay)
            delay *= 2
    raise SystemExit(f"unreachable: {url}")


def licence_of(title: str) -> str:
    params = {
        "action": "query", "format": "json", "titles": title,
        "prop": "imageinfo", "iiprop": "extmetadata",
    }
    data = json.loads(_get(API + "?" + urllib.parse.urlencode(params), timeout=60))
    for page in data.get("query", {}).get("pages", {}).values():
        meta = (page.get("imageinfo") or [{}])[0].get("extmetadata", {})
        return str(meta.get("LicenseShortName", {}).get("value", "unknown"))
    raise SystemExit(f"{title}: not found on Commons")


def fetch(title: str, width: int) -> tuple[bytes, str]:
    licence = licence_of(title)
    if not any(token in licence.lower() for token in ALLOWED):
        raise SystemExit(f"{title}: licence is {licence!r}, not shippable")
    url = (
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        + urllib.parse.quote(title.removeprefix("File:").replace(" ", "_"))
        + f"?width={width}"
    )
    return _get(url), licence


def grade(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    image = ImageEnhance.Color(image).enhance(0.9)
    image = ImageEnhance.Contrast(image).enhance(1.12)
    # A generous share of the palette's navy unifies six photographers into
    # one set and cools photographs shot under a desert sun.
    duotone = ImageOps.colorize(
        image.convert("L"), black=(5, 9, 18), white=(196, 220, 255)
    )
    image = Image.blend(image, duotone, 0.32)
    # Copy and instrument readouts sit on top of this. Underexposed frames
    # keep the console legible; bright ones were fogging the whole page.
    return ImageEnhance.Brightness(image).enhance(0.66)


def main() -> int:
    MEDIA.mkdir(parents=True, exist_ok=True)
    for stem, title in FILES.items():
        raw, licence = fetch(title, WIDTH)
        image = grade(Image.open(io.BytesIO(raw)))
        if image.width > WIDTH:
            height = round(image.height * WIDTH / image.width)
            image = image.resize((WIDTH, height), Image.LANCZOS)
        target = MEDIA / f"{stem}.jpg"
        image.save(target, "JPEG", quality=62, optimize=True, progressive=True)
        print(f"wrote {target.name} {image.size[0]}x{image.size[1]} "
              f"{target.stat().st_size // 1024}KB ({licence})")
        # The API is shared infrastructure; a burst of six downloads earns a
        # 429 quickly enough to break the run.
        time.sleep(1.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
