"""The console is a contract, not a screenshot.

Every element id the front end reaches for is asserted here, and every view is
asserted to exist, so a markup change that silently orphans a handler fails the
build rather than a user request. The previous suite only checked that the
page and its two assets returned 200, which passed happily while the JS was
still wired to ids that no longer existed.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="fastapi is required for the console tests")

from fastapi.testclient import TestClient  # noqa: E402

STATIC = Path(__file__).resolve().parents[1] / "static"


@pytest.fixture(scope="module")
def client():
    """A client just rich enough to assert the shell renders.

    No index or model is needed: these assertions read the served markup and
    the asset files, not the pipeline.
    """
    from fastapi.testclient import TestClient as _TestClient

    import api

    return _TestClient(api.app)


@pytest.fixture(scope="module")
def page(client):
    return client.get("/").text


def _ids(html: str) -> set[str]:
    return set(re.findall(r'id="([^"]+)"', html))


def _views(html: str) -> set[str]:
    """Section ids only.

    Matching every `view-` prefixed id also picks up the topbar's heading
    (`view-title`, `view-sub`), which are not destinations.
    """
    return set(re.findall(r'<section[^>]*\bid="view-([a-z]+)"', html))


def test_both_assets_are_served(client):
    for asset in ("/static/styles.css", "/static/app.js"):
        assert client.get(asset).status_code == 200, asset


def test_the_shell_is_never_cached_and_points_at_current_assets(client):
    """A cached shell is indistinguishable from a failed deploy.

    The page used to link bare `/static/styles.css` and `/static/app.js`, so a
    browser could keep rendering an old background and old frames after the
    server had already changed them.
    """
    response = client.get("/")
    assert response.status_code == 200
    cache = response.headers.get("cache-control", "")
    assert "no-store" in cache, "the console shell can be cached by the browser"

    css_match = re.search(r"/static/styles\.css\?v=([0-9a-f]{8,})", response.text)
    js_match = re.search(r"/static/app\.js\?v=([0-9a-f]{8,})", response.text)
    assert css_match, "the stylesheet URL is not content-versioned"
    assert js_match, "the script URL is not content-versioned"
    assert "__CSS_VERSION__" not in response.text
    assert "__JS_VERSION__" not in response.text

    import api

    assert css_match.group(1) == api._asset_version("styles.css")
    assert js_match.group(1) == api._asset_version("app.js")

    # Query strings must not break the static routes.
    assert client.get(f"/static/styles.css?v={css_match.group(1)}").status_code == 200
    assert client.get(f"/static/app.js?v={js_match.group(1)}").status_code == 200


def test_every_id_the_script_uses_exists(page):
    """`$(\"id\")` and `getElementById` are the only wiring mechanism used.

    A stale id here means a dead control: the handler silently throws on
    click and the panel never updates.
    """
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    referenced = set(re.findall(r'\$\("([^"]+)"\)', script))
    referenced |= set(re.findall(r'getElementById\("([^"]+)"\)', script))
    # Ids the script creates itself (the filter escape hatch, for one) are
    # legitimately absent from the static markup.
    created = set(re.findall(r'id="([a-z-]+)"', script))
    missing = sorted(referenced - _ids(page) - created)
    assert not missing, f"app.js references ids absent from index.html: {missing}"


def test_every_view_has_a_section_and_a_nav_entry(page):
    views = _views(page)
    assert views, "no views found"
    for view in views:
        assert f'data-view="{view}"' in page, f"view-{view} has no nav entry"


def test_the_navigation_stays_deliberate(page):
    """Three destinations, each of which an operator actually works in.

    Overview was added later as the landing surface. The previous build carried
    five tabs: Corpus Grid, Briefings and Configuration were panels describing
    data that lives in Documents, or a raw settings dump whose useful values
    are already shown as readouts. Those three stay gone.
    """
    views = _views(page)
    assert views == {"home", "query", "documents"}
    for dropped in ("corpus", "analytics", "config"):
        assert f"view-{dropped}" not in page, f"the {dropped} tab is back"


def test_overview_is_the_landing_surface(page):
    """The first thing a visitor sees must be Overview, not the query box.

    A landing page whose header still says "Query" while the hero is on screen
    reads as a half-finished build.
    """
    assert 'id="view-home" class="view is-active"' in page or (
        'class="view is-active" id="view-home"' in page
    ), "the home view is not the active one on load"
    assert '<h1 id="view-title">Overview</h1>' in page, (
        "the header still labels the landing surface as another view"
    )
    first_nav = page.index('class="nav-item is-active"')
    assert 'data-view="home"' in page[first_nav : first_nav + 120], (
        "the active nav entry is not Overview"
    )
    # The hero must offer a way into the actual product.
    assert 'id="hero-open-query"' in page


def test_the_hero_motion_respects_reduced_motion(page):
    """A full-bleed looping background is a vestibular trigger.

    The hero is the first thing anyone sees, so it is the one place that has to
    stop completely rather than merely slow down.
    """
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    # Split on the at-rule, not the phrase: a comment at the top of the file
    # also contains "prefers-reduced-motion", and matching that yields the whole
    # stylesheet instead of the block.
    marker = "@media (prefers-reduced-motion: reduce)"
    assert marker in css, "no reduced-motion block at all"
    blocks = css.split(marker)
    hero_rules = "\n".join(blocks[1:])
    assert ".hero-media video" in hero_rules, "the hero video is not disabled"
    assert ".hero-media canvas" in hero_rules, "the hero canvas is not quieted"
    assert 'id="hero-canvas"' in page


def test_the_hero_video_is_local_and_optional(page):
    """No CDN: an unreachable third-party asset would blank the hero mid-demo.

    The element points at a repo-local path and starts transparent, and only
    becomes visible once the browser reports that it is actually playing. A
    missing file therefore leaves the canvas carrying the motion instead.
    """
    assert 'src="/static/media/hero.mp4"' in page, "the video source is not local"
    assert "http://" not in page.split('id="hero-video"')[1][:400], (
        "the hero video points at a remote host"
    )
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    hero = script.split("function startHero")[1].split("\nfunction ")[0]
    # Reveal only on a real `playing` event, never optimistically.
    assert '"playing"' in hero, "the video is revealed without confirming playback"
    assert 'is-live' in hero
    # It must not animate while the hero is off screen or the tab is hidden.
    assert "IntersectionObserver" in hero
    assert "visibilitychange" in hero


def test_destructive_actions_are_not_in_the_topbar(page):
    """Clear and rebuild belong to maintenance, not beside the primary action.

    A one-click path from the toolbar to wiping the corpus is exactly what the
    confirmation token was added to prevent, so the button placement matters
    as much as the guard.
    """
    topbar = page.split('<header class="topbar">')[1].split("</header>")[0]
    for button in ("btn-clear", "btn-rebuild"):
        assert f'id="{button}"' not in topbar, f"{button} is back in the topbar"
    assert 'class="maint"' in page


def test_the_query_console_is_the_dashboard_hero(page):
    deck = page.split('id="view-query"')[1].split("</section>")[0]
    # The console must come before the telemetry rail in the DOM so it leads
    # for both screen readers and keyboard users.
    assert deck.index('id="ask-form"') < deck.index('id="kpi-row"')
    assert "deck-rail" in deck


def test_no_placeholder_dashes_survive_in_the_shipped_markup(page):
    """An empty <span> renders as a bare dash, which reads as broken chrome."""
    # The em-dash placeholders are intentional only before JS fills them.
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    filled = set(re.findall(r'\$\("(rd-[a-z]+|kpi-[a-z-]+)"\)\.textContent', script))
    # Counted figures go through setNum so they can animate.
    filled |= set(re.findall(r'setNum\("(kpi-[a-z-]+)"', script))
    for match in re.findall(r'id="(rd-[a-z]+|kpi-[a-z]+)"[^>]*>&mdash;', page):
        assert match in filled, f"{match} is never filled by app.js"


def test_the_reduced_motion_block_covers_the_ambient_layers():
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    marker = "@media (prefers-reduced-motion: reduce)"
    assert marker in css
    block = "\n".join(css.split(marker)[1:])
    for layer in (".orb", ".grid-veil", ".sweep"):
        assert layer in block, f"{layer} still animates under reduced motion"


def test_the_upload_input_cannot_swallow_drag_events(page):
    """Regression: the intake drop target was dead.

    The file input was styled as a full-size invisible overlay
    (`position:absolute; inset:0; opacity:0`) inside the dropzone. It therefore
    sat above the drop target, and the browser routed dragover/drop to the
    input, so the dropzone's own handlers never fired and dragging a PDF in did
    nothing. The input is now visually hidden and inert, and an explicit Browse
    button is the pointer target.
    """
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    rule = css.split(".dropzone input[type=\"file\"]")[1].split("}")[0]
    for banned in ("inset:", "width: 100%", "height: 100%", "opacity: 0"):
        assert banned not in rule, (
            f"`{banned}` makes the input cover the dropzone and swallow drops"
        )

    # The input must not be a label wrapping itself, and there must be a
    # real, clickable way to open the picker.
    assert "<label class=\"dropzone\"" not in page, (
        "a label wrapping the input double-activates the picker"
    )
    assert 'type="button" class="btn btn-ghost" id="upload-browse"' in page

    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert '$("upload-browse").onclick' in script, "the Browse button is not wired"
    dropzone = script.split("function bindDropzone")[1].split("\n}")[0]
    assert '"drop"' in dropzone, "no drop handler on the dropzone"
    assert "DataTransfer" in dropzone


def test_intake_reports_progress_and_failure_on_screen(page):
    """A silent upload is indistinguishable from a broken one.

    The status line and the toast are the only feedback the operator gets, so
    both the success and the failure paths must write to them.
    """
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    # The handler body runs to the next top-level `$("...")` binding, which is
    # the only reliable end marker: any `};` inside lands mid-function.
    start = script.index('$("upload-form").onsubmit')
    end = script.index("\n  $(", script.index("bindDropzone();", start))
    submit = script[start:end]
    assert "intakeStatus(" in submit, "no progress text on upload"
    # intakeStatus() maps a kind to the `is-<kind>` class, so the call sites
    # pass the bare kind.
    assert 'intakeStatus(err.message, "err")' in submit, (
        "upload failures are not surfaced in the status line"
    )
    assert '"ok")' in submit, "upload success is not surfaced in the status line"
    assert '"busy"' in submit, "no in-progress state while ingesting"
    assert 'id="upload-note"' in page

    # The kinds the handler passes must all have a style.
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    for kind in ("ok", "err", "busy"):
        assert f".intake-status.is-{kind}" in css, f"no style for is-{kind}"


def test_a_selected_file_is_named_on_screen(page):
    """Regression: the operator never saw which file they had picked.

    The selection line reported only a count and a size - "1 file · 1.2 MB" -
    so a file input that silently failed to open, or a drop that landed on the
    wrong document, looked identical to success. The batch is now listed by
    name.
    """
    assert 'id="upload-list"' in page, "no element lists the selected files"

    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function renderFileList" in script
    renderer = script.split("function renderFileList")[1].split("\n}")[0]
    assert "f.name" in renderer, "the list never reads the file name"
    assert "shortName" in renderer, "the name is not shortened for display"
    assert "file-name" in renderer, "the name has no dedicated element"

    # It must be called on selection, and cleared afterwards.
    assert script.count("renderFileList(") >= 3, (
        "renderFileList must be wired to the change handler, to the empty "
        "case, and to the post-ingest reset"
    )
    # A 64-character name would wrap the row, so long names are truncated.
    assert "base.length > 64" in script


def test_the_hero_motion_is_defence_coded_not_generic():
    """The hero should read as a tactical scope, not a particle field.

    It began as drifting nodes and pulses, which could have belonged to any
    product page. The defence language is the point: a rotating beam, range
    rings, bearing spokes and contacts that brighten as they are illuminated.
    """
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    hero = script.split("function startHero")[1].split("\nfunction showView")[0]

    for element in (
        "createConicGradient",  # the rotating beam
        "createRadialGradient",  # a contact's return halo
        "CONTACTS",
    ):
        assert element in hero, f"{element} is missing from the hero animation"

    # Range rings and bearing spokes.
    assert hero.count("ctx.arc") >= 2, "no range rings drawn"
    assert "ctx.moveTo(cx, cy)" in hero, "no bearing spokes drawn"

    # Contacts must decay after the beam passes, or they just glow forever.
    assert "k.level = Math.min(1" in hero, "contacts never illuminate"
    assert "k.level = Math.max(0" in hero, "contacts never decay"

    # An amber contact type distinguishes an unknown return from a friendly.
    assert "k.kind === 3" in hero and "251,191,36" in hero, (
        "no unknown-contact colouring"
    )


def test_the_glass_system_is_built_from_cooperating_parts(page):
    """A single translucent fill does not read as glass.

    Frosted surfaces need a translucent fill, a blurred backdrop, a rim that is
    lighter where light catches it, and an inset sheen along the top edge. The
    panels previously had only a flat fill plus a 1px border, which is why they
    looked like flat translucent boxes.
    """
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    for token in (
        "--glass-fill",
        "--glass-blur",
        "--glass-rim",
        "--glass-sheen",
        "--glass-drop",
    ):
        assert token in css, f"the {token} token is missing"
        assert f"{token}:" in css, f"{token} is referenced but never defined"

    rule = css.split(".panel, .tile, .cite, .brief, .toast, .chip, .code,")[1]
    rule = rule.split("}")[0]
    assert "var(--glass-fill)" in rule, "panels do not use the glass fill"
    assert "var(--glass-blur)" in rule, "panels are not frosted"
    assert "var(--glass-sheen)" in rule, "panels have no inset sheen"
    assert "var(--glass-drop)" in rule, "panels cast no depth shadow"

    # The rim needs a mask trick, since a gradient cannot be a border.
    assert "-webkit-mask" in css and "content-box" in css, (
        "the gradient rim is not actually masked to the border"
    )


def test_the_hero_scop_does_not_flatten_the_animation(page):
    """The scrim has to protect the copy without hiding the radar.

    A single uniform scrim at the opacity that makes the headline readable
    rendered the scope invisible, which defeated the point of drawing it.
    """
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    scrim = css.split(".hero-scrim {")[1].split("}")[0]
    stops = scrim.count("rgba(4,8,15")
    assert stops >= 4, "the scrim is not a gradient across the hero"
    # It must get materially lighter toward the right, where the scope is.
    opacities = [float(x) for x in re.findall(r"rgba\(4,8,15,\.(\d+)\)", scrim)]
    assert opacities[-1] < opacities[0] - 0.5, (
        "the scrim does not lighten over the scope, so it stays hidden"
    )


def test_the_hero_covers_all_four_stats_on_one_row():
    """auto-fit wrapped the stats to 3+1, stranding MODEL on its own row."""
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    # Split on the declaration block, not the first `}`: `minmax(0, 1fr)`
    # contains a brace, so a naive split truncates before grid-template-columns.
    stats = css.split(".hero-stats {")[1].split("grid-template-columns")[1]
    assert "repeat(4," in stats, "the hero stats are not four explicit tracks"
    assert "auto-fit" not in stats, "auto-fit still wraps the stats"


def test_the_backdrop_artwork_is_present_and_local(page):
    """Four generated SVG layers plus six defence photographs, all local.

    The generator covers the instrument chrome. The photographs are US
    government releases that scripts/fetch_backdrop_photos.py verifies and
    grades on download, so they ship in the repo rather than from a CDN,
    which would be a single point of failure during a recorded demo.
    """
    css_path = (STATIC / "styles.css").read_text(encoding="utf-8")
    # filename stem -> the class that carries it. The stems are hyphenated but
    # the classes are not, so they cannot be derived mechanically.
    layers = {
        "terrain-contours": "map-contours",
        "coord-grid": "map-grid",
        "sector-arcs": "map-sectors",
        "schematic": "map-schematic",
        "scene-abrams": "map-photo p0",
        "scene-armour-column": "map-photo p1",
        "scene-briefing": "map-photo p2",
        "scene-radar": "map-photo p3",
        "scene-artillery": "map-photo p4",
        "scene-observation": "map-photo p5",
    }
    for stem, cls in layers.items():
        assert cls in page, f"the {stem} layer is missing"
        assert f"/static/media/{stem}." in css_path, f"{stem} is not wired to CSS"

    assert "http://" not in _media_block(page), "a backdrop layer points at a CDN"

    # The artwork must actually exist on disk, or the layers render empty.
    media = STATIC / "media"
    for stem in layers:
        matches = [media / f"{stem}.svg", media / f"{stem}.jpg"]
        path = next((p for p in matches if p.is_file()), None)
        assert path is not None, f"{stem} is referenced but not on disk"
        if path.suffix == ".svg":
            text = path.read_text(encoding="utf-8")
            assert text.lstrip().startswith("<svg"), f"{path.name} is not valid SVG"
            assert path.stat().st_size < 400_000, f"{path.name} is unexpectedly large"
        else:
            # A backdrop photograph that costs a megabyte is a backdrop that
            # arrives after the demo has moved on.
            assert path.stat().st_size < 500_000, f"{path.name} is too heavy"


def _media_block(page: str) -> str:
    start = page.find('<div class="aurora"')
    return page[start : page.find("</div>", start)]


def test_backdrop_layers_drift_at_different_speeds():
    """Differential drift is what produces parallax.

    Running every layer at one speed collapses the effect into a flat pan,
    which is the whole visual difference being bought here.
    """
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    durations = set(re.findall(r"animation:\s*drift-\w+\s+(\d+)s", css))
    assert len(durations) >= 3, f"only {len(durations)} drift speeds; no parallax"
    for layer in ("contours", "grid", "sectors", "schematic"):
        assert f"@keyframes drift-{layer}" in css, f"drift-{layer} has no keyframes"
    # Oversized so a translate never exposes an edge.
    layer_rule = css.split(".map-layer {")[1].split("}")[0]
    assert "inset: -15%" in layer_rule or "-15%" in layer_rule, (
        "the layers are not oversized, so drifting exposes the edge"
    )


def _keyframes(css: str, name: str) -> str:
    start = css.index(f"@keyframes {name}")
    depth = 0
    for index in range(css.index("{", start), len(css)):
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
            if depth == 0:
                return css[start : index + 1]
    raise AssertionError(f"@keyframes {name} is not closed")


def test_the_drift_moves_fast_enough_to_be_seen():
    """A drift nobody can perceive is the same as no drift.

    The first pass ran a few percent over two minutes, which works out to
    about one pixel per second on a laptop screen. The backdrop was reported
    as motionless for exactly that reason. This floors the on-screen speed so
    the same regression cannot be reintroduced by "slow it down a bit".
    """
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    # The layer element is 130% of the viewport, so a translate of A% of that
    # element covers A * 1.3% of a 1400px screen.
    screen = 1400
    for layer in ("contours", "grid", "sectors", "schematic"):
        duration = int(
            re.search(rf"animation:\s*drift-{layer}\s+(\d+)s", css).group(1)
        )
        points = [
            (float(x), float(y))
            for x, y in re.findall(
                r"translate3d\((-?[\d.]+)%,\s*(-?[\d.]+)%", _keyframes(css, f"drift-{layer}")
            )
        ]
        assert len(points) >= 2, f"drift-{layer} does not move"
        travel = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
        px_per_second = travel * 1.3 * screen / 100 / duration
        assert px_per_second >= 2.0, (
            f"drift-{layer} moves at {px_per_second:.1f}px/s and reads as static"
        )


def _frame_declarations(css: str, index: int) -> str:
    """The per-frame custom properties, matched as a top-level rule.

    Splitting on '.p0 {' would find '.map-photo.p0 {' first, which is the
    background-image rule and carries none of the variables.
    """
    match = re.search(rf"^\.p{index} \{{(.*?)\}}", css, re.S | re.M)
    assert match, f"frame {index} has no rule of its own"
    return match.group(1)


def test_the_photographs_dissolve_continuously_and_keep_pushing():
    """The home page must carry real moving imagery, not only line work.

    Six frames, one sixth of the loop each, dissolving with no hold: that is
    what keeps total opacity flat, so the sequence cannot flash black between
    frames or flare where two of them overlap. A separate transform animation
    is what keeps something moving when only part of the reel is at full
    strength.
    """
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    rule = css.split(".map-photo {")[1].split("}")[0]
    assert "dissolve 120s linear infinite" in rule, "the dissolve has no linear ramp"
    assert "push 120s linear infinite" in rule, "the frames are not always pushing"
    assert "calc(var(--i) * 20s)" in rule, (
        "the frames are not staggered by a sixth of the loop"
    )
    assert "background-size: cover" in rule, "the frame does not fill the viewport"
    assert "mask-image" in rule, "the frame edge would show as a hard rectangle"

    dissolve = _keyframes(css, "dissolve")
    # Rises over the first sixth, falls over the next: a 40s window against a
    # 20s stagger, so the outgoing and incoming frames complement each other
    # and total opacity never dips at the handover.
    assert "16.667%" in dissolve and "33.333%" in dissolve, (
        "the dissolve window is not twice the stagger, so it will dip or flare"
    )
    assert dissolve.count("opacity: 0") >= 2, "the dissolve has no fade-out"

    push = _keyframes(css, "push")
    scales = [float(s) for s in re.findall(r"scale\(([\d.]+)\)", push)]
    assert max(scales) - min(scales) >= 0.12, "the push-in is too small to see"

    # Every frame needs its own slot and its own drift direction, otherwise
    # the reel either runs in lockstep or pans as one block.
    seen = set()
    for index in range(6):
        declarations = _frame_declarations(css, index)
        assert f"--i: {index};" in declarations, f"frame {index} has no slot"
        assert "--peak:" in declarations, f"frame {index} has no peak opacity"
        assert "--ax:" in declarations, f"frame {index} has no drift direction"
        seen.add(re.search(r"--ax:\s*(-?[\d.]+)%", declarations).group(1))
    assert len(seen) > 1, "every frame drifts the same way"


def test_the_photographs_are_bright_enough_to_see():
    """An earlier pass graded the frames to near-black duotone and they read as
    texture rather than as photographs. The peak opacity has to stay in a band
    that is clearly visible while still sitting behind the copy.
    """
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    for index in range(6):
        declarations = _frame_declarations(css, index)
        peak = float("0." + re.search(r"--peak:\s*\.(\d+)", declarations).group(1))
        assert 0.40 <= peak <= 0.62, f"frame {index} peaks at {peak}"


def test_the_photographs_stop_under_reduced_motion():
    """animation: none leaves the frames at their 0% keyframe, which is opacity 0.

    Without an explicit override a reduced-motion visitor gets no photograph at
    all, and freezing all six would stack them on top of each other.
    """
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    marker = "@media (prefers-reduced-motion: reduce)"
    blocks = "\n".join(css.split(marker)[1:])
    assert ".map-photo { opacity: 0; }" in blocks, (
        "the dissolving frames would all stack when frozen"
    )
    assert ".map-photo.p0 { opacity: .34; }" in blocks, (
        "no still photograph is held for reduced motion"
    )


def test_the_backdrop_cannot_compete_with_the_content():
    """The background should be unmistakable, but not fight the copy.

    It was first tuned to nearly invisible; then the panels were made opaque
    enough to carry stronger layers. These bounds preserve both decisions.
    """
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    bounds = {
        "contours": (0.28, 0.36),
        "grid": (0.14, 0.20),
        "sectors": (0.22, 0.30),
        "schematic": (0.14, 0.20),
    }
    for layer, (minimum, maximum) in bounds.items():
        rule = css.split(f".map-{layer} {{")[1].split("}")[0]
        # CSS writes these without a leading zero, so `.34` means 0.34.
        value = float("0." + re.search(r"opacity:\s*\.(\d+)", rule).group(1))
        assert minimum <= value <= maximum, (
            f"the {layer} layer sits at {value}, outside {minimum}-{maximum}"
        )


def test_panels_carry_machined_corner_brackets(page):
    """A gradient rim lights the whole edge; brackets mark all four corners.

    Two corners left the other edges visually unresolved. The marks sit on the
    rounded edge, which no border value can do, so they are injected rather than
    hand-written.
    """
    assert 'class="panel panel-hero ticked bracket"' in page
    assert page.count("panel bracket") >= 8, "too few panels are bracketed"

    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert '"corner-tl", "corner-tr", "corner-bl", "corner-br"' in script, (
        "all four corners are not injected"
    )
    assert 'aria-hidden", "true"' in script, "decorative corners must be hidden"

    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    for corner in ("corner-tl", "corner-tr", "corner-bl", "corner-br"):
        assert f".{corner} {{" in css, f"the {corner} style is missing"
    for radius in (
        "border-top-left-radius",
        "border-top-right-radius",
        "border-bottom-left-radius",
        "border-bottom-right-radius",
    ):
        assert radius in css, "a corner bracket is not rounded"
    assert ".panel.bracket::after" in css, "the travelling rim highlight is missing"
    assert "@keyframes rimSweep" in css


def test_corner_rim_motion_and_backdrop_both_stop_under_reduced_motion():
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    marker = "@media (prefers-reduced-motion: reduce)"
    blocks = "\n".join(css.split(marker)[1:])
    assert ".map-layer" in blocks and "animation: none" in blocks, (
        "the drifting backdrop still animates under reduced motion"
    )
    assert ".panel.bracket::after" in blocks and "animation: none" in blocks, (
        "the moving panel rim still animates under reduced motion"
    )


def test_the_donut_centre_figure_is_readable_on_a_dark_panel():
    """Regression: the count was drawn #1a202c on a near-black card.

    The empty-state track was also near-white, which read as a solid disc
    rather than an unused ring. Both now come from CSS classes, so the colours
    live in one place with the rest of the palette.
    """
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    assert "#donut .d-total" in css and "var(--ink)" in css.split("#donut .d-total")[1][:80]
    assert "#donut .d-track" in css

    script = (STATIC / "app.js").read_text(encoding="utf-8")
    donut = script.split("function drawDonut")[1].split("\n}")[0]
    assert "d-total" in donut and "d-track" in donut
    # No hardcoded dark text on the panel colour.
    assert "#1a202c" not in donut and "#edf2f7" not in donut