"""The console is a contract, not a screenshot.

Every element id the front end reaches for is asserted here, and every view is
asserted to exist, so a markup change that silently orphans a handler fails the
build rather than a user request. The previous suite only checked that the
page and its two assets returned 200, which passed happily while the JS was
still wired to ids that no longer existed.
"""

from __future__ import annotations

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
    """Four generated SVG layers, all repo-local.

    The generator exists because raster defence photography could not be
    sourced, and a remote CDN asset would be a single point of failure during a
    recorded demo. SVGs cost kilobytes and never 404.
    """
    css_path = (STATIC / "styles.css").read_text(encoding="utf-8")
    # filename stem -> the class that carries it. The stems are hyphenated but
    # the classes are not, so they cannot be derived mechanically.
    layers = {
        "terrain-contours": "map-contours",
        "coord-grid": "map-grid",
        "sector-arcs": "map-sectors",
        "schematic": "map-schematic",
    }
    for stem, cls in layers.items():
        assert cls in page, f"the {stem} layer is missing"
        assert f"/static/media/{stem}.svg" in css_path, f"{stem} is not wired to CSS"

    assert "http://" not in _media_block(page), "a backdrop layer points at a CDN"

    # The artwork must actually exist on disk, or the layers render empty.
    media = STATIC / "media"
    for stem in layers:
        path = media / f"{stem}.svg"
        assert path.is_file(), f"{path.name} is referenced but not generated"
        text = path.read_text(encoding="utf-8")
        assert text.lstrip().startswith("<svg"), f"{path.name} is not valid SVG"
        assert path.stat().st_size < 400_000, f"{path.name} is unexpectedly large"
    # filename stem -> the class that carries it. The stems are hyphenated but
    # the classes are not, so they cannot be derived mechanically.
    layers = {
        "terrain-contours": "map-contours",
        "coord-grid": "map-grid",
        "sector-arcs": "map-sectors",
        "schematic": "map-schematic",
    }
    for stem, cls in layers.items():
        assert cls in page, f"the {stem} layer is missing"
        assert f"/static/media/{stem}.svg" in css_path, f"{stem} is not wired to CSS"

    assert "http://" not in _media_block(page), "a backdrop layer points at a CDN"

    # The artwork must actually exist on disk, or the layers render empty.
    media = STATIC / "media"
    for stem in layers:
        path = media / f"{stem}.svg"
        assert path.is_file(), f"{path.name} is referenced but not generated"
        text = path.read_text(encoding="utf-8")
        assert text.lstrip().startswith("<svg"), f"{path.name} is not valid SVG"
        assert path.stat().st_size < 400_000, f"{path.name} is unexpectedly large"


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


def test_the_backdrop_cannot_compete_with_the_content():
    """The layers were initially strong enough to show stencilled text through
    the panels, which fought the copy. Opacity is a legibility budget, not a
    taste knob.
    """
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    for layer in ("contours", "grid", "sectors", "schematic"):
        rule = css.split(f".map-{layer} {{")[1].split("}")[0]
        # CSS writes these without a leading zero, so `.15` means 0.15.
        value = float("0." + re.search(r"opacity:\s*\.(\d+)", rule).group(1))
        assert value <= 0.20, (
            f"the {layer} layer sits at {value}, too strong to read behind panels"
        )


def test_panels_carry_machined_corner_brackets(page):
    """A gradient rim lights the whole edge; brackets mark the corners.

    The corners need to sit *on* the rounded edge, which no border value can
    do, so they are a pseudo-element on an injected carrier span.
    """
    assert 'class="panel panel-hero ticked bracket"' in page
    assert page.count("panel bracket") >= 8, "too few panels are bracketed"

    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert '".bracket-corners"' in script, "the corner carrier is never injected"
    assert 'aria-hidden", "true"' in script, "decorative corners must be hidden"

    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    # The two rules are written as a shared selector list, so both cuts live in
    # the block that follows the shared opener.
    shared = css.split(".panel.bracket > .bracket-corners::before,")[1]
    assert "border-right: 0" in shared, "the top-left bracket is not cut"
    assert "border-bottom-right-radius" in shared, "the brackets are not rounded"
    assert "border-top-left-radius" in shared, "the brackets are not rounded"
    assert "rimSweep" in css, "the moving rim highlight is missing"
    assert "@keyframes rimSweep" in css


def test_corner_rim_motion_and_backdrop_both_stop_under_reduced_motion():
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    marker = "@media (prefers-reduced-motion: reduce)"
    blocks = "\n".join(css.split(marker)[1:])
    assert ".map-layer" in blocks and "animation: none" in blocks, (
        "the drifting backdrop still animates under reduced motion"
    )
    assert "bracket-corners" in blocks and "animation: none" in blocks, (
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