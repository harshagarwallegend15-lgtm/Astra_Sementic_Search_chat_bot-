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