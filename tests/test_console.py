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
    """Two destinations, each of which an operator actually works in.

    The previous build carried Corpus Grid, Briefings and Configuration as
    separate tabs: the heatmap and the briefings were panels describing data
    that lives in Documents, and Configuration was a raw settings dump whose
    useful values are already shown as readouts.
    """
    views = _views(page)
    assert views == {"query", "documents"}
    for dropped in ("corpus", "analytics", "config"):
        assert f"view-{dropped}" not in page, f"the {dropped} tab is back"


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
    block = css.split("prefers-reduced-motion")[1]
    for layer in (".orb", ".grid-veil", ".sweep"):
        assert layer in block, f"{layer} still animates under reduced motion"


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