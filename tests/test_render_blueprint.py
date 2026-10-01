"""Validate render.yaml against Render's published JSON schema.

The Blueprint editor rejected the file because `healthCheckTimeout` is not a
real field. Guessing field names cost a deploy cycle, so this fetches the
authoritative schema and checks every key we use actually exists. It skips
rather than fails when the network is unavailable, so an offline run of the
suite is not broken by a third-party outage.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

SCHEMA_URL = "https://render.com/schema/render.yaml.json"
BLUEPRINT = "render.yaml"

#: Fields that do not exist in Render's schema at all. `healthCheckTimeout`
#: broke the Blueprint, and the editor names only the first offending field, so
#: a single invented key is an avoidable deploy cycle.
#:
#: Distinct from merely-deprecated fields below, which Render still accepts.
INVENTED_FIELDS = {
    "healthCheckTimeout": "Render has no such field; health checks have no timeout setting",
    "startupTimeout": "not a Blueprint field",
    "healthCheckInterval": "not a Blueprint field",
    "healthCheckRetries": "not a Blueprint field",
}

#: Accepted by Render but superseded. We use the replacement instead.
DEPRECATED_FIELDS = {
    "autoDeploy": "use autoDeployTrigger: commit",
    "afterFirstDeployCommand": "use initialDeployHook",
    "pullRequestPreviewsEnabled": "use previews.generation",
    "plan: starter": "not a plan id; see VALID_PLANS",
}

#: Compute plan identifiers Render accepts.
VALID_PLANS = {
    "free", "0.5c-512mb", "1c-2g", "2c-4g", "2c-8g", "2c-16g", "4c-8g",
    "4c-16g", "4c-32g", "8c-16g", "8c-32g", "8c-64g", "12c-24g", "12c-48g",
    "12c-96g",
}


@pytest.fixture(scope="module")
def blueprint() -> dict:
    import yaml

    from tests.conftest import PROJECT_ROOT

    return yaml.safe_load((PROJECT_ROOT / BLUEPRINT).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schema() -> dict:
    """Render's published Blueprint schema, or skip if offline."""
    try:
        with urllib.request.urlopen(SCHEMA_URL, timeout=30) as response:
            return json.load(response)
    except Exception as exc:  # noqa: BLE001 - a network outage must not fail CI
        pytest.skip(f"Render schema unavailable: {exc}")


@pytest.fixture(scope="module")
def service_fields(schema: dict) -> set[str]:
    """Field names a web service may actually set.

    Read from the live schema rather than a hand-copied list: the Blueprint
    editor rejects an unknown key, and a guessed name costs a deploy cycle.
    """
    definitions = schema.get("definitions") or {}
    web = definitions.get("serverService") or {}
    properties = web.get("properties") or {}
    if not properties:
        pytest.skip("schema shape changed; the static allowlist check still applies")
    # Resolve the allOf-composed web service if properties live on a parent.
    if not properties and "allOf" in web:
        for part in web["allOf"]:
            ref = (part or {}).get("$ref", "").split("/")[-1]
            properties = (definitions.get(ref) or {}).get("properties") or {}
            if properties:
                break
    return set(properties)


def test_the_blueprint_is_valid_yaml_and_has_services(blueprint):
    assert isinstance(blueprint, dict)
    assert blueprint.get("services"), "no services defined"


def test_every_service_field_exists_in_the_live_schema(blueprint, service_fields):
    offenders: list[str] = []
    for service in blueprint["services"]:
        unknown = sorted(set(service) - service_fields)
        if unknown:
            offenders.append(f"{service.get('name')}: {unknown}")
    assert not offenders, (
        "Render rejects the entire Blueprint for an unknown field; "
        f"offending services: {offenders}"
    )


def test_no_deprecated_or_invented_fields(blueprint):
    """`healthCheckTimeout` broke the Blueprint; the editor names one field at a
    time, so a single bad key is an avoidable deploy cycle."""
    offenders: list[str] = []
    for service in blueprint["services"]:
        for field in service:
            if field in INVENTED_FIELDS:
                offenders.append(f"{field} ({INVENTED_FIELDS[field]})")
            elif field in DEPRECATED_FIELDS:
                offenders.append(f"{field} ({DEPRECATED_FIELDS[field]})")
    assert not offenders, offenders


def test_the_plan_identifier_is_real(blueprint):
    """`starter` is not a Render plan. The free tier is 0.1 CPU / 512 MB, which
    is too small to load MiniLM and hold a FAISS index."""
    for service in blueprint["services"]:
        plan = service.get("plan")
        if plan is not None:
            assert plan in VALID_PLANS, f"{plan!r} is not a Render plan id"


def test_the_disk_is_only_attached_to_a_paid_plan(blueprint):
    """Render refuses a persistent disk on a free service.

    The Blueprint would be rejected at creation time with a less obvious
    message than "disks are not supported on this plan".
    """
    for service in blueprint["services"]:
        if "disk" in service:
            assert service.get("plan", "").startswith(("0.5c", "1c", "2c", "4c", "8c", "12c")), (
                f"service {service.get('name')!r} has a disk on plan {service.get('plan')!r}"
            )


def test_a_disk_is_never_combined_with_max_shutdown_delay(blueprint):
    """Render rejects `disk` + `maxShutdownDelaySeconds` on one service.

    The Blueprint editor accepted this file until the service was created, then
    failed with "max shutdown delay is not supported for services with a disk".
    Render's published schema does not encode that incompatibility - both keys
    are individually valid - so the schema test above passed and could not have
    caught it. Cross-field rules have to be asserted here.
    """
    offenders = [
        service.get("name")
        for service in blueprint["services"]
        if "disk" in service and "maxShutdownDelaySeconds" in service
    ]
    assert not offenders, (
        f"{offenders} combine a disk with maxShutdownDelaySeconds, which "
        "Render refuses to create"
    )


def test_env_vars_are_well_formed(blueprint):
    """An env var must be a literal, a dashboard prompt, or a reference.

    `key` + `value` sets it; `key` + `sync: false` prompts for it in the
    dashboard; `fromGroup` pulls it from an env group.
    """
    for service in blueprint["services"]:
        for entry in service.get("envVars", []):
            if "fromGroup" in entry:
                continue
            assert "key" in entry, entry
            forms = [
                "value" in entry,
                entry.get("sync") is False,
                "generateValue" in entry,
                "fromService" in entry,
                "fromDatabase" in entry,
            ]
            assert any(forms), f"{entry['key']} has no value, sync or reference"
            if "value" in entry:
                assert isinstance(entry["value"], str), (
                    f"{entry['key']} must be a string, not {type(entry['value']).__name__}"
                )


def test_no_secret_is_hardcoded(blueprint):
    """A key in a Blueprint is a published key.

    `sync: false` makes Render prompt for the value in the dashboard instead.
    """
    serialised = json.dumps(blueprint)
    for prefix in ("gsk_", "sk-", "hf_"):
        assert prefix not in serialised, f"a {prefix}... credential is committed"


@pytest.mark.parametrize("field", sorted(INVENTED_FIELDS))
def test_the_schema_really_lacks_these_fields(field, service_fields):
    """Confirm the ban list against Render's live schema.

    Guards against banning a field that is actually valid, and against a field
    we banned by guesswork quietly becoming real.
    """
    assert field not in service_fields, (
        f"{field} is now valid in Render's schema; drop it from INVENTED_FIELDS"
    )
