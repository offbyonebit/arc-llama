from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import asdict

import pytest
from click.testing import CliRunner
from helpers import AsyncContextClient

import arc_llama.mtp_tuning as mtp
from arc_llama.cli import cli
from arc_llama.config import Config, ModelConfig, load_config
from arc_llama.recipes import LaunchRecipe, recipe_to_dict
from arc_llama.server_caps import ServerCaps, _parse_help
from arc_llama.speculation import confidence_value, effective_confidence

CAPS = ServerCaps(
    probed=True,
    supports_mtp=True,
    supports_draft_confidence=True,
    supports_speculative=True,
    supports_draft_model=True,
)


def model(recipe=None):
    return ModelConfig(
        name="target",
        path="/models/target.gguf",
        port=18080,
        gpu_pci_slot="0000:03:00.0",
        recipe=recipe or {"spec_type": "draft-mtp"},
    )


def result(rate=10.0, wall=1000.0):
    return mtp.MTPMeasurement([rate] * 5, [wall] * 5, output_fingerprints=["same"] * 5)


@pytest.mark.parametrize("value", [True, False, -0.1, 1.1, float("nan"), float("inf"), "0.5"])
def test_confidence_rejects_invalid(value):
    with pytest.raises(ValueError):
        confidence_value(value)


def test_legacy_confidence_and_typed_override_roundtrip(tmp_path):
    flags = ["--reasoning", "auto", "--draft-p-min=0.2", "--spec-draft-p-min", "0.75"]
    assert effective_confidence({"extra_flags": flags}) == 0.75
    recipe = LaunchRecipe(spec_draft_p_min=0.5, extra_flags=flags)
    argv = recipe.to_argv()
    assert argv.count("--spec-draft-p-min") == 1
    assert "--draft-p-min=0.2" not in argv
    assert flags == recipe.extra_flags
    cfg = Config(models=[model(recipe_to_dict(recipe))])
    path = tmp_path / "config.toml"
    cfg.save(path)
    loaded = load_config(path).models[0].launch_recipe()
    assert loaded.spec_draft_p_min == 0.5
    assert loaded.to_argv() == argv


def test_legacy_missing_value_rejected():
    with pytest.raises(ValueError):
        effective_confidence({"extra_flags": ["--draft-p-min"]})


def test_capability_detection():
    caps = _parse_help("--spec-type none,draft-mtp --spec-draft-p-min P")
    assert caps.supports_mtp and caps.supports_draft_confidence
    assert not _parse_help("--spec-type draft-simple").supports_mtp


@pytest.mark.parametrize("sidecar", [None, "/models/mtp.gguf"])
def test_validate_embedded_and_sidecar(monkeypatch, sidecar):
    checked = []
    monkeypatch.setattr(mtp, "has_mtp_heads", lambda path: checked.append(path) or True)
    m = model({"spec_type": "draft-mtp", "spec_draft_model": sidecar})
    mtp.validate_mtp(m, CAPS)
    assert checked == [sidecar or m.path]


def test_validate_rejects_unprobed_and_overrides(monkeypatch):
    monkeypatch.setattr(mtp, "has_mtp_heads", lambda path: False)
    with pytest.raises(ValueError, match="probed"):
        mtp.validate_mtp(model(), ServerCaps())
    with pytest.raises(ValueError, match="No MTP"):
        mtp.validate_mtp(model(), CAPS)
    m = model({"extra_flags": ["--spec-type=draft-mtp"]})
    with pytest.raises(ValueError, match="extra_flags"):
        mtp.validate_mtp(m, CAPS)


def test_telemetry_weighting_and_missing(monkeypatch):
    counts = [(2, 1), (10, 9), None, (0, 0), None]
    calls = []

    async def complete(*args, **kwargs):
        calls.append(kwargs)
        timing = {"predicted_n": 16, "predicted_per_second": 20, "prompt_ms": 25}
        if len(calls) > 1:
            pair = counts[len(calls) - 2]
            if pair is not None:
                timing.update(draft_n=pair[0], draft_n_accepted=pair[1])
        return 1.0, {"timings": timing, "choices": [{"text": "same completion"}]}

    monkeypatch.setattr(mtp, "_complete", complete)
    measured = asyncio.run(mtp.measure_mtp(None, "target", repeats=5, gen_tokens=16, prompt="task"))
    assert measured.acceptance == 10 / 12
    assert measured.draft_tokens == 12
    assert len(measured.generation_samples) == 5
    assert all(c["cache_prompt"] is False for c in calls)
    assert all(c["temperature"] == 0.0 and c["seed"] == 42 for c in calls)
    assert mtp.MTPMeasurement().acceptance is None
    assert mtp.draft_counts({"draft_n": 2, "draft_n_accepted": 3}) is None


def test_missing_timings_or_short_generation_rejected(monkeypatch):
    async def complete(*args, **kwargs):
        return 1.0, {"timings": {"predicted_n": 3}}

    monkeypatch.setattr(mtp, "_complete", complete)
    with pytest.raises(RuntimeError, match="token count"):
        asyncio.run(mtp.measure_mtp(None, "target", repeats=5, gen_tokens=16, prompt="task"))


def test_decode_gain_cannot_hide_request_regression():
    assert not mtp.improves(result(20, 1200), result(10, 1000), 0.05)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    original = {
        "ctx": 8192,
        "spec_type": "draft-mtp",
        "spec_draft_model": "/models/mtp.gguf",
        "spec_draft_ngl": 999,
        "spec_draft_n_max": 3,
        "extra_flags": ["--reasoning", "auto", "--draft-p-min", "0.75"],
    }
    m = model(copy.deepcopy(original))
    cfg = Config(models=[m])
    state = copy.deepcopy(original)
    edits = []
    monkeypatch.setattr(mtp, "has_mtp_heads", lambda path: True)

    class ReadyClient(AsyncContextClient):
        async def get(self, *args, **kwargs):
            import httpx

            return httpx.Response(200, request=httpx.Request("GET", "http://test/health"))

    monkeypatch.setattr(mtp.httpx, "AsyncClient", ReadyClient)

    def update(edit):
        for k, v in edit.items():
            if v is None:
                state.pop(k, None)
            else:
                state[k] = v

    async def apply(client, name, edit):
        edits.append(copy.deepcopy(edit))
        update(edit)

    async def restore(client, name, edit, cfg):
        update(edit)

    monkeypatch.setattr(mtp, "_apply_edits", apply)
    monkeypatch.setattr(mtp, "_restore_final_state", restore)
    return cfg, m, original, state, edits, tmp_path / "custom.toml"


def run(harness):
    cfg, m, _, _, _, path = harness
    return asyncio.run(mtp.tune_mtp(cfg, m, CAPS, server_url="http://localhost", config_path=path))


def test_selects_confidence_winner_and_preserves_sidecar(monkeypatch, harness):
    _, _, original, state, edits, _ = harness

    async def measure(*args, **kwargs):
        if state.get("spec_type") is None:
            return result(10, 1000)
        fast = state.get("spec_draft_n_max") == 2 and state.get("spec_draft_p_min") == 0.5
        if fast:
            return result(20, 500)
        return result(12, 800) if state.get("spec_draft_n_max") == 2 else result(11, 900)

    monkeypatch.setattr(mtp, "measure_mtp", measure)
    report = run(harness)
    assert report.applied
    assert state["spec_draft_n_max"] == 2
    assert state["spec_draft_p_min"] == 0.5
    assert state["spec_draft_model"] == original["spec_draft_model"]
    assert state["extra_flags"] == original["extra_flags"]
    assert any("confirm original" == row["label"] for row in report.trials)
    assert all(e["spec_draft_model"] is None for e in edits if e["spec_type"] is None)
    assert len([r for r in report.trials if r["label"].startswith("mtp ")]) == 7
    json.dumps(asdict(report))


def test_no_improvement_restores_exact_recipe(monkeypatch, harness):
    async def measure(*args, **kwargs):
        return result()

    monkeypatch.setattr(mtp, "measure_mtp", measure)
    report = run(harness)
    assert not report.applied
    assert harness[3] == harness[2]


def test_target_only_can_win_without_loading_sidecar(monkeypatch, harness):
    async def measure(*args, **kwargs):
        return result(20, 500) if harness[3].get("spec_type") is None else result()

    monkeypatch.setattr(mtp, "measure_mtp", measure)
    report = run(harness)
    assert report.applied and report.selected == "target-only"
    assert "spec_draft_model" not in harness[3]
    assert "spec_type" not in harness[3]


@pytest.mark.parametrize("failure", [RuntimeError("failed"), asyncio.CancelledError()])
def test_failure_or_cancellation_restores(monkeypatch, harness, failure):
    async def measure(*args, **kwargs):
        raise failure

    monkeypatch.setattr(mtp, "measure_mtp", measure)
    with pytest.raises(type(failure)):
        run(harness)
    assert harness[3] == harness[2]


def test_confirmation_rejects_transient_gain(monkeypatch, harness):
    calls = 0

    async def measure(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 9 or harness[3].get("spec_type") is None:
            return result()
        return result(20, 500) if calls > 2 else result()

    monkeypatch.setattr(mtp, "measure_mtp", measure)
    report = run(harness)
    assert not report.applied and "confirmation" in report.reason
    assert harness[3] == harness[2]


def test_dry_run_never_contacts_server(monkeypatch, tmp_path):
    cfg = Config(models=[model()])
    cfg.server.admin_token = "test-token"
    path = tmp_path / "custom.toml"
    cfg.save(path)
    before = path.read_bytes()
    monkeypatch.setattr("arc_llama.server_caps.probe_server_caps", lambda path: CAPS)
    monkeypatch.setattr(mtp, "has_mtp_heads", lambda path: True)
    outcome = CliRunner().invoke(
        cli, ["--config", str(path), "speculative", "target", "--tune-mtp", "--dry-run"]
    )
    assert outcome.exit_code == 0, outcome.output
    assert "No settings changed" in outcome.output
    assert path.read_bytes() == before


@pytest.mark.parametrize("value", [True, "0.5", -1, 2])
def test_api_rejects_invalid_confidence(monkeypatch, tmp_path, value):
    from fastapi.testclient import TestClient
    from test_server import _edit_app

    app, cfg = _edit_app(monkeypatch, tmp_path)
    before = copy.deepcopy(cfg.models[0].recipe)
    with TestClient(app) as client:
        response = client.post("/admin/models/qwen/edit", json={"spec_draft_p_min": value})
    assert response.status_code == 400
    assert cfg.models[0].recipe == before


def test_api_sidecar_confidence_roundtrip(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from test_server import _edit_app

    app, cfg = _edit_app(monkeypatch, tmp_path)
    sidecar = tmp_path / "mtp.gguf"
    sidecar.write_bytes(b"GGUF")
    state = {"spec_draft_p_min": 0.75, "spec_draft_model": str(sidecar), "spec_draft_ngl": 999}
    with TestClient(app) as client:
        response = client.post("/admin/models/qwen/edit", json=state)
        assert response.status_code == 200
        assert all(cfg.models[0].recipe[k] == v for k, v in state.items())
        response = client.post("/admin/models/qwen/edit", json=dict.fromkeys(state))
    assert response.status_code == 200
    assert all(k not in cfg.models[0].recipe for k in state)


def test_server_must_acknowledge_cleared_sidecar():
    import httpx

    async def check():
        def respond(request):
            return httpx.Response(200, json={"recipe": {"spec_draft_model": "/old/mtp.gguf"}})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="http://test"
        ) as client:
            error = await mtp._apply_edits(client, "target", {"spec_draft_model": None})
            assert "complete MTP recipe" in error

    asyncio.run(check())


def test_partial_edit_failure_rolls_back(monkeypatch, harness):
    async def failed_edit(client, name, state):
        harness[3].pop("spec_draft_model", None)
        return "rebuild failed after persist"

    monkeypatch.setattr(mtp, "_apply_edits", failed_edit)
    with pytest.raises(RuntimeError, match="rebuild failed"):
        run(harness)
    assert harness[3] == harness[2]


def test_rollback_fallback_uses_custom_path_and_preserves_other_settings(monkeypatch, harness):
    cfg, _, original, _, _, path = harness
    cfg.server.admin_token = "test-token"
    cfg.server.port = 12345
    cfg.save(path)

    async def failed_measure(*args, **kwargs):
        fresh = load_config(path)
        fresh.models[0].recipe["spec_draft_n_max"] = 1
        fresh.server.port = 23456  # unrelated concurrent edit must survive
        fresh.save(path)
        raise RuntimeError("measurement failed")

    async def failed_restore(*args, **kwargs):
        return "server unavailable"

    monkeypatch.setattr(mtp, "measure_mtp", failed_measure)
    monkeypatch.setattr(mtp, "_restore_final_state", failed_restore)
    with pytest.raises(RuntimeError, match="Live rollback failed"):
        run(harness)
    repaired = load_config(path)
    assert repaired.models[0].recipe == original
    assert repaired.server.port == 23456


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_api_rejects_nonfinite_confidence(monkeypatch, tmp_path, literal):
    from fastapi.testclient import TestClient
    from test_server import _edit_app

    app, _ = _edit_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/admin/models/qwen/edit",
            content='{"spec_draft_p_min":' + literal + "}",
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 400


def test_unready_service_never_mutates_recipe(monkeypatch, harness):
    import httpx

    class UnreadyClient(AsyncContextClient):
        async def get(self, *args, **kwargs):
            raise httpx.ConnectError("service still starting")

    monkeypatch.setattr(mtp.httpx, "AsyncClient", UnreadyClient)
    with pytest.raises(httpx.ConnectError):
        run(harness)
    assert harness[4] == []
    assert harness[3] == harness[2]


def test_different_outputs_are_not_a_verified_speedup():
    baseline = result(10, 1000)
    candidate = result(20, 500)
    baseline.output_fingerprints = ["original"] * 5
    candidate.output_fingerprints = ["different"] * 5
    assert not mtp.improves(candidate, baseline, 0.05)
    candidate.output_fingerprints = baseline.output_fingerprints
    assert mtp.improves(candidate, baseline, 0.05)


def test_completion_sends_controlled_sampling():
    import httpx

    from arc_llama.benchmark import _complete

    class Client:
        async def post(self, url, json):
            assert json["temperature"] == 0.0
            assert json["seed"] == 42
            return httpx.Response(200, json={}, request=httpx.Request("POST", "http://test" + url))

    asyncio.run(_complete(Client(), "target", "task", 16, temperature=0.0, seed=42))


def test_missing_output_hashes_cannot_be_a_verified_speedup():
    baseline = result(10, 1000)
    candidate = result(20, 500)
    candidate.output_fingerprints = []
    assert not mtp.improves(candidate, baseline, 0.05)
