"""config.toml must survive a failed write, and must not be world-readable.

The old save opened the target with "wb", which truncates immediately, so any
failure between that and the final byte left a partial file behind. The nasty
part is that a truncated TOML file is usually still *valid* TOML: the next
start loaded it without complaint and the user came up with no models, no
GPUs and no admin token, with nothing in the logs to explain where their
setup went. A hard crash would have been kinder.

The file also holds server.admin_token, which is the credential for every
destructive admin endpoint, so its permissions matter.

No GPU or llama.cpp backend needed.
"""

from __future__ import annotations

import os
import sys

import pytest

import arc_llama.config as cfgmod
from arc_llama.config import Config, load_config


def _saved(tmp_path):
    cfg = Config()
    path = tmp_path / "config.toml"
    cfg.save(path)
    return cfg, path


def test_config_survives_a_failure_midway_through_serialising(tmp_path):
    """Disk full, a bad value, a kill: the previous config must still be there."""
    cfg, path = _saved(tmp_path)
    original = path.read_bytes()
    assert original

    real_dump = cfgmod.tomli_w.dump

    def exploding_dump(obj, f):
        f.write(b"[server]\n")  # a partial write reaches the file
        raise OSError(28, "No space left on device")

    cfgmod.tomli_w.dump = exploding_dump
    try:
        with pytest.raises(OSError):
            cfg.save(path)
    finally:
        cfgmod.tomli_w.dump = real_dump

    assert path.read_bytes() == original, "a failed save destroyed the existing config"


def test_truncated_config_would_have_loaded_clean(tmp_path):
    """Documents why this was silent rather than loud: the wreckage parses.

    Nothing here exercises save(); it pins down the property that made the
    old bug so damaging, so nobody later decides a partial write is tolerable
    because "it would just fail to load".
    """
    path = tmp_path / "config.toml"
    path.write_text("[server]\n")
    loaded = load_config(path)
    assert loaded.models == [], "expected the truncated remnant to load as an empty config"


def test_no_temp_file_is_left_behind_when_saving_fails(tmp_path):
    cfg, path = _saved(tmp_path)
    real_dump = cfgmod.tomli_w.dump

    def exploding_dump(obj, f):
        raise OSError(28, "No space left on device")

    cfgmod.tomli_w.dump = exploding_dump
    try:
        with pytest.raises(OSError):
            cfg.save(path)
    finally:
        cfgmod.tomli_w.dump = real_dump

    # Only the save's own scratch files matter here; the shared test fixture
    # puts an isolated .config/ tree in tmp_path too.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".config.toml")]
    assert not leftovers, f"temp files left behind: {leftovers}"


def test_saved_config_is_not_world_readable(tmp_path):
    """server.admin_token authenticates every destructive admin endpoint."""
    if sys.platform == "win32":
        pytest.skip("POSIX permission bits do not apply on Windows")
    _cfg, path = _saved(tmp_path)
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600, f"config.toml is {oct(mode)}; the admin token is readable by other users"


def test_permissions_are_tightened_on_rewrite(tmp_path):
    """A config written by an older version comes back as 0600, not left open."""
    if sys.platform == "win32":
        pytest.skip("POSIX permission bits do not apply on Windows")
    cfg, path = _saved(tmp_path)
    os.chmod(path, 0o644)
    cfg.save(path)
    assert path.stat().st_mode & 0o777 == 0o600


def test_save_replaces_content_rather_than_appending(tmp_path):
    """The rename must land the new file over the old one, not merge with it."""
    cfg, path = _saved(tmp_path)
    cfg.server.port = 12345
    cfg.save(path)
    text = path.read_text()
    assert text.count("[server]") == 1, "old content survived alongside the new"
    assert load_config(path).server.port == 12345


def test_repeated_saves_stay_loadable(tmp_path):
    cfg, path = _saved(tmp_path)
    for port in range(11000, 11010):
        cfg.server.port = port
        cfg.save(path)
        assert load_config(path).server.port == port


def test_save_creates_missing_parent_directory(tmp_path):
    cfg = Config()
    path = tmp_path / "nested" / "deeper" / "config.toml"
    cfg.save(path)
    assert path.exists()
    assert load_config(path) is not None


# ---------------------------------------------------------------------------
# A rejected edit must leave memory and disk agreeing.
# ---------------------------------------------------------------------------


async def test_failed_persist_rolls_back_the_edit_and_reports_failure(monkeypatch, tmp_path):
    """The handler used to log the persist failure and carry on: the caller got
    a 200 listing the fields it "changed", the running server was rebuilt to
    match, and the config on disk still held the old recipe. The edit then
    quietly un-applied at the next restart. Fail the request instead."""
    import httpx
    from test_server import FakeRouter, FakeUpstreamManager

    import arc_llama.server as server_mod
    from arc_llama.config import ModelConfig, ServerConfig

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)

    model = ModelConfig(
        name="qwen",
        path=str(tmp_path / "qwen.gguf"),
        port=18080,
        gpu_pci_slot="0000:03:00.0",
        recipe={"ctx": 4096},
    )
    cfg = Config(server=ServerConfig(admin_token=None), models=[model])
    app = server_mod.create_app(cfg)

    def refuse_to_save(self, path=None):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Config, "save", refuse_to_save)
    # httpx.ASGITransport does not drive FastAPI lifespan hooks, so provide
    # the small set of state objects this admin route normally receives from
    # lifespan startup.
    app.state.cfg = cfg
    app.state.router = FakeRouter(cfg)
    app.state.upstream_mgr = FakeUpstreamManager(cfg.upstreams)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/admin/models/qwen/edit", json={"ctx": 8192})

    assert resp.status_code == 500, f"caller was told the edit succeeded: {resp.status_code}"
    assert model.recipe == {"ctx": 4096}, f"in-memory recipe was not rolled back: {model.recipe}"


def test_config_directory_is_private(tmp_path):
    """The directory holds the admin token as well; keep it 0700 so the token
    is protected even if the file's own mode is lost."""
    if sys.platform == "win32":
        pytest.skip("POSIX permission bits do not apply on Windows")
    d = tmp_path / "cfgdir"
    d.mkdir(mode=0o755)
    Config().save(d / "config.toml")
    assert d.stat().st_mode & 0o777 == 0o700


def test_concurrent_saves_do_not_share_a_temp_name(tmp_path):
    """Two writers must not pick the same scratch file and corrupt each other.
    Threads rather than processes, since that is what a single pid cannot
    disambiguate."""
    import threading

    path = tmp_path / "config.toml"
    Config().save(path)
    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def writer(port: int):
        try:
            barrier.wait()
            c = Config()
            c.server.port = port
            c.save(path)
        except BaseException as e:  # noqa: BLE001 - recorded, asserted below
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(11000 + i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors, f"concurrent saves raised: {errors}"
    assert not [p.name for p in tmp_path.iterdir() if p.name.startswith(".config.toml")]
    # Whoever renamed last wins, but the file must be whole and loadable.
    assert load_config(path).server.port in range(11000, 11006)


def test_missing_config_does_not_claim_token_was_saved(tmp_path, caplog):
    """load_config on a missing file generates an in-memory token with
    persist=False, but the old message still said "saved it to <path>" —
    sending users hunting for a file that was never written."""
    import logging

    missing = tmp_path / "nope" / "config.toml"
    with caplog.at_level(logging.WARNING, logger="arc_llama.config"):
        cfg = load_config(missing)

    assert cfg.server.admin_token, "no token generated at all"
    assert not missing.exists(), "persist=False path unexpectedly wrote a file"
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "saved it to" not in text, f"message claims a save that never happened: {text}"
    assert "in-memory" in text
