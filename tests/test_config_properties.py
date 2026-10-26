"""Generated migration invariants using only synthetic, reproducible configs."""

from __future__ import annotations

import copy
import random
import sys

import pytest
import tomli_w

from arc_llama.config import CONFIG_VERSION, load_config, migrate_config

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def generated_value(rng: random.Random, depth: int = 0):
    scalars = [True, False, rng.randrange(-100000, 100000), "", "日本語 🦙", "line\nquote\"", "C:\\models\\test"]
    if depth >= 3 or rng.randrange(3) == 0:
        return rng.choice(scalars)
    if rng.randrange(2):
        return [generated_value(rng, depth + 1) for _ in range(rng.randrange(4))]
    return {key: generated_value(rng, depth + 1) for key in rng.sample(["ordinary", "quoted.key", "newline\nkey", "unicode-🦙"], rng.randrange(1, 5))}


@pytest.mark.parametrize("seed", range(16))
@pytest.mark.parametrize("version", range(1, CONFIG_VERSION + 1))
def test_migrations_are_idempotent_and_future_values_survive_edits(tmp_path, seed, version):
    rng = random.Random(seed)
    future = [generated_value(rng) for _ in range(5)]
    raw = {
        "version": version,
        "server": {"port": 11437, "admin_token": "synthetic-token", "future_setting": future[0]},
        "paths": {"state_dir": str(tmp_path / "state"), "future_setting": future[1]},
        "models": [{"name": "synthetic", "port": 18080, "gpu_pci_slot": "00:00.0", "path": str(tmp_path / "synthetic.gguf"), "future_setting": future[2]}],
        "gpus": [{"pci_slot": "00:00.0", "sycl_index": 0, "arch": "battlemage", "future_setting": future[3]}],
        "future_section": future[4],
    }
    migrated = migrate_config(copy.deepcopy(raw))
    assert migrate_config(copy.deepcopy(migrated)) == migrated
    path = tmp_path / "config.toml"
    path.write_text(tomli_w.dumps(raw), encoding="utf-8")
    for port in [12340, 12341]:
        config = load_config(path)
        config.server.port = port
        config.save(path)
        saved = tomllib.loads(path.read_text(encoding="utf-8"))
        assert saved["server"]["port"] == port
        assert [saved["server"]["future_setting"], saved["paths"]["future_setting"],
                saved["models"][0]["future_setting"], saved["gpus"][0]["future_setting"], saved["future_section"]] == future
