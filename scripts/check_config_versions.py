"""Round-trip configs between this checkout and published arc-llama releases (issue #78).

Installs each release from PyPI into its own throwaway venv (with uv), then:

- upgrade: the release writes a config with every field set to a non-default value, this
  checkout loads it; every value must survive, and survive this checkout saving it again;
- rollback: this checkout writes such a config, the release loads and saves it, this checkout
  loads the result; reports exactly which settings the rollback cost.

Exits 1 if an upgrade loses anything. Rollback losses are reported, not failures: old releases
cannot keep keys they do not know (see docs/upgrading.md). --update-fixtures rewrites
tests/fixtures/config-versions from the real releases.

    .venv/bin/python scripts/check_config_versions.py [--versions 0.8.0 0.9.1.2] [--work-dir DIR]

Uses no GPU and no model; never touches the real config (HOME points at a scratch folder).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "config-versions"
DEFAULT_VERSIONS = ["0.8.0", "0.9.0", "0.9.1.2"]

# Run by each version's own interpreter: write a config with every field non-default, or
# load one (dumping the values that version sees) and optionally save it again.
PROBE = r'''
import dataclasses, json, logging, sys
from pathlib import Path
import arc_llama.config as C

def fill(obj):
    for f in dataclasses.fields(obj):
        cur = getattr(obj, f.name)
        hint = str(f.type)
        if dataclasses.is_dataclass(cur) or isinstance(cur, list) and cur and dataclasses.is_dataclass(cur[0]):
            continue
        if f.name == "unknown_keys":
            continue
        if isinstance(cur, bool):
            new = not cur
        elif isinstance(cur, int):
            new = cur + 7
        elif isinstance(cur, float):
            new = cur + 1.5
        elif isinstance(cur, dict):
            new = {"PROBE_KEY": "probe-value"}
        elif isinstance(cur, list):
            new = [f"probe-{f.name}"] if hint.startswith("list[str]") else cur
        elif cur is None or cur == "":
            if hint.startswith("int") or "int |" in hint:
                new = 7
            elif hint.startswith("float") or "float |" in hint:
                new = 1.5
            elif hint.startswith("str") or "str |" in hint:
                new = f"probe-{f.name}"
            else:
                new = cur
        else:
            new = cur  # non-empty strings are often validated choices; keep them
        setattr(obj, f.name, new)
    return obj

def entry(cls):
    req = {}
    for f in dataclasses.fields(cls):
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            req[f.name] = 7 if str(f.type).startswith("int") else f"probe-{f.name}"
    return fill(cls(**req))

if sys.argv[1] == "write":
    cfg = C.Config()
    for s in ("server", "paths", "tune", "workload", "agent"):
        fill(getattr(cfg, s))
    for s, cls in (("gpus", "GPUConfig"), ("models", "ModelConfig"), ("upstreams", "UpstreamConfig"),
                   ("mcp_servers", "MCPServerConfig"), ("profiles", "ProfileConfig")):
        getattr(cfg, s).append(entry(getattr(C, cls)))
    cfg.save(Path(sys.argv[2]))
else:
    msgs = []
    class H(logging.Handler):
        def emit(self, r): msgs.append(r.getMessage())
    logging.getLogger("arc_llama").addHandler(H())
    cfg = C.load_config(Path(sys.argv[2]))
    data = dataclasses.asdict(cfg)
    data.pop("unknown_keys", None)
    Path(sys.argv[3]).write_text(json.dumps({"config": data, "warnings": msgs}, indent=1, sort_keys=True, default=str))
    if len(sys.argv) > 4:
        cfg.save(Path(sys.argv[4]))
'''


def flat(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(flat(v, f"{prefix}.{k}" if prefix else k))
        return out
    if isinstance(value, list) and value and isinstance(value[0], dict):
        out = {}
        for i, v in enumerate(value):
            out.update(flat(v, f"{prefix}[{i}]"))
        return out
    return {prefix: value}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=DEFAULT_VERSIONS)
    ap.add_argument("--work-dir", type=Path, help="keep venvs here between runs (default: a temp dir)")
    ap.add_argument("--update-fixtures", action="store_true")
    args = ap.parse_args()
    if not shutil.which("uv"):
        print("uv is required to install the old releases", file=sys.stderr)
        return 2

    work = args.work_dir or Path(tempfile.mkdtemp(prefix="arc-llama-config-versions-"))
    work.mkdir(parents=True, exist_ok=True)
    out = work / "out"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    probe = work / "probe.py"
    probe.write_text(PROBE)
    home = work / "home"  # default paths in the fixtures become /home/user-like, not this machine's
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("XDG_", "ARC_LLAMA_"))}
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["APPDATA"] = str(home / "AppData" / "Roaming")
    env["LOCALAPPDATA"] = str(home / "AppData" / "Local")
    if os.name == "nt":
        env.setdefault("UV_NATIVE_TLS", "true")

    def run(python: Path | str, *argv: str) -> None:
        subprocess.run([str(python), "-I", str(probe), *argv], check=True, env=env, cwd=work)

    current = sys.executable
    run(current, "write", str(out / "current.toml"))
    run(current, "load", str(out / "current.toml"), str(out / "current.json"))
    original = flat(json.loads((out / "current.json").read_text())["config"])

    failed = False
    for version in args.versions:
        venv = work / f"venv-{version}"
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.exists():
            subprocess.run(["uv", "venv", "-q", str(venv)], check=True, env=env)
            subprocess.run(["uv", "pip", "install", "-q", "--python", str(python), f"arc-llama=={version}"], check=True, env=env)

        run(python, "write", str(out / f"{version}.toml"))
        run(python, "load", str(out / f"{version}.toml"), str(out / f"{version}-self.json"))
        run(current, "load", str(out / f"{version}.toml"), str(out / f"up-{version}.json"), str(out / f"up-{version}.toml"))
        run(current, "load", str(out / f"up-{version}.toml"), str(out / f"up2-{version}.json"))
        run(python, "load", str(out / "current.toml"), str(out / f"down-{version}.json"), str(out / f"down-{version}.toml"))
        run(current, "load", str(out / f"down-{version}.toml"), str(out / f"back-{version}.json"))

        theirs = flat(json.loads((out / f"{version}-self.json").read_text())["config"])
        up = json.loads((out / f"up-{version}.json").read_text())
        ours = flat(up["config"])
        resaved = flat(json.loads((out / f"up2-{version}.json").read_text())["config"])
        lost = sorted(k for k, v in theirs.items() if ours.get(k, "<missing>") != v)
        drift = sorted(k for k in ours if resaved.get(k) != ours[k])
        ok = not lost and not drift and not up["warnings"]
        failed |= not ok
        print(f"upgrade  {version} -> this checkout: {'ok' if ok else 'FAILED'} ({len(theirs)} values)")
        for k in lost:
            print(f"    lost {k}: {theirs[k]!r} -> {ours.get(k, '<missing>')!r}")
        for k in drift:
            print(f"    changed by re-saving: {k}")
        for w in up["warnings"]:
            print(f"    warning: {w}")

        back = flat(json.loads((out / f"back-{version}.json").read_text())["config"])
        cost = sorted(k for k in original if back.get(k) != original[k])
        print(f"rollback this checkout -> {version} -> this checkout: {len(cost)} setting(s) reset after {version} saves")
        for k in cost:
            print(f"    {k}: {original[k]!r} -> {back.get(k, '<default>')!r}")

        if args.update_fixtures:
            FIXTURES.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / f"{version}.toml", FIXTURES / f"written-by-{version}.toml")
            shutil.copyfile(out / f"down-{version}.toml", FIXTURES / f"rollback-resaved-by-{version}.toml")
            (FIXTURES / f"loaded-by-{version}.json").write_text(
                json.dumps(json.loads((out / f"{version}-self.json").read_text())["config"], indent=1, sort_keys=True)
            )
    if args.update_fixtures:
        from importlib.metadata import version as pkg_version

        shutil.copyfile(out / "current.toml", FIXTURES / f"written-by-{pkg_version('arc-llama')}.toml")
        print(f"fixtures updated in {FIXTURES}")
    if not args.work_dir:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
