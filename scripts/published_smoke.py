"""Exercise an installed Arc Llama wheel using a pinned tiny CPU model.

Run with Python 3.12+ from a core-only environment, not an editable checkout. Downloads are
checksum-verified, all state is temporary, and no GPU layers are offloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

RUNTIME_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11381/llama-b11381-bin-ubuntu-x64.tar.gz"
RUNTIME_SHA = "7e235592575e8df35f525195b6d7ac773c9d50a4ddb070ba4fb1ff3c073ef4da"
MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_SHA = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
WORK_MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf"
WORK_MODEL_SHA = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"


def download(url: str, path: Path, expected: str) -> Path:
    if path.exists():
        with path.open("rb") as existing:
            if hashlib.file_digest(existing, "sha256").hexdigest() == expected:
                return path
    partial = path.with_suffix(".partial")
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {path.name}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model-size", choices=("0.5b", "1.5b"), default="0.5b")
    parser.add_argument("--expected-version")
    parser.add_argument("--ui-check", type=Path)
    parser.add_argument("--ui-check-timeout", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.ui_check_timeout <= 900:
        parser.error("--ui-check-timeout must be between 1 and 900 seconds")
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = download(RUNTIME_URL, root / "runtime.tar.gz", RUNTIME_SHA)
    model = (
        download(WORK_MODEL_URL, root / "work-1.5b.gguf", WORK_MODEL_SHA)
        if args.model_size == "1.5b"
        else download(MODEL_URL, root / "tiny.gguf", MODEL_SHA)
    )
    runtime_dir = root / "runtime"
    runtime_dir.mkdir(exist_ok=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(runtime_dir, filter="data")
    runtime = next(runtime_dir.rglob("llama-server"))
    if args.prepare_only:
        print("Pinned runtime and model verified.", flush=True)
        return

    from importlib.util import find_spec

    import httpx

    assert not any(find_spec(name) for name in ("textual", "fastembed", "mcp")), (
        "Use a core-only installed environment"
    )

    from arc_llama import __version__
    from arc_llama.config import Config, GPUConfig, ModelConfig

    if args.expected_version and __version__ != args.expected_version:
        raise AssertionError(f"Installed {__version__}; expected {args.expected_version}")
    cfg = Config()
    cfg.server.port = free_port()
    cfg.server.admin_token = "smoke-only-local-token"
    cfg.paths.state_dir = str(root / "state")
    cfg.paths.models_dir = str(root / "models")
    cfg.paths.skills_dir = str(root / "skills")
    cfg.paths.llama_server = str(runtime)
    cfg.tune.auto = False
    cfg.gpus = [GPUConfig("0000:ff:00.0", 0, "battlemage", vram_mb=4096, backend="vulkan")]
    cfg.models = [
        ModelConfig(
            name="published-smoke",
            path=str(model),
            gpu_pci_slot="0000:ff:00.0",
            port=free_port(),
            recipe={
                "ctx": 8192,
                "parallel": 2,
                "n_gpu_layers": 0,
                "threads": 2,
                "extra_flags": ["--device", "none"],
            },
        )
    ]
    config = root / "config.toml"
    cfg.save(config)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "ARC_LLAMA_ADMIN_TOKEN", "ARC_LLAMA_SMOKE_MODEL")
    }
    env["PYTHONUNBUFFERED"] = "1"
    url = f"http://127.0.0.1:{cfg.server.port}"
    with (root / "server.log").open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "arc_llama.cli",
                "-c",
                str(config),
                "serve",
                "--no-scan",
                "--no-auto-tune",
            ],
            cwd=root,
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            with httpx.Client(
                base_url=url,
                timeout=120,
                headers={"Authorization": "Bearer smoke-only-local-token"},
            ) as client:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise AssertionError("Installed server exited; inspect server.log")
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.2)
                else:
                    raise TimeoutError("Installed server startup timed out")
                payload = {
                    "model": "published-smoke",
                    "messages": [{"role": "user", "content": "Say hello briefly."}],
                    "max_tokens": 32,
                    "temperature": 0,
                }
                response = client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                assert response.json()["choices"][0]["message"]["content"].strip()
                payload["stream"] = True
                with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                    response.raise_for_status()
                    events = list(response.iter_lines())
                assert "data: [DONE]" in events
                assert not any('"error"' in line for line in events)
                # Verify all installed HTML script/style references, including new assets.
                from html.parser import HTMLParser

                class Assets(HTMLParser):
                    paths: list[str]

                    def __init__(self):
                        super().__init__()
                        self.paths = []

                    def handle_starttag(self, tag, attrs):
                        values = dict(attrs)
                        target = (
                            values.get("src")
                            if tag == "script"
                            else values.get("href")
                            if tag == "link" and values.get("rel") == "stylesheet"
                            else None
                        )
                        if target and target.startswith("/"):
                            self.paths.append(target)

                for page in ("/", "/chat", "/work"):
                    response = client.get(page)
                    response.raise_for_status()
                    assets = Assets()
                    assets.feed(response.text)
                    assert assets.paths
                    for path in assets.paths:
                        client.get(path).raise_for_status()
                if args.ui_check:
                    subprocess.run(
                        ["node", str(args.ui_check.resolve()), url],
                        check=True,
                        timeout=args.ui_check_timeout,
                        env={
                            **env,
                            "ARC_LLAMA_SMOKE_TOKEN": cfg.server.admin_token,
                            "ARC_WORK_SMOKE_MODEL_SIZE": args.model_size,
                        },
                    )
                status = client.get("/admin/status").json()
                assert status.get("models")
                client.post("/admin/stop-all").raise_for_status()
                print(
                    json.dumps(
                        {
                            "version": __version__,
                            "streaming": "pass",
                            "non_streaming": "pass",
                            "installed_assets": "pass",
                            "backend": "CPU",
                            "runtime": "b11381",
                            "model_size": args.model_size,
                            "ui_check": bool(args.ui_check),
                        }
                    ),
                    flush=True,
                )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    main()
