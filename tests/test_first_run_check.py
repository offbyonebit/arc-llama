from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "first_run_check.py"
spec = importlib.util.spec_from_file_location("first_run_check", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_environment_isolates_homes_caches_and_inherited_options(tmp_path):
    old_bin = tmp_path / "old-runtime"
    old_bin.mkdir()
    (old_bin / "llama-server").touch()
    inherited = {"HOME": "/real/home", "USERPROFILE": "/real/home", "XDG_CUSTOM": "/real/custom",
                 "ARC_LLAMA_PORT": "9999", "ARC_LLAMA_ADMIN_TOKEN": "secret", "HF_TOKEN": "secret",
                 "PYTHONPATH": "/real/modules", "PIP_INDEX_URL": "https://private.example/secret",
                 "PATH": str(old_bin), "SYSTEMROOT": "system", "HTTPS_PROXY": "proxy"}
    env = check.isolated_env(tmp_path / "work", inherited)
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
                 "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR", "XDG_CONFIG_DIRS", "XDG_DATA_DIRS",
                 "XDG_CUSTOM", "HERMES_HOME", "HF_HOME", "TEMP", "TMP"):
        folder = Path(env[name])
        assert folder.is_relative_to(tmp_path / "work") and folder.is_dir()
    assert not {"ARC_LLAMA_PORT", "ARC_LLAMA_ADMIN_TOKEN", "HF_TOKEN", "PYTHONPATH", "PIP_INDEX_URL"} & env.keys()
    assert str(old_bin) not in env["PATH"]
    assert env["SYSTEMROOT"] == "system" and env["HTTPS_PROXY"] == "proxy"
    assert env["PYTHONNOUSERSITE"] == "1" and env["PIP_CONFIG_FILE"] == os.devnull
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert inherited["HOME"] == "/real/home"  # Caller environment is never rewritten.


def test_argument_sources_are_required_and_exclusive(tmp_path):
    for argv in ([], ["--version", "1.3.0", "--dist", "x.whl"]):
        with pytest.raises(SystemExit):
            check.parser().parse_args(argv)
    assert check.parser().parse_args(["--version", "1.3.0rc1"]).version == "1.3.0rc1"
    dist = tmp_path / "package.whl"
    dist.touch()
    assert check.parser().parse_args(["--dist", str(dist)]).dist == dist
    assert check.main(["--dist", str(tmp_path / "missing.whl"), "--dry-run"]) == 1
    assert check.main(["--version", "1.3/invalid", "--dry-run"]) == 1


def test_dry_run_keeps_plan_but_never_installs_or_uses_network(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not spawn or connect")
    monkeypatch.setattr(check.subprocess, "run", forbidden)
    monkeypatch.setattr(check.subprocess, "Popen", forbidden)
    monkeypatch.setattr(check.socket, "socket", forbidden)
    assert check.main(["--version", "1.3.0", "--work-dir", str(tmp_path), "--dry-run"]) == 0
    plan_path, = tmp_path.glob("first-run-logs/*/plan.json")
    plan = json.loads(plan_path.read_text())
    assert not Path(plan["work_dir"]).exists()
    assert plan["commands"][1][-1] == "arc-llama==1.3.0"
    assert plan["commands"][2][-5:-1] == ["arc_llama", "start", "--no-open", "--port"]
    assert "DRY RUN" in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["Hello from the model.", ""])
def test_guided_flow_sends_one_chat_and_requires_nonempty_answer(monkeypatch, answer):
    calls = []
    states = iter([
        {"gpu": {"name": "fake"}, "phase": "idle", "starter": {"id": "chosen"}},
        {"gpu": {"name": "fake"}, "phase": "ready", "model": "downloaded-model"},
    ])
    def request(url, body=None, token=None):
        calls.append((url, body))
        if url.endswith('/health'):
            return {}
        if url.endswith('/admin/session-token'):
            return {"admin_token": "test-only"}
        if url.endswith('/admin/onboarding'):
            return next(states) if body is None else {}
        return {"choices": [{"message": {"content": answer}}]}
    monkeypatch.setattr(check, "request_json", request)
    monkeypatch.setattr(check.time, "sleep", lambda _: None)
    if answer:
        assert check.first_answer("http://local", SimpleNamespace(poll=lambda: None), 10, io.StringIO()) == answer
    else:
        with pytest.raises(RuntimeError, match="empty answer"):
            check.first_answer("http://local", SimpleNamespace(poll=lambda: None), 10, io.StringIO())
    chats = [(url, body) for url, body in calls if url.endswith('/v1/chat/completions')]
    assert len(chats) == 1 and chats[0][1]["model"] == "downloaded-model"
    assert ("http://local/admin/onboarding", {"action": "start", "starter": "chosen"}) in calls


def test_url_audit_captures_actual_statuses_and_redacts_secrets(tmp_path):
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    shutil.copyfile(SCRIPT.with_name("first_run_url_audit.py"), hooks / "sitecustomize.py")
    log = tmp_path / "urls.jsonl"
    env = {**os.environ, "PYTHONPATH": str(hooks), "ARC_FIRST_RUN_URL_LOG": str(log)}
    code = '''import httpx
def respond(request):
    if request.url.path == '/broken': raise httpx.ConnectError('fake', request=request)
    return httpx.Response(404 if request.url.path == '/missing' else 200)
with httpx.Client(transport=httpx.MockTransport(respond)) as client:
    client.get('https://example.test/ok?token=private-value')
    client.get('https://example.test/missing')
    client.get('http://127.0.0.1/health')
    try: client.get('https://example.test/broken')
    except httpx.ConnectError: pass
import io,urllib.request,urllib.response
from email.message import Message
class FakeHTTP(urllib.request.BaseHandler):
    handler_order=100
    def https_open(self, request):
        redirected=request.full_url.endswith('/start')
        headers=Message()
        if redirected: headers['Location']='https://urllib.test/end'
        response=urllib.response.addinfourl(io.BytesIO(b'ok'),
            headers,
            request.full_url, 302 if redirected else 200)
        response.msg='Found' if redirected else 'OK'
        return response
with urllib.request.build_opener(FakeHTTP()).open('https://urllib.test/start') as response:
    response.read()
from pip._vendor import requests as pip_requests
class Adapter(pip_requests.adapters.BaseAdapter):
    def send(self, request, **kwargs):
        response=pip_requests.Response()
        response.status_code=201; response.url=request.url; response._content=b'ok'
        return response
    def close(self): pass
with pip_requests.Session() as session:
    session.mount('https://', Adapter())
    session.get('https://pip.test/package.whl')
'''
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=15)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row['status'] for row in rows[:3]] == [200, 404, 0]
    assert ('https://urllib.test/start', 302) in [(row['url'], row['status']) for row in rows]
    assert ('https://urllib.test/end', 200) in [(row['url'], row['status']) for row in rows]
    assert ('https://pip.test/package.whl', 201) in [(row['url'], row['status']) for row in rows]
    assert 'private-value' not in log.read_text()
    assert 'REDACTED' in rows[0]['url']
