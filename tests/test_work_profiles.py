"""Work profiles: validation, storage, how a profile shapes the OpenCode config, and the endpoints."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from helpers import ConfigRouter as FakeRouter

import arc_llama.server as server_mod
from arc_llama.agent.harness import opencode as oc
from arc_llama.agent.harness.opencode import build_config, filter_commands, group_tools, probe_tools
from arc_llama.agent.profiles import ProfileStore, clean_profile
from arc_llama.config import Config, ServerConfig

SERVERS = ["searxng", "github", "memory"]


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.asyncio
async def test_capability_discovery_marks_partial_catalogs(tmp_path, monkeypatch, failed):
    async def commands(*_a, **_k):
        return [{"name": "pdf", "description": "PDF", "source": "skill"}]

    async def tools(*_a, **_k):
        if failed:
            raise oc.OpenCodeError("synthetic discovery failure")
        return [{"id": "read", "group": "builtin"}, {"id": "question", "group": "builtin"}]

    monkeypatch.setattr(oc, "list_commands", commands)
    monkeypatch.setattr(oc, "list_tools", tools)
    monkeypatch.setattr(oc, "mcp_server_names", lambda _root: ["demo"])
    result = await oc.list_capabilities(tmp_path)
    assert result["tools_complete"] is not failed
    assert result["warnings"] == (["synthetic discovery failure"] if failed else [])
    assert result["tools"] == ([] if failed else [{"id": "read", "group": "builtin"}])
    assert result["skills"] == [{"name": "pdf", "description": "PDF"}]


def test_clean_profile_validates_and_dedupes():
    ok = clean_profile({"name": "  Research ", "mcp": ["searxng", "searxng"], "skills": ["pdf"], "web": True})
    assert ok == {"name": "Research", "description": "", "mcp": ["searxng"], "skills": ["pdf"], "tools_off": [], "web": True}
    for bad in ({}, {"name": " "}, {"name": "x", "mcp": "github"}, {"name": "x", "skills": [1]}, {"name": "x", "web": "yes"}, []):
        with pytest.raises(ValueError):
            clean_profile(bad)


def test_store_round_trip_unique_names_and_delete(tmp_path):
    store = ProfileStore(tmp_path / "profiles.json")
    store.save("profile-aaaa", {"name": "Writing", "skills": ["docs"]})
    store.save("profile-bbbb", {"name": "Code", "mcp": ["github"]})
    assert [p["name"] for p in store.list()] == ["Code", "Writing"]
    with pytest.raises(ValueError, match="already a profile named"):
        store.save("profile-cccc", {"name": "writing"})
    store.save("profile-aaaa", {"name": "Writing", "skills": []})              # renaming to itself is fine
    assert store.get("profile-aaaa")["skills"] == []
    with pytest.raises(ValueError):
        store.save("../bad", {"name": "x"})
    assert store.delete("profile-aaaa") and not store.delete("profile-aaaa")


def test_profile_starts_only_listed_servers_and_denies_the_rest():
    cfg = build_config("http://x", "m", mcp_servers=SERVERS, profile={"mcp": ["searxng"], "skills": ["pdf"], "web": False})
    assert cfg["mcp"] == {"searxng": {"enabled": True}, "github": {"enabled": False}, "memory": {"enabled": False}}
    rules = cfg["permission"]
    assert rules["github_*"] == "deny" and rules["memory_*"] == "deny"
    assert rules["searxng_*"] == "ask" and rules["searxng_list*"] == "allow"          # still guarded, as before
    assert "github_list*" not in rules
    assert rules["skill"] == {"*": "deny", "pdf": "allow"}
    assert all(rules[tool] == "deny" for tool in ("webfetch", "websearch", "codesearch"))


def test_profile_can_keep_web_tools_and_ignores_unknown_servers():
    cfg = build_config("http://x", "m", mcp_servers=SERVERS, profile={"mcp": ["searxng", "gone"], "skills": [], "web": True})
    assert cfg["mcp"]["searxng"] == {"enabled": True} and "gone" not in cfg["mcp"]
    assert cfg["permission"]["webfetch"] == "ask" and cfg["permission"]["skill"] == {"*": "deny"}


def test_no_profile_leaves_the_users_setup_alone():
    cfg = build_config("http://x", "m", mcp_servers=SERVERS)
    assert "mcp" not in cfg and "skill" not in cfg["permission"] and cfg["permission"]["github_*"] == "ask"


def test_filter_commands_hides_what_a_profile_switches_off():
    commands = [
        {"name": "review", "description": "", "source": "command", "takes_arguments": True},
        {"name": "pdf", "description": "", "source": "skill", "takes_arguments": False},
        {"name": "docs", "description": "", "source": "skill", "takes_arguments": False},
        {"name": "github:fix", "description": "", "source": "mcp", "takes_arguments": True},
        {"name": "memory:recall", "description": "", "source": "mcp", "takes_arguments": False},
    ]
    kept = filter_commands(commands, {"mcp": ["memory"], "skills": ["pdf"], "web": False})
    assert [c["name"] for c in kept] == ["review", "pdf", "memory:recall"]
    assert filter_commands(commands, None) is commands


@pytest.fixture
def client(tmp_path, monkeypatch):
    class Upstreams:
        def __init__(self, *_a, **_k):
            pass

        async def models(self):
            return []

        def find_model(self, _id):
            return None

        def upstreams_status(self):
            return []

    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", Upstreams)
    cfg = Config(server=ServerConfig(admin_token="secret"))
    cfg.paths.state_dir = str(tmp_path / "state")
    with TestClient(server_mod.create_app(cfg)) as c:
        yield c


AUTH = {"Authorization": "Bearer secret"}


def test_profile_endpoints_need_admin_and_round_trip(client):
    assert client.get("/v1/agent/profiles").status_code == 401
    assert client.get("/v1/agent/capabilities").status_code == 401
    made = client.put("/v1/agent/profiles/profile-aaaa", json={"name": "Research", "mcp": ["searxng"], "web": True}, headers=AUTH)
    assert made.status_code == 200 and made.json()["id"] == "profile-aaaa"
    assert [p["name"] for p in client.get("/v1/agent/profiles", headers=AUTH).json()["profiles"]] == ["Research"]
    assert client.put("/v1/agent/profiles/profile-bbbb", json={"name": "research"}, headers=AUTH).status_code == 400
    assert client.put("/v1/agent/profiles/x", json={"name": "A"}, headers=AUTH).status_code == 400
    assert client.delete("/v1/agent/profiles/profile-aaaa", headers=AUTH).json() == {"deleted": True}


def test_a_run_gets_its_profile_and_an_unknown_one_is_refused(client, tmp_path, monkeypatch):
    seen = {}

    async def fake_run(**kwargs):
        seen.update(kwargs)
        yield {"type": "done"}

    monkeypatch.setattr(server_mod, "run_opencode", fake_run)
    client.put("/v1/agent/profiles/profile-aaaa", json={"name": "Research", "mcp": ["searxng"]}, headers=AUTH)
    body = {"model": "m", "task": "x", "harness": "opencode", "root": str(tmp_path)}
    ok = client.post("/v1/agent", json={**body, "tool_profile_id": "profile-aaaa"}, headers=AUTH)
    assert ok.status_code == 200 and seen["tool_profile"]["mcp"] == ["searxng"]
    assert client.post("/v1/agent", json={**body, "tool_profile_id": "profile-zzzz"}, headers=AUTH).status_code == 400
    builtin = {"model": "m", "task": "x", "harness": "builtin", "tool_profile_id": "profile-aaaa"}
    assert client.post("/v1/agent", json=builtin, headers=AUTH).status_code == 400
    seen.clear()
    client.post("/v1/agent", json=body, headers=AUTH)
    assert seen["tool_profile"] is None


def test_capabilities_and_commands_follow_the_profile(client, tmp_path, monkeypatch):
    async def fake_caps(folder):
        return {"mcp": ["searxng", "github"], "skills": [{"name": "pdf", "description": "d"}], "web": ["webfetch"]}

    async def fake_list(folder, model):
        return [{"name": "pdf", "description": "", "source": "skill", "takes_arguments": False},
                {"name": "docs", "description": "", "source": "skill", "takes_arguments": False}]

    monkeypatch.setattr(server_mod, "list_capabilities", fake_caps)
    monkeypatch.setattr(server_mod, "list_commands", fake_list)
    caps = client.get("/v1/agent/capabilities", params={"root": str(tmp_path)}, headers=AUTH).json()
    assert caps["mcp"] == ["searxng", "github"] and caps["skills"][0]["name"] == "pdf"
    client.put("/v1/agent/profiles/profile-aaaa", json={"name": "P", "skills": ["pdf"]}, headers=AUTH)
    listed = client.get("/v1/agent/commands", params={"root": str(tmp_path), "profile_id": "profile-aaaa"}, headers=AUTH)
    assert [c["name"] for c in listed.json()["commands"]] == ["pdf"]
    everything = client.get("/v1/agent/commands", params={"root": str(tmp_path)}, headers=AUTH)
    assert len(everything.json()["commands"]) == 2


def test_individual_tools_are_denied_and_edit_tools_go_together():
    profile = {"mcp": ["github"], "skills": [], "web": True, "tools_off": ["github_create_pull_request", "write", "quota_status", "bash"]}
    rules = build_config("http://x", "m", mcp_servers=SERVERS, profile=profile)["permission"]
    assert rules["github_create_pull_request"] == "deny" and rules["quota_status"] == "deny" and rules["bash"] == "deny"
    assert rules["edit"] == "deny" and "write" not in rules          # OpenCode gates edit, write and apply_patch with one permission
    assert rules["github_*"] == "ask"                                # the rest of an allowed server stays guarded


def test_group_tools_sorts_builtin_mcp_and_plugin_tools():
    tools = group_tools(["bash", "github_get_me", "github_list_issues", "searxng_web_search", "quota_status", "read_mcp_resource"],
                        ["searxng", "github"])
    assert {t["id"]: t["group"] for t in tools} == {
        "bash": "builtin", "github_get_me": "github", "github_list_issues": "github",
        "searxng_web_search": "searxng", "quota_status": "plugin", "read_mcp_resource": "builtin"}
    assert group_tools(["a_b_x"], ["a", "a_b"])[0]["group"] == "a_b"             # the longest server name wins


async def test_probe_tools_reports_what_the_harness_sends_the_model(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_OPENCODE_SEND_TOOLS", "bash,github_create_pr,quota_status")
    monkeypatch.setenv("FAKE_OPENCODE_LOG", str(tmp_path / "log.jsonl"))

    async def stop(self):
        return None

    def terminate_now(self):
        import contextlib

        import httpx
        with contextlib.suppress(httpx.HTTPError):
            httpx.post(self.url + "/__exit", timeout=2)

    monkeypatch.setattr(oc._Server, "stop", stop)
    monkeypatch.setattr(oc._Server, "terminate_now", terminate_now)
    fake = str(Path(__file__).parent / "fixtures" / "fake_opencode.py")
    names = await probe_tools(tmp_path, build_config("http://x", "m"), binary=fake, timeout=20)
    assert names == ["bash", "github_create_pr", "quota_status"]
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any(e.get("method") == "DELETE" for e in log)                      # the scan session is removed afterwards
