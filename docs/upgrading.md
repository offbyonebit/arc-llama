# Upgrading and rolling back

Arc Llama keeps its settings in one TOML file: `~/.config/arc-llama/config.toml` on Linux (or
`$XDG_CONFIG_HOME/arc-llama/config.toml`), `%APPDATA%\arc-llama\config.toml` on Windows, or the file
given with `--config`. This page says what happens to that file when you move between versions.

Arc Llama keeps two kinds of state, and they move between versions differently:

- **Configuration:** the TOML file above.
- **Application data:** under `paths.state_dir`, which defaults to `~/.local/state/arc-llama` (or
  `$XDG_STATE_HOME/arc-llama`). It includes:
  - chats (`chats/`);
  - Work projects and jobs (`work/work.sqlite3` plus job copies);
  - Work conversations, profiles and Undo backups;
  - installed runtimes and harnesses;
  - caches.

## Upgrading

Upgrading keeps every setting. Configs written by 0.8.0, 0.9.0, 0.9.1.2 and 1.0.0 load in this version
with every value unchanged and no warnings; settings added since then start at their defaults.
Nothing to do by hand.

### 1.0 and 1.1

**Configuration:** fully compatible in both directions.
- Upgrading keeps all 57 settings 1.0.0 writes.
- Rolling back costs nothing: 1.0.0 already keeps settings it doesn't know. That includes 1.1's
  `agent.opencode_bin` and `agent.hermes_bin`, which 1.0.0 doesn't use, so Work on 1.0.0 falls back
  to OpenCode or Hermes on `PATH`.
- Unknown sections, unknown keys inside known sections and unknown keys inside model entries all
  survive saves by either version.
- One difference: 1.0.0 writes `ARC_LLAMA_ADMIN_TOKEN` from the environment into the file when it
  saves; 1.1 does not.

**Chats and Work jobs:** the storage formats are identical in 1.0.0 and 1.1, so no migration runs
either way. Moving between them:
- keeps every chat, including interrupted replies;
- keeps workspaces and jobs;
- never runs a completed job again;
- pauses a job whose model call was cut off by the restart, for manual resume; that call is never
  replayed.

The rehearsal results are in the rehearsal section below and in `docs/release-readiness-1.1.md`.

## Rolling back to an older version

An older version starts with a newer config, but it does not know the settings added after it. It
logs one line for each, such as:

```
Ignoring unknown config key 'llama_server' in ModelConfig
```

and runs without them. **If the older version then saves the config** (for example after a change
in the dashboard or a model download), those settings are removed from the file. Upgrading again
brings them back at their defaults, not your values.

Settings a rollback can reset:

| Rolling back to | Settings reset once that version saves |
|---|---|
| 0.9.0 or 0.9.1.x | Per-model `llama_server` and `env` (dedicated runtimes); `server.inference_first_response_seconds`, `server.inference_stream_idle_seconds` |
| 0.8.x | The above, plus `server.switch_drain_seconds` |

To roll back without losing them:

1. Stop Arc Llama and copy the config file somewhere safe, e.g.
   `cp ~/.config/arc-llama/config.toml ~/arc-llama-config.before-rollback.toml`.
2. Install the older version and use it.
3. After upgrading again, copy the saved file back before starting Arc Llama.

## Backing up before an upgrade

Copying only the config file protects settings. It does **not** protect chats or Work jobs: in the
1.1 rehearsal, a failed upgrade that damaged `work/work.sqlite3` left Work unreadable after a
config-only restore. To be able to return to exactly where you were:

1. Stop Arc Llama (`systemctl --user stop arc-llama.service`, or stop `arc-llama serve`). Copying
   while it runs can capture a half-written database.
2. Copy the config folder (the config file and your skills) and the whole state directory:

   ```bash
   mkdir -p ~/arc-llama-backup
   cp -a ~/.config/arc-llama ~/arc-llama-backup/config
   cp -a ~/.local/state/arc-llama ~/arc-llama-backup/state
   ```

   Use your `--config` path, `paths.state_dir` and `paths.skills_dir` if you changed them. Downloaded
   models (`paths.models_dir`) are not touched by upgrades and need no copy. The state directory also
   holds installed runtimes and harnesses, which can be large. Excluding `runtime/` and `harnesses/` is safe; they
   can be reinstalled.
3. Upgrade and start Arc Llama.
4. If the upgrade fails: stop Arc Llama, install the previous version, replace both folders with the
   copies, then start it:

   ```bash
   rm -rf ~/.local/state/arc-llama
   cp -a ~/arc-llama-backup/state ~/.local/state/arc-llama
   rm -rf ~/.config/arc-llama
   cp -a ~/arc-llama-backup/config ~/.config/arc-llama
   ```

The backup contains your chats and the admin token, so keep it private. The config file is saved
with owner-only permissions; `cp -a` keeps them.

## The 1.0 and 1.1 rehearsal

The rehearsal used installed wheels:
- `arc_llama-1.0.0-py3-none-any.whl`, SHA-256
  `f2b11d418b09a9f8bf65e2aaf51f0528cbf78b478943e0b4f8458ab8751f42d2` (the PyPI release);
- `arc_llama-1.1.0.dev0-py3-none-any.whl`, SHA-256
  `8f9bc164688e6604e563ae20ca249e0c7c6d8ccaea6b22a7395160e0b3c0869d` (the candidate).

Each ran in its own virtual environment, with isolated `HOME`/`XDG_*`/`HERMES_HOME` and fully
synthetic state:
- a hand-edited config with custom values and unknown keys at three levels;
- two chats, one ending in an interrupted reply;
- two Work projects in temporary folders;
- a synthetic GGUF registered through `arc-llama add` and `model-runtime`, served by a scripted
  stand-in for `llama-server` that logs every request;
- one completed job, and one job left mid model-call by stopping the server.

It moved 1.0.0 → 1.1 → 1.0.0 → 1.1. Every transition passed:
- startup;
- config preserved;
- chats (including the interrupted reply) visible;
- workspaces and jobs visible;
- the completed job made no new model calls;
- the cut-off job paused with no replayed call;
- a chat edit, a workspace edit and a model setting persisted across a restart;
- unknown config keys survived each version's saves;
- an independent inventory, read from the files rather than through Arc Llama, matched.

The full config-plus-state backup above restored the original inventory exactly after a simulated
failed upgrade, and 1.0.0 ran normally on it.

**Not covered:** these were Linux, CPU-only runs with a stand-in model server. The rehearsal did not
cover Windows, a real GPU, real models, or Work conversations made with OpenCode or Hermes.

From this version on, Arc Llama keeps settings it does not recognise: a config written by a later
version keeps those settings when this version saves it, so rolling back from future releases to
this one does not lose them.

## Checking it yourself

`scripts/check_config_versions.py` installs published releases in throwaway environments (no GPU,
no model, your real config untouched) and repeats both round trips:

```
.venv/bin/python scripts/check_config_versions.py --versions 0.8.0 0.9.1.2
```

The automated suite runs the same checks on fixtures those releases wrote
(`tests/test_config_versions.py`).
