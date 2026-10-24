# Upgrading and rolling back

Arc Llama keeps its settings in one TOML file: `~/.config/arc-llama/config.toml` on Linux (or
`$XDG_CONFIG_HOME/arc-llama/config.toml`), `%APPDATA%\arc-llama\config.toml` on Windows, or the file
given with `--config`. This page says what happens to that file when you move between versions.

## Upgrading

Upgrading keeps every setting. Configs written by 0.8.0, 0.9.0 and 0.9.1.2 load in this version
with every value unchanged and no warnings; settings added since then start at their defaults.
Nothing to do by hand.

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
