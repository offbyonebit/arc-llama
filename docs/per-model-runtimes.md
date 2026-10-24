# Per-model runtimes

Each model can choose its own llama-server executable and environment. Models
without overrides continue using `paths.llama_server` and the service environment.

Open the model's **Model settings → Runtime for this model**. Set its executable
path and enter environment variables as `KEY=value`, one per line. Click **Apply**.
Only that model's instance is drained and rebuilt; other models keep their launch
plans. The selected model loads on the next request. Clear both fields and Apply
to return to the defaults.

For example, Ornith can use a patched build while every other model keeps the
normal runtime:

```toml
[paths]
llama_server = "/opt/llama/default/llama-server"

[[models]]
name = "ornith"
path = "/models/ornith.gguf"
port = 18080
gpu_pci_slot = "0000:03:00.0"
llama_server = "/opt/llama/ornith-patched/llama-server"
env = { LD_LIBRARY_PATH = "/opt/llama/ornith-patched/lib", GGML_TEST_FLAG = "1" }
```

Existing recipe flags (including `extra_flags`) are appended as separate arguments.
The executable path is a single value and may contain spaces. Commands and environment
values are literal: no shell, variable interpolation, or command substitution runs.
Use a wrapper executable if startup needs more involved preparation.

GPU assignment and backend remain in the GPU configuration; choose a binary
compatible with that backend. Overrides apply after automatic GPU/oneAPI defaults, so an explicit environment
variable wins. They do not modify Arc Llama's own process or another model.
Environment values remain local configuration; tune fingerprints hash them, and
recipe submissions never include them.

Capability probes and local architecture checks use the selected runtime. Changing
the executable or environment invalidates that model's tune/compatibility identity.
Global runtime updates leave pinned model executables untouched and check only
models that inherit the global binary. An explicitly selected canary still tests
the candidate build.

The CLI also supports configuration without editing TOML:

```sh
arc-llama model-runtime ornith --llama-server /opt/llama/ornith-patched/llama-server --env LD_LIBRARY_PATH=/opt/llama/ornith-patched/lib
arc-llama model-runtime ornith --unset-env LD_LIBRARY_PATH
arc-llama model-runtime ornith --reset
```

CLI/TOML edits require restarting the running service to reload configuration.
Use the Model settings UI for a live change limited to one model. CLI output lists
environment names without printing values.
