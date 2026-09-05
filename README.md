# arc-llama

> Plug-and-play `llama.cpp` runtime for Intel Arc GPUs.

`arc-llama` is a single command-line tool that detects your Intel Arc card,
applies the right SYCL/oneAPI environment for your generation, downloads or
registers GGUF models, and runs an OpenAI-compatible server in front of them.
It encodes the gotchas (SIGSEGVs in the persistent device-code cache, IPEX-LLM
bundle env-var traps, KV-cache quant behaviour per architecture) so you don't
have to discover them the hard way.

It's built for the day you unbox an Arc card, install drivers, and want
something useful before lunch.

> ⭐ If this saved you a few hours, a star on this repo keeps me building.

> [!NOTE]
> **Status: 0.6.2.** Tested end-to-end on Battlemage B60 on Linux and Windows:
> `arc-llama install-runtime` fetches a portable Vulkan `llama-server` and
> serves real inference with no oneAPI install or source build. HF download,
> streaming, and the OpenAI-compatible API all pass. Other SKUs (A770, A380,
> B580) need community confirmation -- open an issue if something breaks on
> your card.

## What you get

- **Auto-discovery of GPUs *and models*.** `arc-llama init` finds your Intel
  card and walks the configured scan paths for `.gguf` files, registering
  every one with a sensible recipe , context length sized to your VRAM,
  KV-cache class inferred from the filename. You should never need
  `arc-llama add` for a GGUF that's already on disk.
- **Auto-discovery** of every Intel GPU on the host (`Alchemist`, `Battlemage`,
  Lunar Lake iGPU). PCI device-ID table covers the common SKUs and falls back
  to OpenCL device-name parsing for the rest.
- **Per-arch SYCL profiles** , env vars like `SYCL_CACHE_PERSISTENT=0` are
  applied automatically, and known-bad ones (e.g. `GGML_SYCL_DISABLE_OPT`,
  `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS`) are stripped from the
  inherited shell environment.
- **Smart defaults** for `-ctx`, `--cache-type-k/v`, and `-ngl` based on the
  detected VRAM and the model's quantized tensor size, never starts a model you can't fit.
- **Model registry** in TOML at `$XDG_CONFIG_HOME/arc-llama/config.toml`,
  trivially editable.
- **One process per model**, swapped in/out by an internal router. Default
  policy is single-resident across all GPUs (good for thermals); flip it to
  multi-resident if you have headroom.
- **OpenAI-compatible API** at `http://127.0.0.1:11437/v1/...`. Plug it into
  Open WebUI, OpenCode, anything that speaks OpenAI.
- **A web UI** at `http://127.0.0.1:11437/` , ships with the install. Model
  picker, load/stop buttons, **inline ctx + KV-quant editing**, GPU + VRAM
  panel. Pure HTML/JS, no build step.
- **A terminal UI** (`arc-llama tui`) using Textual , same load/stop/edit
  controls, no browser needed. Optional install: `pip install 'arc-llama[tui]'`.
- **Background autotune.** Drop in a GGUF, use it once, and `arc-llama serve`
  measures a faster recipe in the next idle window — no manual `tune`. Sweeps
  abort instantly if a real request arrives.
- **No magic with your existing stack.** It uses your `llama-server` binary;
  you're never locked into a specific build.

## Quick start

```bash
# 1. Install
pip install arc-llama

# Or install in editable mode for development:
# git clone https://github.com/offbyonebit/arc-llama
# cd arc-llama
# pip install -e .

# 2. Detect GPUs and write a starter config (no llama-server needed yet)
arc-llama init

# 3. Download a portable Vulkan llama-server and wire it into the config.
#    No oneAPI, no building llama.cpp. (Use --backend sycl for the SYCL build.)
arc-llama install-runtime

# 4. Look at what was found
arc-llama doctor
arc-llama gpus

# 5. Auto-register every GGUF found under your scan paths.
#    `init` ran this once; rerun any time you drop new files in.
arc-llama scan
# (or for one-offs: arc-llama add /path/to/some.gguf,
#  or HF: arc-llama add unsloth/gemma-4-31B-it-GGUF:Q4_K_M --from-hf)

# 6. Run the OpenAI-compatible server (also serves the web UI at /)
arc-llama serve

# 7. Drop a GGUF and use it once — auto-tune fires after the idle window,
#    or tune manually now:
arc-llama benchmark <model>
arc-llama tune <model>
arc-llama tune --status            # print per-model tune state, no sweep
arc-llama serve --no-auto-tune     # disable the background sweeps

# 8. (Optional) Open the terminal UI in another window
arc-llama tui

# 9. (Optional) Install a systemd --user unit
arc-llama systemd --write
systemctl --user daemon-reload
systemctl --user enable --now arc-llama.service
```

Then point any OpenAI-compatible client at `http://127.0.0.1:11437/v1`:

```bash
curl http://127.0.0.1:11437/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-31b-q4_k_m",
    "messages": [{"role": "user", "content": "hi"}]
  }'
```

## Requirements

### Linux

- Kernel **6.14+ recommended** for Battlemage (`xe` driver; 6.8 is the minimum
  where `xe` exists, but 6.14+ is stable for BMG) or 5.17+ for Alchemist
  (`i915`). This matches the threshold `arc-llama doctor` warns on.
- User in the `render` and `video` groups (`arc-llama doctor` will tell you).

### Windows

- Windows 10/11 with Intel Arc graphics drivers installed.
- For the SYCL backend: Intel oneAPI Base Toolkit installed.
- `arc-llama doctor` will report which tools and runtime libraries it finds.
- The `systemd` command is not available on Windows; use Task Scheduler or run
  `arc-llama serve` manually.

### Both platforms

- ReBAR enabled in BIOS; without it llama.cpp falls back to slow paths on Arc.
- A `llama-server` built with the SYCL or Vulkan backend.

For a SYCL build on Linux, the supported path is:
```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --config Release -j
```

If you installed oneAPI to a non-standard prefix (tarball install, relocated
`/opt/intel`, etc.), arc-llama detects `setvars.sh` (Linux) or `setvars.bat`
(Windows) from `$ONEAPI_ROOT`, `$CMPLR_ROOT`, and common prefixes, and sources
it automatically when the runtime libraries are not visible to the system
loader. You can also pin the path in `config.toml`:

```toml
[paths]
oneapi_setvars = "/your/prefix/oneapi/setvars.sh"   # Linux
# oneapi_setvars = "C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat"  # Windows
```

## Benchmark & autotune

Static defaults can't know whether *your* card/model/llama.cpp build prefers
f16 or q8_0 KV cache, a 512 or 2048 ubatch, or flash attention on/off — the
SYCL backend's answer genuinely differs per SKU and per revision. So measure:

```bash
# One-shot measurement (prompt-eval + generation tok/s, VRAM)
arc-llama benchmark qwen3-7b

# Sweep context lengths × KV types
arc-llama benchmark qwen3-7b --sweep-ctx 8192,32768 --kv f16 --kv q8_0

# Staged greedy sweep: KV type → ubatch → flash attention. Winner is written
# into the model's recipe and persisted. ~6–9 configs, ~10 min on Battlemage.
arc-llama tune qwen3-7b
arc-llama tune qwen3-7b --target generation   # optimise chat latency only
arc-llama tune qwen3-7b --dry-run             # look, don't touch
arc-llama tune --status                       # print state, no sweep
```

Manual `tune` needs a running `arc-llama serve` so measurements inherit the
exact SYCL env and router policy your real requests get. When enabled, the
background autotuner runs the same `tune_model` path over loopback HTTP. Set
`[tune] auto = false` or pass `--no-auto-tune` to disable. Candidates that fail
to start (e.g. compute-buffer OOM from a bigger ubatch) simply lose the round
— the tuner always leaves the model in a working config.

arc-llama also probes your `llama-server --help` once per binary to emit the
right flag dialect (`-fa on|off|auto` on current builds vs boolean `-fa` on
pre-b6300 ones), so hand-built and prebuilt binaries both work.

## Multi-GPU

## Speculative decoding

Arc Llama supports llama.cpp's native speculative-decoding modes when the
installed `llama-server` advertises them. Existing MTP GGUFs continue to be
auto-configured. For ordinary models, register a smaller model from the same
family and let Arc Llama choose it conservatively:

```bash
arc-llama speculative qwen3-30b --dry-run
arc-llama speculative qwen3-30b --auto
# or: arc-llama speculative qwen3-30b --draft qwen3-4b
arc-llama speculative qwen3-30b --ngram
```

Drafts are stored by registered model name and resolved only when the target
starts. They must use a tokenizer compatible with the target; using a smaller
model from the same family is a candidate, not proof. Arc Llama checks the llama.cpp help surface and falls back to normal
target-only inference if the requested flags are unavailable. Same-GPU drafts
can be slower or consume too much VRAM, so `--auto` selects a plausible local
candidate but does not claim a speedup: benchmark it on your own hardware
before relying on it. Cross-vendor or cross-GPU draft/target pipelines are not
implemented; verification remains inside one llama-server process.

`arc-llama init` registers every Intel GPU it finds. Each model in the config
is bound to a specific PCI slot, and the SYCL device selector
(`ONEAPI_DEVICE_SELECTOR=level_zero:N`) is set per-model. Add your second card,
re-run `arc-llama init --force` to refresh `[[gpus]]`, then add models against
either GPU.

The default swap policy is **single-resident across all GPUs** , pick a model,
the router stops anything else first. Flip `server.single_resident = false` in
the config if you want different-GPU models to coexist.

## Upstreams

arc-llama can merge models from other OpenAI-compatible endpoints (e.g. Ollama,
vLLM, or another arc-llama instance) into its own model list and proxy requests
to them transparently:

```bash
# Add an upstream
arc-llama upstream add ollama http://127.0.0.1:11434

# List upstreams
arc-llama upstream list

# Remove
arc-llama upstream remove ollama
```

Upstream models appear in `/v1/models` with `owned_by: "upstream:NAME"` and are
routed directly to the upstream endpoint — no local llama-server is started.
The model list is cached for 30 seconds and refreshed on demand.

## Configuration reference

On Linux the config lives at `$XDG_CONFIG_HOME/arc-llama/config.toml` (usually
`~/.config/arc-llama/config.toml`). On Windows it lives at
`%APPDATA%\arc-llama\config.toml`.

```toml
version = 1

[server]
host = "127.0.0.1"
port = 11437
single_resident = true

[paths]
llama_server = "/usr/local/bin/llama-server"   # Windows: "C:\\...\\llama-server.exe"
models_dir   = "~/.local/share/arc-llama/models" # Windows: "%LOCALAPPDATA%\\arc-llama\\models"
state_dir    = "~/.local/state/arc-llama"        # Windows: "%LOCALAPPDATA%\\arc-llama"

[tune]
auto         = true      # idle-time background sweeps
idle_seconds = 120

[[gpus]]
pci_slot   = "0000:03:00.0"   # Windows: PNPDeviceID such as "PCI\\VEN_8086&DEV_E211&..."
sycl_index = 0
arch       = "battlemage"
backend    = "sycl"          # or "vulkan" for a Vulkan llama-server build
vram_mb    = 24480
enabled    = true
name       = "Arc Pro B60"

[[models]]
name             = "qwen3-7b"
display_name     = "Qwen 3 7B"
path             = "/home/me/models/qwen3-7b-q4_k_m.gguf"   # Windows: use double backslashes or forward slashes
gpu_pci_slot     = "0000:03:00.0"
port             = 18080
kv_class         = "default"
aliases          = ["qwen3-7b-q4_k_m.gguf"]

[models.recipe]
ctx              = 32768
cache_type_k     = "q8_0"
cache_type_v     = "q8_0"
n_gpu_layers     = 999
parallel         = 1
extra_flags      = []

[[upstreams]]
name = "ollama"
url  = "http://127.0.0.1:11434"
```

> [!NOTE]
> The optional agent/coding-assistant mode is experimental. Enable it by setting
> `ARC_LLAMA_EXPERIMENTAL_AGENT=1` before running `arc-llama agent`, `code`,
> or `agent-tui`.

`kv_class` controls the KV-cache size estimate that `arc-llama add` uses to
pick a context length. Currently:

| value             | per-token f16 KV | typical for                                  |
|-------------------|------------------|----------------------------------------------|
| `default`         | ~80 KiB          | most ≤30B dense models, conservative ceiling |
| `qwen3_27b_dense` | ~70 KiB          | Qwen 3 27B dense                             |
| `moe_a3b`         | ~24 KiB          | Qwen 3 30B/35B-A3B MoE                       |
| `gemma_swa`       | ~16 KiB          | Gemma 3/4 (interleaved sliding-window attn)  |

`arc-llama add` sizes the context length to the detected GPU's VRAM and the
model's file size. You can cap the auto-suggested value with the environment
variable `ARC_LLAMA_MAX_CTX` (e.g. `ARC_LLAMA_MAX_CTX=8192`), or override per
model with `--ctx N`. Use `--ubatch-size N` and `--batch-size N` to override
the prompt-processing batch defaults on cards where the auto-selected values
do not fit your workload.

## Architecture

```
┌──────────────────────┐
│  OpenAI client       │  Open WebUI, OpenCode, curl, ...
│  (port 11437)        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  arc-llama serve     │  FastAPI, /v1/chat/completions etc.
│   (router + state)   │
└──────────┬───────────┘
           │ ensure_active(model)
           ▼
┌──────────────────────┐
│  Router              │  swaps llama-server subprocesses per request
│  (single/multi-res)  │  applies arch SYCL env, picks safe ctx/KV
└──────────┬───────────┘
           │ subprocess.Popen
           ▼
┌──────────────────────┐
│  llama-server (SYCL) │  one per registered model, on demand
│  bound to GPU N      │
└──────────────────────┘
```

The router serialises swaps with an `asyncio.Lock`, so concurrent requests for
the same model fan out to one warm backend. Health is polled at
`{backend_url}/health`; cold-start budget is 120 s by default to absorb the
SYCL JIT recompile that plain `llama.cpp` pays on each fresh launch.

## Why not just use Ollama / vLLM?

- **Ollama (IPEX-LLM bundle):** the Intel-supported port has reproducible
  inference bugs on Battlemage with Qwen2.5-class models , sequential calls
  collapse to NaN-derived gibberish. arc-llama runs `llama-server` directly so
  you avoid that path entirely.
- **vLLM-XPU:** still maturing on Arc; weaker quant support. Worth trying for
  dense >30B if you want throughput, but not yet a one-command experience.
- **Plain `llama-server` + scripts:** what most Arc owners do today. arc-llama
  is the formalisation of those scripts, with the gotchas baked in.

## UIs

Two front-ends are bundled and both talk to the same admin endpoints
(`/admin/status`, `/admin/load/{name}`, `/admin/stop/{name}`, `/admin/stop-all`):

- **Web UI** at `http://<host>:<port>/` (default `127.0.0.1:11437`). Single
  static page polled every 5 s. Status, GPUs, model list, per-model
  Load/Stop buttons, "Stop all" panic button. No build step, no JS deps.
- **Terminal UI** via `arc-llama tui` , Textual-based. Bindings: `r` refresh,
  `l` load selected model, `s` stop selected, `S` stop all, `q` quit. Run it
  alongside `arc-llama serve` (or against a remote one with `--server`).

Both use brightness/dim for status (loaded vs idle) , no red/green palettes.

## Container

A Dockerfile is included that builds llama-server with the SYCL backend
(FP16 math path on by default) and installs arc-llama in a single image:

```bash
# Build (generic: JIT-compiled device code, works on any Intel GPU)
docker build -t arc-llama:latest .

# Build with AOT device code for your GPU generation — kills the ~20s SYCL
# JIT recompile every cold start pays (Battlemage can't use the JIT cache):
docker build --build-arg GGML_SYCL_DEVICE_ARCH=bmg-g21 -t arc-llama:bmg .  # B-series
docker build --build-arg GGML_SYCL_DEVICE_ARCH=acm-g10 -t arc-llama:acm .  # A770/750/580

# Run (GPU access required)
docker run --rm -it \
  --device /dev/dri:/dev/dri \
  --group-add video --group-add render \
  -p 11437:11437 \
  -v $HOME/models:/models:ro \
  arc-llama:latest
```

The entrypoint auto-runs `arc-llama init` on first launch if no config exists,
then starts `arc-llama serve`. Mount your own `config.toml` for full control:

```bash
docker run ... \
  -v $PWD/config.toml:/root/.config/arc-llama/config.toml:ro \
  arc-llama:latest
```

## Open WebUI

[Open WebUI](https://github.com/open-webui/open-webui) is a self-hosted ChatGPT-style interface. arc-llama speaks the OpenAI API, so they wire together directly.

### One command with docker compose

```bash
# GGUF models from $HOME/models; Open WebUI at http://localhost:3000
docker compose up

# Or point at a different model directory
MODELS_DIR=/mnt/data/models docker compose up
```

The first time you open `http://localhost:3000` you'll create a local admin account. arc-llama's models appear automatically in the model picker.

### Manual (bare-metal `arc-llama serve`)

1. Run `arc-llama serve` (listening on `127.0.0.1:11437` by default)
2. In Open WebUI → **Settings → Admin Panel → Connections**, add an OpenAI connection:
   - **Base URL:** `http://127.0.0.1:11437/v1`
   - **API Key:** any non-empty string (arc-llama does not validate keys)
3. Models appear immediately in the chat model picker.

If Open WebUI and arc-llama are on different machines, replace `127.0.0.1` with the host IP and run `arc-llama serve --host 0.0.0.0` (or set `ARC_LLAMA_HOST=0.0.0.0`).

## LMStudio

LMStudio's local server (port 1234 by default) exposes an OpenAI-compatible API. Add it as an arc-llama upstream and its models appear alongside your local Arc models in a single endpoint — useful for running a second model on a CPU or a different GPU while your Arc card handles local GGUF inference.

```bash
# LMStudio running on the same machine
arc-llama upstream add lmstudio http://127.0.0.1:1234

# LMStudio on the host from inside a Docker container
docker compose exec arc-llama arc-llama upstream add lmstudio http://host.docker.internal:1234
```

LMStudio models appear in `/v1/models` with `owned_by: "upstream:lmstudio"` and requests are proxied transparently — no local llama-server is started for them.

To point Open WebUI (or any other client) at arc-llama only, and let it discover both local Arc models and LMStudio models through the same `/v1` endpoint, no extra configuration is needed — the model list is already merged.

## Roadmap

- ~~HF model download (`arc-llama add org/repo:quant --from-hf`).~~ ✅
- ~~Streaming response forwarding (`stream: true`).~~ ✅
- ~~Container image with `llama-server` + arc-llama prebuilt.~~ ✅
- ~~`arc-llama benchmark` , quick prompt-eval/gen tok/s harness.~~ ✅
- ~~`arc-llama tune` , measure-and-persist recipe autotuner.~~ ✅
- ~~`arc-llama install-runtime` , download a prebuilt llama-server (Vulkan-first).~~ ✅
- ~~`arc-llama tune --all` , sweep every registered model in one run.~~ ✅
- ~~Background auto-tune on first use, aborting on new requests.~~ ✅

## Contributing

PRs and issues welcome. The most useful contributions today are:

1. Confirming or fixing PCI device-ID → arch mappings for your card. If
   `arc-llama gpus` shows `unknown` for a working Arc card, please open an
   issue with `lspci -nn` output.
2. Reporting architectures where the default SYCL env profile crashes or
   underperforms.
3. Trying the smoke tests on hardware other than the maintainer's Battlemage
   B60 development box.

## Support

This project is free and I don't ask for anything. If it's useful to you,
a star on the repo is appreciated, and if you want to follow along with
other things I'm building, you can find them under
[@offbyonebit](https://github.com/offbyonebit).

If you'd like to support development, you can [sponsor me on GitHub](https://github.com/sponsors/offbyonebit).

## License

MIT , see [LICENSE](LICENSE).
