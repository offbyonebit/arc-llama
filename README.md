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
- **Speech on the same port** , `/v1/audio/transcriptions` backed by
  `llama-server` (Qwen3-ASR, SYCL included) and `/v1/audio/speech` backed by a
  pluggable TTS engine ([OmniVoice](https://github.com/k2-fsa/OmniVoice) today,
  with voice cloning). Speech backends stay pinned in VRAM so a voice command
  never cold-starts your LLM. See [Speech](#speech).
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

`arc-llama init` registers every Intel GPU it finds. Each model in the config
is bound to a specific PCI slot, and the SYCL device selector
(`ONEAPI_DEVICE_SELECTOR=level_zero:N`) is set per-model. Add your second card,
re-run `arc-llama init --force` to refresh `[[gpus]]`, then add models against
either GPU.

The default swap policy is **single-resident across all GPUs** , pick a model,
the router stops anything else first. Flip `server.single_resident = false` in
the config if you want different-GPU models to coexist.

## Speech

arc-llama serves both audio endpoints alongside your LLM, so one port covers
the chat, the STT and the TTS needs of something like Home Assistant:

| | Endpoint | Engine | Runs on |
|---|---|---|---|
| Speech to text | `/v1/audio/transcriptions` | `llamacpp` | your existing `llama-server`, **SYCL** included |
| Text to speech | `/v1/audio/speech` | `omnivoice` (pluggable) | a Python sidecar under its own interpreter |

Both directions share the same registry, ports, GPU binding, VRAM accounting
and load/evict lifecycle as your chat models — an audio model is just an entry
with a `task`.

### Speech to text

```bash
# Downloads the weights *and* the audio projector, then registers both
arc-llama audio add ggml-org/Qwen3-ASR-0.6B-GGUF:Q8_0 --from-hf \
  --name qwen3-asr --alias whisper-1

# Or point at files you already have (the mmproj is auto-detected if it
# sits beside the weights as mmproj-<name>.gguf)
arc-llama audio add ~/models/Qwen3-ASR-0.6B-Q8_0.gguf --name qwen3-asr

arc-llama audio list
```

Transcription always runs on `llama-server`. It is the binary you already
have, it is the only ASR runtime with a SYCL build, and it inherits the arch
env profiles and device selector the LLM path uses.

llama.cpp keeps the audio encoder in a separate `mmproj-*.gguf`. It is
required, not optional: without it `llama-server` loads the model as a plain
text LLM and transcription returns fluent nonsense rather than failing, so
arc-llama refuses to register a transcription model without one. `--from-hf`
picks the projector matching your weights' quant — a bf16 projector next to
q8_0 weights works but costs about twice the VRAM for no accuracy you asked
for.

> [!IMPORTANT]
> **arc-llama pins `-c` for ASR models, and you want it to.** `llama-server`
> defaults to `-c 0`, meaning "whatever the GGUF was trained for", and
> Qwen3-ASR advertises 65536 — that allocates several GB of KV cache in front
> of under 2 GB of weights, so a 1.7B transcription model ends up occupying
> more VRAM than the 27B it sits next to. The default here is 4096, which
> covers minutes of speech. Raise it in the recipe for long-form dictation:
>
> ```toml
> [audio_models.recipe]
> ctx = 8192
> cache_type_k = "q8_0"
> cache_type_v = "q8_0"
> ```

> [!NOTE]
> Qwen3-ASR emits its transcripts as `language English<asr_text>the actual
> words`, and llama.cpp forwards that verbatim
> ([#26749](https://github.com/ggml-org/llama.cpp/issues/26749)) — which a
> Home Assistant voice pipeline then tries to match as part of your command.
> arc-llama strips the framing by default; pass `--no-strip-markers` to keep
> the raw output. Streamed responses (`stream=true`) are forwarded raw.

Transcribe with either the multipart upload (what Home Assistant, Open WebUI
and the OpenAI SDKs send) or the JSON form:

```bash
curl http://127.0.0.1:11437/v1/audio/transcriptions \
  -F model=qwen3-asr \
  -F file=@speech.wav
```

Add `-F stream=true` for incremental deltas; that needs a model registered
with `--mode streaming`.

### Text to speech

TTS is served by a **pluggable engine**. [OmniVoice](https://github.com/k2-fsa/OmniVoice)
ships today — zero-shot voice cloning across 600+ languages — and
`arc-llama audio engines` lists what your build has.

OmniVoice is a Python library, not a binary, and it pulls in torch and
transformers. arc-llama depends on neither and does not want to, so the model
runs in a small sidecar process under **its own interpreter**. Two ways to set
that up:

```bash
# A. Separate environment (recommended): keeps torch out of arc-llama's venv
python -m venv ~/venvs/omnivoice
~/venvs/omnivoice/bin/pip install omnivoice \
  --extra-index-url https://download.pytorch.org/whl/xpu
arc-llama audio set-python ~/venvs/omnivoice/bin/python

# B. One environment: install the extra, and skip set-python entirely
pip install "arc-llama[tts]" \
  --extra-index-url https://download.pytorch.org/whl/xpu
```

> [!IMPORTANT]
> **Pass the XPU index.** PyTorch distinguishes its accelerator builds only by
> a local version tag (`2.11.0+xpu`) and by which index serves them, never by
> package name — so a plain `pip install torch` on an Arc box pulls the CUDA
> build: several GB of NVIDIA runtime that can never touch your GPU. Working
> in this repo instead? `uv sync --extra tts` already routes torch, torchao
> and Triton to Intel's index via `tool.uv.sources`.

Then register the model. The Hugging Face repo id is fine — the engine
resolves and downloads it itself:

```bash
arc-llama audio add k2-fsa/OmniVoice --task tts --name omnivoice --alias tts-1

# Optional: fewer diffusion steps trades a little quality for latency
arc-llama audio add k2-fsa/OmniVoice --task tts --option num_step=16
```

#### Voices

The OpenAI `voice` field resolves against a registry you control. A voice is
either **cloned** from a reference clip or **designed** from attributes:

```bash
# Clone: 3–10 s of clean speech. Supplying --ref-text is worth the typing —
# without it the backend loads Whisper to transcribe the clip on first use.
arc-llama audio voice add glados \
  --ref-audio ~/voices/glados.wav \
  --ref-text "All right, look. We've both said a lot of things." \
  --alias alloy

# Design: no reference audio at all
arc-llama audio voice add narrator --instruct "male, low pitch, british accent"

arc-llama audio voice list
```

**A fine-tuned model needs no voice at all.** If you trained the speaker into
the weights, it is already the model's voice — register nothing and any `voice`
a client sends is ignored, which is what you want, since OpenAI clients are
obliged to send one. Give it a name only if you want `voice: "glados"` to be
explicit alongside other voices:

```bash
arc-llama audio voice add glados --auto
```

Do *not* give a fine-tune a clone or design voice as its `default_voice`: that
layers a prompt on top of weights that already encode the speaker, and the
prompt wins.

`--alias` is what makes hardcoded clients work: something that always sends
OpenAI's `alloy` gets your GLaDOS. An unrecognised voice falls back to the
model's `default_voice` rather than erroring, because a substituted voice is a
far better failure for a speech client than none at all. Adding a voice takes
effect on the next request — no model reload.

```bash
curl http://127.0.0.1:11437/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "omnivoice", "input": "Oh. It'\''s you.", "voice": "glados"}' \
  --output hello.mp3
```

`input`, `voice`, `response_format` (`mp3`/`opus`/`aac`/`flac`/`wav`/`pcm`),
`speed` and `instructions` all behave as OpenAI documents them. `wav` and
`pcm` are written from the standard library and always work; the compressed
formats go through libsndfile and fall back to `ffmpeg`, so install ffmpeg if
you want `mp3` (the OpenAI default) to be available.

#### Quantized models

A torchao-quantized checkpoint is picked up automatically — point `--task tts`
at the directory and arc-llama does the rest:

```bash
arc-llama audio add ~/models/OmniVoice_INT8 --task tts --name glados-tts
```

Detection is by the presence of `quantized_state.pt`, because that file is
what makes the directory unloadable any other way. `torchao`'s `quantize_()`
replaces Linear weights with tensor subclasses that `save_pretrained` cannot
serialise, so quantization scripts emit a `torch.save` state dict beside an
otherwise normal model directory — config, tokenizer, `audio_tokenizer/`, but
no `model.safetensors`. `from_pretrained` alone therefore has nothing to load.
arc-llama instead materialises the base model, re-applies `quantize_()` to
`llm` and `audio_heads` to recreate the same module shapes, and reads the
saved tensors into that.

The base defaults to `k2-fsa/OmniVoice`; override it if you quantized from a
fine-tune, and `dtype` defaults to `bfloat16` for a quantized checkpoint
because that is what the tensors carry:

```bash
arc-llama audio add ~/models/OmniVoice_INT8 --task tts \
  --option base_model=me/OmniVoice-glados \
  --option compile=true
```

> [!IMPORTANT]
> If the weights are missing, arc-llama **refuses to start** rather than
> warning and continuing on the base model. Falling back there does not
> produce an obvious failure — it produces fluent audio in the wrong voice,
> which is much harder to notice than a backend that will not come up. A
> checkpoint that shares no parameter names with the model is rejected for the
> same reason: `load_state_dict(strict=False)` is required here (the audio
> tokenizer is not in the file) and would otherwise report success having
> loaded nothing.

#### Adding an engine

An engine is one class: `build_plan` says how to launch a backend and
`build_payload` maps an OpenAI speech body onto it. Register it in
`arc_llama/tts/` and everything else — the router lifecycle, eviction, the
VRAM fit guard, the endpoint, in-flight accounting — works unchanged, because
nothing outside that package dispatches on an engine name. A `llama-tts`
engine would be a natural next one.

### Both directions

**Audio models are pinned by default**, which cuts both ways: they are exempt
from the single-resident swap policy, *and* loading one evicts nothing. A small
speech model displacing your LLM would make every voice command pay a full cold
start on the next reply, which is the whole thing pinning exists to avoid — so
a pinned model is a declared co-resident in both directions.

Their footprint is still charged against the GPU's VRAM budget, so nothing
overcommits silently: if the speech model genuinely does not fit alongside
what is loaded, the load is refused with the options spelled out rather than
resolved by evicting. Pass `--swappable` at registration to opt back into
swapping — then it evicts, and can be evicted, like any other model.

There is deliberately no equivalent flag for LLMs. Pinning those too would
disable single-residency altogether; the intended shape is a small speech model
resident beside whichever LLM is current.

Either way each audio model gets its own backend subprocess on its own port,
so `load`, `stop`, drain and the health gate behave exactly as they do for a
chat model. Run `arc-llama doctor` to check an engine's prerequisites — for a
Python engine it verifies the interpreter can actually import it.

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
# Optional: interpreter for a Python TTS backend (OmniVoice). Leave empty when
# arc-llama's own environment has it, e.g. after `pip install arc-llama[tts]`.
tts_python   = "~/venvs/omnivoice/bin/python"

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

[[audio_models]]
name              = "qwen3-asr"
display_name      = "Qwen3 ASR 0.6B"
path              = "/home/me/models/Qwen3-ASR-0.6B-Q8_0.gguf"
gpu_pci_slot      = "0000:03:00.0"
port              = 18090
engine            = "llamacpp"   # asr: always llamacpp. tts: a TTS engine
                                 # name, e.g. "omnivoice"
task              = "asr"        # asr | tts
mode              = "offline"    # offline | streaming
aliases           = ["whisper-1"]
always_resident   = true         # exempt from single-resident eviction
strip_asr_markers = true         # drop Qwen3-ASR's "language X<asr_text>" framing
# vram_mb         = 1200         # declared footprint for the fit guard

# Launch knobs live here, exactly as they do for [models.recipe].
[audio_models.recipe]
mmproj            = "/home/me/models/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"  # asr: required
ctx               = 4096         # asr: -c. Never left to llama.cpp's
                                 # default of 0 ("use the GGUF's 65536").
cache_type_k      = "f16"
cache_type_v      = "f16"
n_gpu_layers      = 999
extra_flags       = []

[[audio_models]]
name              = "omnivoice"
path              = "k2-fsa/OmniVoice"   # repo id; the engine resolves it
gpu_pci_slot      = "0000:03:00.0"
port              = 18091
engine            = "omnivoice"
task              = "tts"
aliases           = ["tts-1"]

[audio_models.recipe]
# python                  = "~/venvs/omnivoice/bin/python"  # overrides paths.tts_python
device                  = "xpu"       # torch device; default xpu on a SYCL GPU
dtype                   = "float16"
default_voice           = "glados"    # used when `voice` matches nothing
default_language        = "English"
default_response_format = "mp3"       # used when the request omits it
# Engine-specific knobs, passed through untouched. Anything only one engine
# understands lives here rather than becoming a config field.
options                 = { num_step = 16 }

# Voices are top level, not per model: the same reference clip should still
# name the same voice after switching engines.
[[voices]]
name       = "glados"
ref_audio  = "/home/me/voices/glados.wav"
ref_text   = "All right, look. We've both said a lot of things."
language   = "English"
aliases    = ["alloy"]       # so a client hardcoding OpenAI's id works
# instruct = "female, low pitch"   # design a voice instead of cloning one
# models   = ["omnivoice"]         # empty = usable by every TTS model

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
