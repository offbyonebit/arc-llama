# GPU setup

Two things trip up first runs on Intel Arc under Linux: missing device
permissions, and a slow first request when using the SYCL backend. Neither is
a bug in Arc Llama. This page explains both and how to fix them.

Run `arc-llama doctor` at any point. It reports everything below.

## Device permissions (Linux)

Intel GPUs are exposed as device nodes under `/dev/dri/`. Your user needs to
be in the `render` and `video` groups to open them. If it is not, `doctor`
prints:

```text
user groups:
    render         warn
    video          warn
    → add yourself with `sudo usermod -aG render,video $USER` and re-login.
```

### Fix

```bash
sudo usermod -aG render,video $USER
```

Group changes apply to new login sessions only. Log out and back in (or
reboot), then confirm:

```bash
id -nG | tr ' ' '\n' | grep -E '^(render|video)$'
arc-llama doctor
```

Both groups should print and `doctor` should no longer warn. To try it in one
shell without logging out, `newgrp render` starts a subshell with the new
group, but a full re-login is the reliable fix.

### Services and containers

- A systemd **user** unit runs as you, so it inherits your groups after you
  log in again.
- A systemd **system** unit needs `SupplementaryGroups=render video` in its
  `[Service]` section.
- Docker needs the device and both groups, as in the README:
  `--device /dev/dri:/dev/dri --group-add video --group-add render`.

### Windows

Group checks do not apply. `doctor` prints that they are unavailable. Install
the current Intel Arc graphics driver instead.

## First request is slow on SYCL (about 20 seconds)

With the SYCL backend, the first inference after each cold start can take
roughly **20 seconds** longer than later requests. This is expected, not a
hang.

SYCL `llama-server` builds compile GPU kernels at runtime ("JIT"). Normally
the compiled kernels are cached on disk, but on Battlemage Arc Llama sets
`SYCL_CACHE_PERSISTENT=0`. With the cache enabled, `llama-server` crashed
with a `SIGSEGV` in `PersistentDeviceCodeCache::getItemFromDisc` using
`libsycl.so.9` from oneAPI 2026.0. So every fresh launch pays the compile
again. A model that stays loaded pays it once.

Arc Llama's cold-start budget is 120 seconds by default, so a 20 second warm-up
does not trigger a timeout. `arc-llama benchmark` reports it separately as
`Warm-up ... (SYCL JIT)` so it does not skew your tokens-per-second numbers.

### Remove the delay with an AOT build

Build `llama-server` with ahead-of-time (AOT) device code for your GPU. The
kernels are then compiled at build time and there is nothing to compile at
launch. Pass your GPU's `ocloc` device string to CMake:

| GPU | `GGML_SYCL_DEVICE_ARCH` |
| --- | --- |
| Battlemage: B570, B580, Pro B60 | `bmg-g21` |
| Alchemist: A770, A750, A580, Pro A60 | `acm-g10` |
| Alchemist: A380, A310 | `acm-g11` |

```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_DEVICE_ARCH=bmg-g21 \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --config Release -j
```

Use the string from the table that matches your card. For several GPU
generations in one binary, pass a comma-separated list, for example
`-DGGML_SYCL_DEVICE_ARCH="acm-g10,bmg-g21"`.

With Docker, pass it as a build argument:

```bash
docker build --build-arg GGML_SYCL_DEVICE_ARCH=bmg-g21 -t arc-llama:bmg .
```

Then point Arc Llama at the new binary in `config.toml`:

```toml
[paths]
llama_server = "/path/to/your/build/bin/llama-server"
```

`arc-llama doctor` shows which `llama-server` it found. If you are unsure
which card you have, `arc-llama gpus` lists each GPU's name and architecture.
The table above matches the mapping Arc Llama uses internally
(`AOT_ARCH_BY_DEVICE_ID` in `arch.py`).

### Or use Vulkan

The portable Vulkan runtime that `arc-llama setup` installs does not use SYCL,
so this SYCL warm-up does not apply to it. Vulkan is the default because it
needs no oneAPI install and no build. SYCL is the option to reach for when
your own benchmarks show it is faster on your model. Compare them with
`arc-llama benchmark`.

## Also required: Resizable BAR

Both backends need Resizable BAR (ReBAR) enabled in the BIOS. Without it,
llama.cpp falls back to slow paths on Arc. `doctor` reports the ReBAR state.
