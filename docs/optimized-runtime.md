# Arc Llama Runtime (optional optimized runtime)

Arc Llama Runtime is a modified, optimized build of llama.cpp for Intel Arc B-series (Battlemage) GPUs, distributed by the
Arc Llama project as free, signed, closed binaries under the MIT license. On an Arc Pro B60 it decodes Gemma 4
26B-A4B up to ~2x faster and Ornith 1.5 35B-A3B (Qwen3.5-MoE) ~14% faster than stock llama.cpp, with output checked
against llama.cpp's CPU backend for every release.

It is optional. Stock llama.cpp remains the default, and nothing changes unless you install it.

## Requirements

- Linux x86-64
- An Intel Arc B-series (Battlemage) GPU
- The Intel oneAPI runtime (not bundled)
- OpenSSH's `ssh-keygen` (used to verify the release signature)

`arc-llama install-runtime --optimized` refuses, and says why, on any other host.

## Install

```bash
arc-llama install-runtime --optimized                # latest release
arc-llama install-runtime --optimized --runtime-version 1.0.0+b11624
arc-llama install-runtime --optimized --archive ./arc-llama-runtime-<version>-linux-x64-sycl-bmg.tar.gz
```

For `--archive`, keep the release's `SHA256SUMS` and `SHA256SUMS.sig` next to the archive.

Before anything is extracted, Arc Llama checks the Ed25519 signature on `SHA256SUMS` against the release key shipped
inside Arc Llama, then the archive's SHA-256. A missing or bad signature, a checksum mismatch, or an archive that is not
an Arc Llama Runtime leaves your existing runtime and configuration untouched.

Then restart `arc-llama serve`.

## Tuned settings

Each release carries per-model settings it was validated with, matched on the model's GGUF architecture (for example
the Qwen3.5-MoE and Gemma 4 MoE families). Arc Llama applies them only to models launched with that runtime. A model's
own environment overrides (Model settings, or `env = { ... }` in the config) always take precedence.

## Going back

```bash
arc-llama install-runtime --backend sycl     # stock llama.cpp
```

This reuses a stock runtime that is already installed. Per-model runtimes (`llama_server` per model) let you run Arc
Llama Runtime for some models and stock llama.cpp for others.

## Credits

A modified version of [llama.cpp](https://github.com/ggml-org/llama.cpp), Copyright (c) 2023-2026 The ggml authors,
MIT; the modifications are Copyright (c) 2026 offbyonebit, MIT. Each release ships llama.cpp's license and the notices of
the libraries it bundles. Not affiliated with or endorsed by the llama.cpp project, Intel or Meta.
