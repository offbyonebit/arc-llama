# Incident: Flux2TEModel_ segfault in torch.nn.functional.embedding on Intel XPU (2026-09-14)

Status: root cause identified, GPU fix verified, defensive code added.
Scope: diagnosis document. The fix lives in the ComfyUI launcher
(`/mnt/storage/comfyui-xpu/start.sh`, outside the repo) plus defensive
adapter improvements in `arc-llama-vision/` (this repo).

## Summary

A 512x512 FLUX.2 Klein render through the vision companion's `comfyui`
adapter was reported as a segmentation fault inside
`torch.nn.functional.embedding` while `Flux2TEModel_` encoded the first
prompt of the Qwen3-8B Q2_K GGUF text encoder on the Intel Arc Pro B60
(XPU). The model, the workflow, and the GGUF weights are not at fault. The
crash was caused by a **mixed SYCL/Unified-Runtime (UR) stack inside the
ComfyUI process**, produced by launching ComfyUI from a login shell whose
environment pointed at the system oneAPI 2026.1 installation.

The observed failure is a process-killing `SIGSEGV`; the vision companion
then polled a dead prompt until timeout, surfacing a generic 503. The
identical workflow with the identical GGUF renders in ~22s on the same GPU
when the process binds the correct runtime.

## Evidence

Timeline (all 2026-09-14, from `user/comfyui.log`, `journalctl`, and
`coredumpctl`):

- `10:39:12` ComfyUI started (v0.20.1, torch 2.11.0+xpu, device `xpu:0
  Intel(R) Arc(TM) Pro B60 Graphics`).
- `10:40:03` prompt received: the companion's Flux.2 Klein GGUF workflow
  (`UnetLoaderGGUF` + `CLIPLoaderGGUF type=flux2` + VAE).
- `10:40:09` text encoder GGUF loaded: `gguf qtypes: Q6_K (1), F32 (145),
  Q2_K (145), Q3_K (72), Q4_K (36)`; `Dequantizing token_embd.weight`.
- `10:40:15` `Requested to load Flux2TEModel_` -> `loaded completely;
  5303.28 MB loaded`. Log ends here, mid-encode.
- `10:40:21` systemd-coredump: process 82326 (python) `SIGSEGV`. The
  vision plugin logged `POST /plugins/vision/generate 503` to arc-llama
  (journal), and the companion's `httpx` call returned 503.

Core dump analysis (`coredumpctl gdb 82326`) gives the decisive stack:

```
#0  __strlen_avx2
#1  ur::level_zero::urProgramBuildExp(...)
    from /mnt/storage/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0
#2  urProgramBuildExp (ur_tracing_layer) from .../2026.1/lib/libur_loader.so.0
#3  urProgramBuildExp (libur_loader)
#4  sycl::detail::ProgramManager::build(...) from .../torch/lib/../../../../libsycl.so.8
#5-#17  ProgramManager::getBuiltURProgram -> enqueueImpKernel -> handler::finalize
#18 at::native::xpu::index_select_kernel(...) from .../torch/lib/libtorch_xpu.so
#19-#22 index_select xpu dispatch chain
#24 at::native::embedding_symint(at::Tensor const&, at::Tensor const&, ...)
#27 torch::autograd::THPVariable_embedding  <- torch.nn.functional.embedding
```

Registers at fault: `rdi=0xffffffff` passed to `strlen`, in other words the
UR adapter dereferenced an options pointer that is garbage under its
struct-layout reading. The crash happens during the **JIT build of the
`index_select` device kernel**, which is the kernel `F.embedding` dispatches
on XPU. This runs on the **first text-encode** (token-embedding lookup),
which is why the fresh load completed and the process died at the first
prompt: no other kernel build path had been exercised before
`embed_tokens`.

Library-version proof (the crash precondition):

- The ComfyUI venv ships the matched stack in `venv/lib`:
  `libsycl.so.8.0.0` built with **DPC++ 2025.3.2** (string inside the
  binary), plus `libur_loader.so.0.12.0` and level-zero UR adapters from
  the same release.
- The coredump's mapped libraries show the process actually bound:
  `libsycl.so.8` from the venv (torch RPATH) **but** `libur_loader.so.0`
  and both `libur_adapter_level_zero*.so.0` from **oneAPI 2026.1**
  (`/mnt/storage/opt/intel/oneapi/compiler/2026.1/lib`, itself symlinked
  from `/opt/intel/oneapi`).
- `md5sum` confirms the two trees ship different binaries for the same
  sonames (e.g. `libur_loader.so.0`: `c8d7…` in venv vs `ca86…` in 2026.1).
  Both export `urProgramBuildExp@@LIBUR_LOADER_0.12`, so the loader binds
  silently with no version check.
- Why the wrong tree won: `/home/slowe/.bashrc` sources
  `setvars.sh --force` for every login shell (for the llama.cpp SYCL
  stack), exporting the 2026.1 `LD_LIBRARY_PATH`. `libsycl.so.8`'s
  RUNPATH is only `$ORIGIN`, and transitive resolution of
  `libur_loader.so.0` consults `LD_LIBRARY_PATH` **before** the venv's
  RPATH. Verified with `ldd` under the polluted environment: the venv
  `libsycl.so.8` binds the 2026.1 loader and, through the loader's
  `$ORIGIN/../lib` RUNPATH, the 2026.1 adapters.
- Counter-proof that the model/env is fine when the runtime is not mixed:
  after `start.sh` was corrected (see below), the same workflow on the
  same GGUF rendered successfully in **21.58 s** ("Prompt executed in 21.58
  seconds", `user/comfyui_8188.log`), with the identical
  `Dequantizing token_embd.weight` + `Flux2TEModel_` load sequence.

## Root cause

Torch XPU wheels bundle their own SYCL runtime and Unified Runtime, which
must be used as a matched set. Launching the ComfyUI process with a
system-wide oneAPI `LD_LIBRARY_PATH` (from a login shell's `setvars.sh`)
silently substitutes a differently-versioned UR loader and adapters under
the same soname. The first JIT-built device kernel (`index_select` for
`F.embedding`) then passes build options through a struct layout the
mismatched adapter misreads, dereferencing garbage.

This is an external runtime environment issue, not a defect in
arc-llama, arc-llama-vision, ComfyUI, the GGUF conversion, or the
weights.

## Fix (GPU-side, no CPU fallback)

`/mnt/storage/comfyui-xpu/start.sh` now sanitizes the launch environment
before exec'ing the venv python: it unsets `LD_LIBRARY_PATH`, `LIBRARY_PATH`,
`CPATH`, `CPLUS_INCLUDE_PATH`, `C_INCLUDE_PATH`, `PKG_CONFIG_PATH`, and
`CMAKE_PREFIX_PATH`, keeping only device-selection variables
(`ONEAPI_DEVICE_SELECTOR`, `ZES_ENABLE_SYSMAN`, `SYCL_CACHE_PERSISTENT=0`).
The torch wheel then resolves its bundled, matched SYCL/UR/MKL stack through
its own RPATH. The same GGUF workflow then renders full-speed on the XPU
(21.58 s at 512x512, 20 steps).

Verification commands (no GPU contact needed for the binding check):

```
# poisoned shell shows the mismatch:
LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH) \
  ldd /mnt/storage/comfyui-xpu/venv/lib/libsycl.so.8 | grep ur_loader
#  -> libur_loader.so.0 => /mnt/.../oneapi/compiler/2026.1/lib/libur_loader.so.0

# sanitized environment pins the matched stack (what the fixed start.sh does):
env -u LD_LIBRARY_PATH \
  ldd /mnt/storage/comfyui-xpu/venv/lib/libsycl.so.8 | grep ur_loader
#  -> libur_loader.so.0 => /mnt/storage/comfyui-xpu/venv/lib/libur_loader.so.0
```

Minimal reproduction (as observed, no code required):

1. Log into the host (bashrc sources oneAPI 2026.1 `setvars.sh`).
2. Run the old `start.sh` (no env handling) and submit the companion's
   Flux.2 Klein GGUF workflow.
3. The server dies on the first prompt's text encode; systemd-coredump
   records SIGSEGV with the stack above.

Mitigation for future launches:

1. Always launch ComfyUI through the fixed `start.sh` (or any environment
   that drops the oneAPI `LD_LIBRARY_PATH`).
2. `arc_llama_vision.xpu_runtime.sanitize_launch_env()` builds exactly
   this sanitized environment for any tooling that spawns the GPU
   process from Python.
3. `arc_llama_vision.xpu_runtime.detect_ur_runtime_mixing(pid)` reads
   `/proc/<pid>/maps` of a running GPU process and reports the mixed
   binding with a human explanation, so the condition can be caught
   before the first crash instead of debugging a core afterward.

## Defensive changes in this repo (arc-llama-vision)

- `src/arc_llama_vision/xpu_runtime.py` (new): stdlib-only
  `detect_ur_runtime_mixing()`, `sanitize_launch_env()`, and
  `describe_binding()` as above.
- `src/arc_llama_vision/comfyui.py`: the poll loop now distinguishes a
  crashed backend from a slow one. When a submitted prompt is absent from
  `GET /queue` (both `queue_running` and `queue_pending`) twice in a row
  and has no `GET /history/{id}` entry, the adapter raises
  `BackendUnavailableError` immediately ("ComfyUI lost the render: ...")
  instead of stalling for `poll_timeout` (20 min default) against a dead
  server. A malformed or unknown `/queue` shape never triggers the
  lost-prompt path (history remains the completion authority), and a
  queued-but-never-finishing prompt still takes the normal timeout path.

Tests: `arc-llama-vision/tests/test_xpu_runtime.py` (new) covers the
mixed/clean/unavailable classifications with the real production paths,
the environment sanitizer, and stdlib-only import hygiene;
`tests/test_comfyui_adapter.py` gains lost-prompt tests (crash fail-fast,
two-confirmation requirement, malformed-queue tolerance, and the queue
fixture for all fake servers).

## Not a defect / do not chase

- The GGUF text encoder and its Q2_K quants: loads and encodes fine on
  the matched runtime.
- `token_embd.weight` dequantization at load (ComfyUI-GGUF's OOM
  workaround): by design for vocab>=64k.
- ComfyUI's XPU kernel selection: `index_select` is the correct kernel
  for `F.embedding` on XPU.
- The vision companion's workflow graph: byte-identical graph worked
  post-fix.