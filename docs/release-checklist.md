# Release-candidate checklist

Run this checklist from a clean candidate branch or worktree. Record the OS,
GPU, driver, backend, Python, Arc Llama version, llama.cpp build, model, and
result for every hardware run.

## Automated validation

```bash
ruff check src tests registry-repo/scripts
mypy --check-untyped-defs src
pytest tests/ -q -ra --tb=short -p no:cacheprovider
python -m build
```

Review the pytest skip report before calling a release green. Every skip must
be classified as platform-specific coverage, a deliberately absent local
fixture or model covered by an integration or hardware run, or an intentionally
unavailable optional integration. An unexplained skip, or an increase in the
skip count without a corresponding test change, blocks the release. Regular
CI reports skip reasons; it does not claim fixture-backed inference coverage
when the fixture is absent.

Install the wheel into a new environment without extras. Confirm `textual`,
`fastembed`, and `mcp` are absent, then run:

```bash
arc-llama --version
arc-llama --help
arc-llama run --help
arc-llama serve --help
arc-llama recipes --help
```

Open `/` and `/chat` with the network disconnected. Confirm the dashboard,
Markdown rendering, syntax highlighting, model selection, settings, streaming,
history export/import, and error states work without browser-console errors.

## Hardware matrix

Test Windows and Linux with at least one consumer Alchemist card and one
consumer Battlemage card. Test Vulkan on every host and SYCL wherever the Intel
oneAPI runtime is supported. On each combination:

1. Start from a new Python environment and no Arc Llama configuration.
2. Run `arc-llama doctor` and save the diagnostics.
3. Run `arc-llama run MODEL --setup-only`; verify GPU, backend, context, KV
   cache, model size, and estimated fit.
4. Start `arc-llama run MODEL`, request `/v1/models`, then exercise streaming
   and non-streaming chat completions.
5. Repeat with a local path containing spaces and with an explicit Hugging Face
   GGUF specification.
6. Interrupt a runtime/model download and verify the previous installation and
   configuration still work.
7. Feed an invalid checksum, unsafe archive, oversized registry, malformed
   registry, and incompatible speculative draft; verify each fails safely.
8. Run the opt-in smoke test with `ARC_LLAMA_SMOKE_MODEL` and retain the result.

## Release records

- Confirm the project version is newer than PyPI and matches the wheel/sdist.
- Ensure every user-visible change appears under `Unreleased`, then convert it
  to the final version and date.
- Inspect wheel and sdist contents for caches, virtual environments, secrets,
  local paths, and missing static/data files.
- Confirm Linux and Windows CI pass on Python 3.10, 3.12, and 3.14, and that
  the bare-wheel job passes on the oldest supported Python.
- Publish a release candidate first and hold the 0.9 stabilization period before
  declaring 1.0 compatibility.
