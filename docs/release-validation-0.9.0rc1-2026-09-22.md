# Arc Llama 0.9.0rc1 Linux validation

**Date:** 2026-09-22

**Branch:** `release/1.0-prep`

**Platform:** Linux

**Python:** 3.12 virtual environment (`.venv`)

The Linux checks completed successfully against the release-candidate branch:

| Check | Result |
| --- | --- |
| Ruff (`ruff check src tests registry-repo/scripts`) | Passed |
| Mypy (`mypy --check-untyped-defs src`) | Passed; 43 source files |
| JavaScript syntax checks for bundled UI scripts | Passed |
| Python tests (`pytest tests/ -q -ra --tb=short -p no:cacheprovider`) | **1014 passed, 13 skipped** |
| Browser regression suite (`node run.mjs` from `browser-tests/`) | **8/8 passed** |
| Package build (`uv build`) | Passed; wheel and sdist created |

The 13 skips were expected and documented by pytest: nine GGUF metadata tests
and three launcher tests had no local GGUF fixtures, and the inference smoke
test was skipped because `ARC_LLAMA_SMOKE_MODEL` was not set. The suite emitted
one Starlette deprecation warning recommending `httpx2`; it did not affect the
result.

Built artifacts:

- `arc_llama-0.9.0rc1-py3-none-any.whl`
- `arc_llama-0.9.0rc1.tar.gz`

## Hardware smoke test

The opt-in end-to-end inference smoke test also passed on the Linux Intel Arc
Pro B60 with the configured SYCL `llama-server` backend. It loaded the local
`lfm2.5` model, completed non-streaming and streaming chat-completion requests,
and shut the service down cleanly: **1 passed in 26.70s**.

The working tree was clean after validation.
