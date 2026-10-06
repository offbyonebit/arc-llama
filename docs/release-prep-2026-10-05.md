# Arc Llama 0.9.0 local release preparation — 2026-10-05

This records the local release-preparation state. **No release candidate or
0.9.0 release was published.** The project owner asked to skip another public
candidate, so this work prepares and validates a local `0.9.0` package without
claiming a public candidate soak.

## Integrated local changes

- Checkout: `/home/slowe/arc-llama`, branch `maintenance/rc1-local-follow-up`,
  base commit `4c2237d`.
- PRs #64–#67 are integrated in the working tree; their original PRs remain
  separate. The prepared change was committed and pushed as
  `3724761` on this branch. Draft integration PR #68 targets `main`; it is not
  merged, and no release was published.
- The branch was merged locally with the current `origin/main` tip
  (`57764df`), which removes the vision companion from this repository because
  it is maintained separately. The companion suite had passed before that
  repository split.
- Package metadata is set to `0.9.0` in `pyproject.toml` and `uv.lock`.
- The README identifies this as unpublished release preparation. The changelog
  records the locally integrated PR changes under `Unreleased`.

## Local validation

- Current core CI-style suite after merging `main` (Python 3.12):
  **1143 passed, 1 skipped** (the opt-in live smoke test), with one Starlette
  deprecation warning. The companion suite previously passed **86 tests**.
- Current browser suite after merging `main`: **10 passed**.
- Native inference on the existing `lfm2.5` model after applying PRs:
  **1 smoke test passed**, including streaming and non-streaming responses.
- Current focused regression run for the GPU ownership and semantic-search
  changes: **25 passed**.
- Ruff, mypy (48 source files), JavaScript syntax, `uv lock --check`, and
  `git diff --check` passed on the current release-preparation tree.
- Built a wheel and source archive locally from the source distribution:
  `arc_llama-0.9.0-py3-none-any.whl` and `arc_llama-0.9.0.tar.gz`. Archive
  inspection confirmed the `0.9.0` wheel metadata, required UI assets, and GPU
  setup guide in the source archive. Installing the wheel into a temporary
  target and running its CLI returned `python -m arc_llama, version 0.9.0`.
  Artifacts are under `/tmp/arc-llama-0.9.0-release-prep/` and are not
  published.

## GPU compatibility evidence

The available physical GPU is an Intel Arc Pro B60. Intel identifies it as
Battlemage/Xe2 and lists 24 GB GDDR6, oneAPI support, and Vulkan 1.3 in the
[B60 specifications](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html).
Intel's [supported API table](https://www.intel.com/content/www/us/en/support/articles/000005524/graphics.html)
lists Vulkan 1.3 for both consumer Alchemist and Battlemage families.

The upstream [llama.cpp SYCL guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
lists B580 as a verified Battlemage device and A770, A750, and A730M as verified
Alchemist devices. Its [Intel GPU benchmark discussion](https://github.com/ggml-org/llama.cpp/discussions/23313)
contains B60 SYCL and Vulkan results, including a B60/B580 system where both
are identified as BMG G21.

This evidence supports the Arc family/backend mapping and makes the B60 a
relevant Battlemage representative. It does not certify an individual consumer
SKU, every driver version, Windows behavior, or Arc Llama itself on untested
hardware. The B60's 24 GB memory is twice the B580's 12 GB, so its model-fit
results do not transfer to the B580. Physical coverage remains Linux B60 on
Vulkan and SYCL; native Windows and physical consumer Alchemist/Battlemage
runs were not performed.

## Remaining release gates

- The exact integrated tree has not run on GitHub Actions yet. Draft PR #68
  targets `main` to trigger Linux and Windows CI on Python 3.10, 3.12, and
  3.14, plus the bare-wheel job.
- The original host-RAM incident's exact old workload remains untested after
  the ownership changes. The passing bounded Qwen run documented in
  `release-validation-linux-2026-10-03.md` uses the CPU vision projector and
  does not prove that the old all-GPU-projector recipe is safe.
- The repository's release plan calls for a public release-candidate soak.
  That gate was intentionally skipped at the owner's direction; this report
  does not claim the planned soak occurred.

The local package is prepared and the available code/inference checks pass, but
the missing exact-tree CI and incident-workload evidence remain review points
before deciding to publish `0.9.0`.
