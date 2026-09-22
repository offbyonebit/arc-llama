# arc-llama community recipes

Shared tune recipes for [arc-llama](https://github.com/offbyonebit/arc-llama), keyed by
a **shareable fingerprint**: GPU arch + backend + model class + workload profile +
VRAM bucket + tuner schema version. Machine-local details (paths, mtimes, llama-server
build) are deliberately excluded so a recipe measured on one B580 works on another.

## How it works

- `arc-llama tune --share` writes a submission JSON to disk and prints a pre-filled PR
  link. Nothing is uploaded automatically — you choose to open the URL.
- CI on this repo validates every submission (`scripts/validate.py`) and regenerates
  the bundled `dist/recipes.json` (`scripts/aggregate.py`).
- Aggregation never combines prompt and generation records from different
  recipes. Matching recipe/build trials use median measurements; competing
  groups remain in the bundle for auditing and the balanced winner is selected.
- The bundle ships inside the `arc-llama` wheel each release; `arc-llama recipes update`
  pulls the newest asset. Lookups are local and offline; a miss just means you run
  your own sweep as usual.
- `arc-llama recipes apply MODEL` checks confidence and build provenance, then
  runs a local A/B benchmark and restores the original recipe unless the shared
  result wins.

## Submitting

1. `arc-llama tune <model> --share`
2. Open the printed URL, attach the JSON file from the printed path, submit the PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the schema and review rules.
