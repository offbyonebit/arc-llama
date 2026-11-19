# README demo

Records `docs/media/demo.mp4`, `demo.webp` and `demo-poster.png`. Everything runs in an isolated
home under `/tmp/arc-demo` (`env.sh`): no real config, models list, chats or OpenCode data shows up.
Needs an Arc GPU, Chrome, ffmpeg with libx264/libwebp, the Inter font, and Playwright's ffmpeg
(`PLAYWRIGHT_BROWSERS_PATH=<dir> npx playwright install ffmpeg` from `browser-tests/`).

Stop anything else using the GPU and port 11437 first.

## One-time setup

```bash
source scripts/demo/env.sh
uv venv $DEMO/venv && uv pip install -p $DEMO/venv/bin/python dist/arc_llama-<version>-py3-none-any.whl
ln -sf $DEMO/venv/bin/arc-llama $DEMO/bin/arc-llama
printf '#!/bin/sh\necho "$1" > %s/opened-url\n' "$DEMO" > $DEMO/bin/open-url && chmod +x $DEMO/bin/open-url
ln -s /path/to/Ornith-1.5-35B-Q4_K_M.gguf $DEMO/models/
arc-llama init --force --no-scan --llama-server /path/to/sycl/llama-server --workload-context short
arc-llama add $DEMO/models/Ornith-1.5-35B-Q4_K_M.gguf --name ornith-35b --display-name "Ornith 1.5 35B-A3B" --ctx 16384
arc-llama install-harness opencode
```

Then add `extra_flags = ["--reasoning-budget", "0"]` to the model's `[models.recipe]` (so the chat shot
streams the answer rather than a long hidden reasoning pass) and save the file as `$DEMO/config-model.toml`.

## Record and cut

```bash
PLAYWRIGHT_BROWSERS_PATH=<dir> node scripts/demo/record.mjs all   # or terminal|discover|chat|work
python3 scripts/demo/edit.py                                      # writes docs/media/
```

`record.mjs` drives the real app with Playwright (1440x900, dark, fresh profile), captures it through
Chrome's screencast at 2x as quality-100 JPEGs (Playwright's own video is low-bitrate VP8 that smears
text), and notes when each event happened. `edit.py` speeds up waits 2-4x, keeps streamed text at real speed, adds the captions and
fades, and exports the files. The shot list and prompts are at the top of each script.

## 1.2 trailer

The original `record.mjs all` / `edit.py` demo is unchanged. The trailer needs the same tools,
plus ffmpeg's libass filter for kinetic Inter ExtraBold type. Use the setup above on the B60,
with the release/1.2 wheel and a current catalog offering the Ornith 1.0 9B starter. Fresh setup
chooses its runtime; both Vulkan and SYCL are accepted and the active backend is recorded.
Install the Hermes harness too (`arc-llama install-harness hermes`) for the montage. The saved
`$DEMO/config-model.toml` still supplies the original Work model/harness configuration.

Optionally register a 27B model in that saved config, with its MTP/draft recipe configured there.
Set `TRAILER_27B_MODEL` to its registered model ID when recording and pass `--with-27b` when editing.
Without it, first run and the montage use the extra time while the speed counter stays at four seconds
(at 120 BPM). The speculative decoding title appears only if new server log lines during that take report a positive drafted/accepted token count;
unrecognized or absent evidence omits the title. No recipe flag alone counts as active decoding.

```bash
source scripts/demo/env.sh
# Optional: export TRAILER_27B_MODEL=<registered-27b-model-id>
PLAYWRIGHT_BROWSERS_PATH=<dir> node scripts/demo/record.mjs trailer
python3 scripts/demo/trailer.py --music /path/to/licensed-track.wav --bpm 120
# Add --with-27b to the edit command if that take was recorded.
```

The recorder uses the real first-run Download and start flow (allow time/disk space for weights),
then captures its answer, Thinking budget 256 and save, the original Work flow, and the montage.
The terminal's detected-device line is explicitly labelled as live app data: `start` no longer
prints it. Trailer takes live in `$DEMO/rec/trailer`, separately from the demo takes.
Every speed take stores its own UI-parsed tok/s and assigned GPU name/VRAM in JSON.
The first-run JSON also stores the app's OpenAPI version, starter name and active backend.
Missing speed/hardware evidence aborts the edit. Version cards use that app version, including
`1.1.1` until the independent version bump merges; no trailer code changes the package version.

The cut is 33 seconds at 120 BPM, 1440x900 at 30 fps. Other BPM values scale the beat schedule;
streaming always plays at 1x, with a held last frame if the answer finishes early. Long downloads
use head/tail excerpts at 8x. `timing.json` records frame-quantized cuts and verifies beat alignment
within one frame. Omit `--music` for silent MP4; no music is stored here. WebP is always silent and
steps down fps/quality/size to stay below 3 MiB. Real output goes to `docs/media/trailer-1.2.mp4`,
`trailer-1.2.webp` and `trailer-1.2-poster.png`; the existing README demo is never replaced.
Watch the real render before deciding whether to update the README separately.

No-GPU timing/type proof (never commit these outputs):

```bash
python3 scripts/demo/trailer.py --placeholder --with-27b --out /tmp/trailer-1.2-proof
```

This synthesizes labelled dark clips and JSON with `"tps": "PLACEHOLDER"`, renders `##.#`,
and exports six PNG frame grabs plus `timing.json`. Placeholder output defaults to
`/tmp/arc-trailer-placeholder`; writing placeholders under `docs/media` is rejected.
`--rec <directory>` and `--out <directory>` can override the real take/output directories.
