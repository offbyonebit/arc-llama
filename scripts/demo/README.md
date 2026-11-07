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

`record.mjs` drives the real app with Playwright (1440x900, dark, fresh profile) and notes when each
event happened. `edit.py` speeds up waits 2-4x, keeps streamed text at real speed, adds the captions and
fades, and exports the files. The shot list and prompts are at the top of each script.
