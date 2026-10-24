#!/usr/bin/env bash
# Run every automated check before calling a change done. Usage: scripts/check.sh [--live]
#   --live also checks real OpenCode tool profiles and Hermes ACP (slow, starts MCP servers; uses fake inference).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
failed=()

step() {
  local name=$1; shift
  printf '\n== %s\n' "$name"
  if "$@"; then printf 'ok: %s\n' "$name"; else failed+=("$name"); printf 'FAILED: %s\n' "$name"; fi
}

step "ruff" .venv/bin/ruff check src tests scripts registry-repo/scripts
step "mypy" .venv/bin/mypy src
step "javascript syntax" bash -c 'for f in browser-tests/*.mjs src/arc_llama/static/*.js src/arc_llama/static/*/*.js; do case "$f" in *vendor*) continue;; esac; node --check "$f" || exit 1; done'
step "pytest" "$PY" -m pytest -q
step "browser suite" bash -c 'cd browser-tests && node run.mjs'
if [ "${1:-}" = "--live" ]; then
  step "real OpenCode profile enforcement" "$PY" scripts/check_opencode_profiles.py
  step "real Hermes ACP integration" "$PY" scripts/check_hermes.py
fi

printf '\n'
if [ ${#failed[@]} -eq 0 ]; then echo "All checks passed."; else echo "FAILED: ${failed[*]}"; exit 1; fi
