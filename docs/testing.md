# Testing and diagnosing stalls

Run the core checks from the repository environment:

```bash
python -m pytest tests/ -q -ra --tb=short -p no:cacheprovider
ruff check src tests registry-repo/scripts
mypy --check-untyped-defs src
```

Pytest emits thread stacks after 30 seconds in a test. CI jobs have a hard
15-minute deadline, so a stalled test cannot occupy a worker indefinitely.
For local Linux investigations, use an outer deadline as well:

```bash
timeout 120s python -m pytest tests/test_config_atomic_save.py -vv -p no:cacheprovider
```

## Dashboard and package checks

See [Architecture and change boundaries](architecture.md) for component owners
and the invariants to preserve when extracting code.

Run the browser regression suite after dashboard or chat behavior changes:

```bash
cd browser-tests
npm test
```

The polling regression cases count status/job requests, retain model-card node
identity across unchanged responses, verify updated settings and keyboard focus,
and exercise failed completion-refresh retries. A held status response verifies
that simultaneous forced refreshes share one fresh follow-up read. Synthetic
visibility events check the hidden-tab timer cadence and one refresh on return;
they do not claim to measure a browser's own background throttling.

The harness serves the real frontend files against synthetic API responses.
It does not start models or change user configuration. Static UI tests also
check JavaScript syntax recursively, including dashboard and chat components.

For a structural pass that must not run inference, explicitly exclude the live
smoke test:

```bash
env -u ARC_LLAMA_SMOKE_MODEL python -m pytest tests/ -m "not live_inference" -q -ra --tb=short -p no:cacheprovider
```

When adding frontend assets, build a wheel and check that scripts referenced by
`index.html` and `chat.html` are included under `arc_llama/static/`. A browser
pass from a source checkout does not verify installed-package assets.

## Restricted execution environments

A TestClient stall is not necessarily an application startup deadlock. On
2026-09-18, the persistence rollback test stalled inside the restricted agent
sandbox, as did entering `TestClient(FastAPI())` with no Arc Llama application.
The same rollback test passed outside that sandbox in 0.41 seconds. The full
baseline suite at commit `81c33f4` passed there with 936 passed and 13 skipped
in 18.94 seconds. The stack dump showed the main thread waiting in AnyIO's
blocking portal and its event-loop thread waiting in the selector.

When this signature recurs, first compare a minimal empty FastAPI TestClient
inside and outside the restricted environment. Use the approved environment
for the real checks; do not bypass lifespan, suppress tests, or change runtime
synchronization to mask an execution-environment problem. These observations
do not identify the exact sandbox restriction responsible.

The baseline skips required local GGUF fixtures or an explicitly configured
GPU inference smoke model. Unit-test success is not hardware validation.
