# Architecture and change boundaries

This map explains where existing behavior belongs. Use it to keep feature work
and structural refactors small, with explicit dependencies and unchanged public
interfaces. It is not a second feature roadmap.

## Entry points and feature owners

Paths below are relative to `src/arc_llama/`.

| Responsibility | Current owner | Boundary |
| --- | --- | --- |
| Command line coordinator | `cli.py` | Own shared context, logging, initial setup and run/serve orchestration; register command families and retain compatibility exports. Shared model/runtime policy belongs in service modules. |
| Command families | `cli_models.py`, `cli_diagnostics.py`, `cli_runtime.py`, `cli_performance.py`, `cli_support.py`, `cli_agent.py`, `cli_recipes.py`, `cli_upstream.py` | Own model catalog, diagnostics, runtime, performance, keys/plugins, and optional command groups. Dependencies are supplied by the coordinator. |
| HTTP application | `server.py` | Assemble the FastAPI app, authentication, lifespan, protocol handlers, and response streaming. App-scoped services live on `app.state`. |
| Library and frontend API groups | `api/library.py`, `api/integration.py` | Register existing admin routes with centrally supplied authentication and helpers. Read app-scoped services from request state; retain the configured persistence path. |
| Model selection and residency | `router.py` | Resolve registered models, serialize load/switch decisions, enforce memory admission, and coordinate draining. |
| Model subprocess | `launcher.py`, `preflight.py`, `gpu_ownership.py` | Build launch plans, reject predictable launch failures, manage process lifetime and cross-process GPU ownership. |
| Resource arbitration | `resources.py` | Coordinate shared text inference and exclusive plugin work through leases. Delegate model process management to the router. |
| Runtime installation | `runtime.py`, `runtime_update.py`, `binary.py`, `binary_caps.py` | Select and validate runtime assets, manage installation/update/rollback, and inspect binary capabilities. |
| Hardware discovery and launch policy | `detect.py`, `arch.py`, `platform_checks.py`, `policy.py` | Identify hardware and apply backend-specific constraints. Configuration and detection are not proof of successful inference. |
| Model discovery and registration | `models.py`, `gguf_meta.py`, `model_library.py` | Scan/register GGUFs, read metadata, search/download from Hugging Face, and report storage/file availability. |
| Explicit architecture checks | `model_compatibility.py`, `compatibility_io.py` | Own bounded header/evidence rules, app-scoped admission and deadlines, cancellable Hub reads, and temporary CPU-only probes. These checks do not establish inference compatibility or load model weights. |
| Configuration and recipes | `config.py`, `recipes.py`, `recipe_share.py`, `workload.py` | Validate and persist configuration, select recipes, and describe workload goals. Frontends consume these rules rather than reimplementing them. |
| Tuning and measurement | `tune.py`, `autotune.py`, `benchmark.py`, `stream_metrics.py`, `perf_history.py` | Run explicit or configured tuning, measure real requests, and retain bounded performance history. |
| Chat persistence | `chat_store.py` | Store conversations and folders; HTTP handlers expose the operations. Browser drafts remain separate from saved conversations. |
| Upstream compatibility | `upstream.py`, `integration.py`, protocol handlers in `server.py` | Route to configured upstreams and provide frontend connection guidance. |
| Optional extensions | `plugin_api.py`, `plugins.py`, `plugin_scaffold.py`, `agent/`, `skills.py` | Keep optional imports and lifecycle isolated. Companion implementations remain separate repositories. |

## Browser organization

The dashboard uses local classic scripts without a build step. `index.html`
loads component factories before `app.js`, then the coordinator constructs them
with explicit callbacks. Loading a component script must not start polling,
download files, or run inference.

- `static/app.js` owns shared authentication, the latest model status snapshot,
  model selection, readiness, navigation, and application bootstrap.
- `static/dashboard/library.js` owns Hugging Face search, download-job display
  and polling, and disk-usage controls. It asks the coordinator to refresh status
  or review a registered model; it does not own selection.
- `static/dashboard/measurements.js` owns System measurement rendering and
  generation history. Its data comes from measured server results.
- `static/dashboard/plugins.js` owns the plugin catalog, action placement,
  and layout editor. Catalog and layout data remain private to this controller.
- `static/dashboard/integration.js` owns frontend connection guidance, tabs,
  and copy controls. Commands are displayed and copied, never executed.
- `static/chat.js` owns shared conversation/model state, authentication,
  send/stop/load lifecycle, usage accounting, plugin composer, and bootstrap.
- `static/chat/history.js` owns saved-chat storage, synchronization, history UI,
  folders, and import/export.
- `static/chat/attachments.js` owns attachment processing and composer controls.
- `static/chat/settings.js` owns model settings, edit drafts, and presets.
- `static/chat/rendering.js` owns safe message/Markdown rendering and thinking
  display. Streaming requests, aborts, usage, and turn persistence remain in
  the coordinator.

Chat loads its factories before `chat.js` through a separate `ArcChat` namespace.
Pass explicit state accessors for values the coordinator replaces, such as the
model list; copying their current value would leave a component stale.

Components own their private state and timers. The coordinator owns shared
state. Keep the factory namespace narrow; pass dependencies instead of making
components reach into each other's mutable state. Bootstrap must bind each
control and start each polling loop only once.

## Request and resource lifetime

A local inference request enters through `server.py`, resolves its model through
the router, passes launch/admission checks, and is forwarded to `llama-server`.
Streaming cleanup is part of the request lifetime, not just the initial HTTP
response. The server's global request count, router's per-model counts, and
resource leases must be released on success, failure, disconnect, and cancellation.

Preserve these rules during refactors:

1. A draining model cannot acquire new work through the ready-model fast path.
2. Preflight checks happen before evicting a healthy resident when applicable.
3. GPU ownership lasts until the model process has actually exited.
4. Plugin leases arbitrate GPU use without independently managing model processes.
5. Tuning respects active work and restores its temporary configuration on exit.
6. Read-only status and UI polling do not trigger model loading or inference.
7. File availability, estimated memory fit, process health, and successful
   inference remain distinct claims.
8. Authentication, loopback token restrictions, redaction, and plugin route
   validation stay in force when routes or helpers move.

## How to reorganize safely

Extract one responsibility at a time from the current working tree. Preserve
existing commands, URLs, configuration fields, and public imports. Keep wrappers
where callers need compatibility; avoid duplicate implementations. Make feature
changes separately from structural moves, and document intentionally changed
contracts before changing callers.

This organization pass covers dashboard and chat components, library/integration
API groups, and CLI command families. Further extractions should follow a
responsibility that needs independent changes or testing. Runtime/residency
changes deserve their own focused review and hardware evidence. Do not split a
file merely to meet a line-count target.

## Verification and planning sources

- [Testing](testing.md): Python tests, lint, types, and restricted-environment
  guidance. `browser-tests/run.mjs` exercises real packaged frontend source with
  synthetic APIs; it does not validate GPU inference.
- Build a wheel after adding or moving frontend assets and verify every locally
  referenced script is included. A source-checkout browser pass alone does not
  prove that an installed package works offline.
- [1.0 plan](1.0-plan.md) owns core release scope and gates.
  [Readiness audit](1.0-readiness-audit.md) records supporting findings.
- [Repository boundary](../LOCAL_REPO_BOUNDARIES.md) keeps the vision companion
  outside this checkout. [Plugins](plugins.md) describes its integration boundary.
- A configured GPU, a unit-test pass, and browser fixture results are not a
  supported-hardware certification. Record actual inference evidence separately.
