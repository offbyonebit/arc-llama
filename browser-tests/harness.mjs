// Real-browser regression suite for the bundled arc-llama web UI.
//
// Kept deliberately separate from the Python suite: this directory has its
// own package.json (playwright only), the repo's pytest testpaths never
// include it, and nothing here starts a model or touches real user config.
//
// What it covers, per the regression contract:
//   1. model selection (picker renders, choice persists in sessionStorage)
//   2. chat send (user bubble + streamed assistant reply + context meter)
//   3. structured load failure (message/action/diagnostics id + retry +
//      expandable diagnostics, from the real /admin/load error shape)
//   4. refresh preserving edits (a typed draft and the selected model both
//      survive a page reload)
//   5. keyboard navigation (Enter sends, Shift+Enter makes a newline, Esc
//      closes the settings panel)
//
// Determinism: every arc-llama HTTP endpoint is served from the repository's
// real static directory, and every API the UI calls is answered by an
// explicit mock installed with Playwright route interception before any page
// script runs. No GPU, no llama-server, no network beyond loopback.

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const STATIC = join(HERE, "..", "src", "arc_llama", "static");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".md": "text/markdown; charset=utf-8",
};

// ---------------------------------------------------------------------------
// Deterministic API mocks
// ---------------------------------------------------------------------------

const ASSISTANT_REPLY = [
  "data: " + JSON.stringify({ choices: [{ delta: { content: "Hello" } }] }) + "\n\n",
  "data: " + JSON.stringify({ choices: [{ delta: { content: " from arc-llama" } }] }) + "\n\n",
  "data: " + JSON.stringify({
    choices: [{ delta: {}, finish_reason: "stop" }],
    usage: { prompt_tokens: 10, completion_tokens: 6, total_tokens: 16 },
  }) + "\n\n",
  "data: [DONE]\n\n",
].join("");

const STRUCTURED_LOAD_FAILURE = {
  error: {
    category: "model_missing",
    message: "Model file not found: /models/missing.gguf.",
    action: "Update the model path or remove this registration.",
    diagnostics_id: "model_missing-browser-test",
    details: {
      path: "/models/missing.gguf",
      api_token: "[REDACTED]",
    },
  },
};

const MODELS = {
  object: "list",
  data: [
    {
      id: "qwen",
      object: "model",
      owned_by: "arc-llama",
      created: 1,
      metadata: { display_name: "Qwen 3", loaded: true, ctx: 8192 },
    },
    {
      id: "gemma",
      object: "model",
      owned_by: "arc-llama",
      created: 2,
      metadata: { display_name: "Gemma", loaded: false, ctx: 4096 },
    },
  ],
};

const ADMIN_STATUS = {
  server: {
    host: "127.0.0.1",
    port: 11437,
    single_resident: true,
    auto_tune: false,
    switch_drain_seconds: 30,
    switch_interrupt_policy: "reject_new",
  },
  gpus: [
    { pci_slot: "0000:03:00.0", sycl_index: 0, arch: "battlemage", vram_mb: 24576, name: "Arc Pro B60", enabled: true },
  ],
  models: [
    {
      name: "qwen",
      display_name: "Qwen 3",
      path: "/models/qwen.gguf",
      model_file_mb: 4300,
      gpu_pci_slot: "0000:03:00.0",
      port: 18080,
      loaded: true,
      ctx: 8192,
      kv_class: "default",
      aliases: [],
      vram_estimate: { estimated_mb: 5200, confidence: "estimated_from_file_size", headroom_mb: 19376, fit: true },
    },
    {
      name: "gemma",
      display_name: "Gemma",
      path: "/models/gemma.gguf",
      model_file_mb: 2600,
      gpu_pci_slot: "0000:03:00.0",
      port: 18081,
      loaded: false,
      ctx: 4096,
      kv_class: "default",
      aliases: [],
    },
  ],
  upstreams: [],
};

const METRICS = {
  uptime_seconds: 12.5,
  loads: 3,
  stops: 1,
  load_errors: 0,
  last_load_at: 123.0,
  last_error: null,
  active_models: ["qwen"],
  timings: {
    models: {
      qwen: {
        cold_start: { count: 1, last_s: 21.3, median_s: 21.3, p95_s: 21.3 },
        ttft: { count: 2, last_s: 0.31, median_s: 0.29, p95_s: 0.31 },
        generation_tok_s: { count: 2, last_tok_s: 24.8, median_tok_s: 24.5, p95_tok_s: 24.8 },
      },
    },
    queue_wait: { count: 2, last_s: 0.02, median_s: 0.01, p95_s: 0.02 },
  },
  autotune: { auto: false, models: [] },
  gpus: ADMIN_STATUS.gpus,
};

const CHATS = {
  object: "list",
  data: [
    {
      id: "chat-1",
      title: "Browser test chat",
      folder: "default",
      created_at: 1,
      updated_at: 1,
      message_count: 0,
    },
  ],
};

const sessionTokenRequests = [];

function installApiMocks(context) {
  const json = (body, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });
  return context.route("**/api/**", (route) => route.fallback())
    .then(() =>
      context.route("**/*", async (route) => {
        const url = new URL(route.request().url());
        // Remote publisher avatars are explicitly mocked; keep this suite offline.
        if (url.hostname !== "127.0.0.1") return route.abort();
        if (url.pathname.startsWith("/plugin") || url.pathname === "/plugins/vision/generate") {
          // No plugin is loaded in these tests; return a minimal error so a
          // stray plugin action fails deterministically instead of hanging.
          return route.fulfill(json({ detail: "no plugin" }, 501));
        }
        if (route.request().method() === "POST" && url.pathname === "/admin/load/gemma") {
          return route.fulfill(json(STRUCTURED_LOAD_FAILURE, 409));
        }
        if (url.pathname === "/admin/load/qwen") {
          return route.fulfill(json({ name: "qwen", loaded: true }));
        }
        switch (url.pathname) {
          case "/v1/models":
            return route.fulfill(json(MODELS));
          case "/admin/status":
            return route.fulfill(json(ADMIN_STATUS));
          case "/admin/metrics":
            return route.fulfill(json(METRICS));
          case "/admin/plugins":
            return route.fulfill(json({ plugins: [] }));
          case "/admin/session-token":
            return route.fulfill(json({ admin_token: "browser-test-token" }));
          case "/admin/integration":
            return route.fulfill(json({}));
          case "/admin/ui/layout":
            return route.fulfill(json({ layout: { toolbar: [], plugins: [], chat: [] }, actions: [], hidden: [] }));
          case "/v1/chats":
            if (route.request().method() === "GET") return route.fulfill(json(CHATS));
            return route.fulfill(json({ id: "chat-1", title: "t", created_at: 1, updated_at: 1, folder: "", messages: [] }));
          case "/v1/chats/folders":
            return route.fulfill(json({ object: "list", data: [{ folder: "default", count: 1 }] }));
          case "/v1/chat/completions":
            if (route.request().method() === "POST" && url.searchParams.size === 0) {
              return route.fulfill({ status: 200, contentType: "text/event-stream", body: ASSISTANT_REPLY });
            }
            break;
          default:
            break;
        }
        // Anything else under the app's API surface gets a harmless 404 so
        // mis-mocked calls are visible failures rather than silent hits on
        // the static file server.
        if (url.pathname.startsWith("/v1/") || url.pathname.startsWith("/admin/")) {
          return route.fulfill(json({ detail: `unmocked ${url.pathname}` }, 404));
        }
        // Non-API paths fall through to the static file server.
        return route.fallback();
      })
    );
}

// record session-token hits so a future assertion can prove the UI bootstraps
// its token only on the bundled origin (kept out of the mocked-API surface).
function recordSessionTokenCalls(page) {
  page.on("request", (req) => {
    if (req.url().includes("/admin/session-token")) sessionTokenRequests.push(req.url());
  });
}
void sessionTokenRequests;



// ---------------------------------------------------------------------------
// Static file server: the repository's real bundled UI
// ---------------------------------------------------------------------------

async function startStaticServer(staticRoot = STATIC) {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url || "/", "http://localhost");
      let pathname = decodeURIComponent(url.pathname);
      if (pathname === "/") pathname = "/index.html";
      if (pathname === "/chat") pathname = "/chat.html";
      const target = normalize(join(staticRoot, pathname));
      if (!target.startsWith(staticRoot + "/") && !target.startsWith(staticRoot + "\\")) {
        res.writeHead(403).end("forbidden");
        return;
      }
      const info = await stat(target).catch(() => null);
      if (!info || !info.isFile()) {
        res.writeHead(404).end("not found");
        return;
      }
      const body = await readFile(target);
      res.writeHead(200, { "content-type": MIME[extname(target)] || "application/octet-stream" });
      res.end(body);
    } catch {
      res.writeHead(500).end("server error");
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  return { server, port };
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

async function openChat(page, origin, { model = "qwen" } = {}) {
  await page.goto(`${origin}/chat${model ? `?model=${model}` : ""}`);
  await page.waitForSelector("#model-select option", { state: "attached", timeout: 5000 });
  // give fetchModels/fetchStatus/loadPluginActions a tick to finish
  await page.waitForTimeout(150);
}


export { ASSISTANT_REPLY, STRUCTURED_LOAD_FAILURE, MODELS, ADMIN_STATUS, METRICS, CHATS, installApiMocks, recordSessionTokenCalls, startStaticServer, openChat };
