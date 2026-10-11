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
import { chromium } from "playwright";

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

export { installApiMocks, recordSessionTokenCalls, startStaticServer };

// ---------------------------------------------------------------------------
// Static file server: the repository's real bundled UI
// ---------------------------------------------------------------------------

async function startStaticServer() {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url || "/", "http://localhost");
      let pathname = decodeURIComponent(url.pathname);
      if (pathname === "/") pathname = "/index.html";
      if (pathname === "/chat") pathname = "/chat.html";
      const target = normalize(join(STATIC, pathname));
      if (!target.startsWith(STATIC)) {
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

async function testChatSelection({ page, origin }) {
  await openChat(page, origin);
  const options = await page.$$eval("#model-select option", (els) => els.map((e) => e.value));
  if (options.join(",") !== "qwen,gemma") throw new Error(`unexpected model options: ${options}`);
  await page.selectOption("#model-select", "gemma");
  const stored = await page.evaluate(() => sessionStorage.getItem("arc-llama-selected-model"));
  if (stored !== "gemma") throw new Error(`selected model not persisted, got ${stored}`);
  const status = await page.textContent("#status-text");
  // gemma is not loaded per the mocked admin status
  if (status !== "idle") throw new Error(`expected idle status, got ${status}`);
  await page.selectOption("#model-select", "qwen");
  await page.waitForTimeout(100);
}

async function testChatHistoryComponents({ page, origin }) {
  const chats = [
    { id: "chat-history-a", title: "General notes", folder: "", created_at: 1, updated_at: 3, message_count: 1 },
    { id: "chat-history-b", title: "Project notes", folder: "Project", created_at: 2, updated_at: 4, message_count: 1 },
  ];
  let imported = null;
  await page.route("**/v1/chats", route => route.fulfill({
    status: 200, contentType: "application/json", body: JSON.stringify({ object: "list", data: chats }),
  }));
  await page.route("**/v1/chats/folders", route => route.fulfill({
    status: 200, contentType: "application/json", body: JSON.stringify({ data: [{ name: "Project", count: 1 }] }),
  }));
  await page.route("**/v1/chats/chat-history-b", route => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({ id: "chat-history-b", title: "Project notes", folder: "Project", model: "qwen", messages: [{ role: "user", content: "Saved project message" }] }),
  }));
  await page.route("**/v1/chats/export", route => route.fulfill({
    status: 200, contentType: "application/json", body: JSON.stringify({ chats }),
  }));
  await page.route("**/v1/chats/import", async route => {
    imported = route.request().postDataJSON();
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ imported: 1, skipped: 0, errors: 0 }) });
  });
  await openChat(page, origin);
  await page.click("#history-toggle");
  await page.waitForSelector("#h-list .h-card");
  const titles = await page.locator("#h-list .h-card-title").allTextContents();
  if (!titles.includes("Project notes") || !titles.includes("General notes")) throw new Error(`history component omitted chats: ${titles}`);
  await page.selectOption("#h-folder", "Project");
  if ((await page.locator("#h-list .h-card-title").allTextContents()).join(",") !== "Project notes") throw new Error("folder filter did not isolate its chats");
  await page.locator("#h-list .h-card").click();
  await page.waitForSelector(".message.user .content");
  if (!(await page.locator(".message.user .content").innerText()).includes("Saved project message")) throw new Error("history component did not restore the selected chat");

  await page.click("#history-toggle");
  const downloadPromise = page.waitForEvent("download");
  await page.click("#h-export");
  const download = await downloadPromise;
  if (!download.suggestedFilename().startsWith("arc-llama-chats-")) throw new Error("history export used an unexpected filename");

  await page.setInputFiles("#h-import-input", {
    name: "fixture.json", mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify({ chats: [{ title: "Imported fixture", messages: [] }] })),
  });
  await page.waitForFunction(() => document.querySelector(".message.error .content")?.textContent.includes("Imported 1, skipped 0, errors 0"));
  if (!imported || imported.overwrite !== false || imported.chats[0].title !== "Imported fixture") throw new Error("history import did not preserve its safe import payload");
}

async function testChatSend({ page, origin }) {
  await openChat(page, origin);
  await page.fill("#message-input", "hi there");
  await page.press("#message-input", "Enter");
  await page.waitForSelector("#chat-log .message.user", { timeout: 5000 });
  await page.waitForSelector("#chat-log .message.assistant", { timeout: 5000 });
  await page.waitForFunction(() => {
    const el = document.querySelector("#chat-log .message.assistant .content");
    return el && el.textContent.includes("Hello from arc-llama");
  }, { timeout: 5000 });
  const userText = await page.textContent("#chat-log .message.user .content");
  if (!userText.includes("hi there")) throw new Error("user bubble missing prompt text");
  // The streamed reply persisted into the conversation state, and the ctx
  // meter became visible with usage-derived numbers.
  await page.waitForFunction(() => document.getElementById("ctx-meter").classList.contains("visible"), {
    timeout: 5000,
  });
}

async function testStructuredLoadFailure({ page, origin }) {
  await openChat(page, origin, { model: "gemma" });
  let releaseLoad;
  const pendingLoad = new Promise(resolve => { releaseLoad = resolve; });
  await page.route("**/admin/load/gemma", async route => {
    await pendingLoad;
    return route.fulfill({ status: 409, contentType: "application/json", body: JSON.stringify(STRUCTURED_LOAD_FAILURE) });
  });
  // gemma is not loaded: sending a message triggers /admin/load/gemma, which
  // the mock answers with the real structured failure shape.
  await page.fill("#message-input", "please answer");
  await page.press("#message-input", "Enter");
  try {
    await page.waitForSelector(".load-wait-card", { state: "visible" });
    if (!(await page.textContent(".load-wait-card")).includes("Starting model")) throw new Error("missing readable loading stage");
  } finally {
    releaseLoad(); // A failed assertion must not leave a route handler blocked.
  }
  await page.waitForSelector(".load-failure-card", { timeout: 5000 });
  const card = page.locator(".load-failure-card");
  const text = await card.innerText();
  if (!text.includes("Model file not found")) throw new Error("failure card missing message");
  if (!text.includes("Update the model path")) throw new Error("failure card missing action");
  if (!text.includes("model_missing-browser-test")) throw new Error("failure card missing diagnostics id");
  const category = await card.getAttribute("data-failure-category");
  if (category !== "model_missing") throw new Error(`bad category ${category}`);
  // The waiting card is gone (honest loading stage ends with the failure).
  const waiting = await page.locator(".load-wait-card").count();
  if (waiting !== 0) throw new Error("loading wait card survived a failed load");
  // Diagnostics start collapsed, and expand to a bounded JSON dump.
  const details = card.locator(".load-failure-details");
  if (!(await details.isHidden())) throw new Error("diagnostics should start collapsed");
  await card.locator(".load-failure-details-toggle").click();
  if (!(await details.isVisible())) throw new Error("diagnostics did not expand");
  const detailsText = await details.innerText();
  if (!detailsText.includes("/models/missing.gguf")) throw new Error("diagnostics missing path");
  if (detailsText.includes("must-not-leak")) throw new Error("unredacted secret rendered");
  // The redacted marker from the mocked (already-redacted) payload shows.
  if (!detailsText.includes("[REDACTED]")) throw new Error("redaction marker missing");
  // The user turn stays in the transcript for the retry to reuse.
  // Retry button re-attempts: the mock fails deterministically again.
  await card.locator(".load-failure-retry").click();
  await page.waitForSelector(".load-failure-card", { timeout: 5000 });
  // No assistant bubble was created for the failed send.
  const assistants = await page.locator("#chat-log .message.assistant").count();
  if (assistants !== 0) throw new Error(`assistant bubble appeared for failed send (${assistants})`);
}

async function testRefreshPreservesEdits({ page, origin }) {
  await openChat(page, origin, { model: "" });
  await page.selectOption("#model-select", "gemma");
  await page.fill("#message-input", "draft that must survive refresh");
  await page.reload();
  await page.waitForSelector("#model-select option", { state: "attached", timeout: 5000 });
  await page.waitForTimeout(300); // init() restores the draft after models load
  const restoredModel = await page.evaluate(() => sessionStorage.getItem("arc-llama-selected-model"));
  if (restoredModel !== "gemma") throw new Error(`model selection lost after refresh: ${restoredModel}`);
  const restoredDraft = await page.evaluate(() =>
    sessionStorage.getItem("arc-llama-draft-" + sessionStorage.getItem("arc-llama-selected-model"))
  );
  if (restoredDraft !== "draft that must survive refresh") {
    throw new Error(`draft not persisted after refresh, got ${JSON.stringify(restoredDraft)}`);
  }
  const inputValue = await page.inputValue("#message-input");
  if (inputValue !== "draft that must survive refresh") {
    throw new Error(`composer did not restore draft, got ${JSON.stringify(inputValue)}`);
  }
}

async function testKeyboardNavigation({ page, origin }) {
  await openChat(page, origin);
  // Enter sends; Shift+Enter inserts a newline instead.
  await page.focus("#message-input");
  await page.type("#message-input", "first line");
  await page.keyboard.press("Shift+Enter");
  await page.keyboard.type("second line");
  const value = await page.inputValue("#message-input");
  if (!value.includes("\n")) throw new Error("Shift+Enter did not add a newline");
  const assistantBefore = await page.locator("#chat-log .message").count();
  await page.keyboard.press("Enter");
  await page.waitForFunction(
    (n) => document.querySelectorAll("#chat-log .message").length > n,
    assistantBefore,
    { timeout: 5000 }
  );
  // Escape closes the settings panel when open.
  await page.click("#settings-toggle");
  if (!(await page.locator("#settings-panel").evaluate((el) => el.classList.contains("open")))) {
    throw new Error("settings panel did not open");
  }
  await page.keyboard.press("Escape");
  await page.waitForTimeout(120);
  const stillOpen = await page.locator("#settings-panel").evaluate((el) => el.classList.contains("open"));
  if (stillOpen) throw new Error("Escape did not close the settings panel");
}

async function testSettingsSurviveStatusPoll({ page, origin }) {
  await openChat(page, origin);
  await page.click("#settings-toggle");
  await page.fill("#s-ctx", "16384");
  await page.selectOption("#s-ctk", "f16");
  await page.evaluate(() => document.activeElement.blur());
  await page.evaluate(() => fetchStatus());
  if (await page.inputValue("#s-ctx") !== "16384") throw new Error("status poll erased unsaved context");
  if (await page.inputValue("#s-ctk") !== "f16") throw new Error("status poll erased unsaved KV setting");
  if (!(await page.textContent("#s-fit")).includes("Unsaved")) throw new Error("draft was shown as an applied estimate");
}

async function testRetrySendsOriginalTurnOnce({ page, origin }) {
  await openChat(page, origin, {model: "gemma"});
  await page.fill("#message-input", "retry this exact message");
  await page.press("#message-input", "Enter");
  await page.waitForSelector(".load-failure-card");
  let loads = 0;
  const requests = [];
  await page.route("**/admin/load/gemma", route => {
    loads++;
    return route.fulfill({status: 200, contentType: "application/json", body: JSON.stringify({loaded: true})});
  });
  await page.route("**/v1/chat/completions", route => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({status: 200, contentType: "text/event-stream", body: ASSISTANT_REPLY});
  });
  await page.locator(".load-failure-retry").click();
  await page.waitForFunction(() => document.querySelector(".message.assistant .content")?.textContent.includes("Hello from arc-llama"));
  if (loads !== 1 || requests.length !== 1) throw new Error("retry did not issue exactly one load and completion");
  const turns = requests[0].messages.filter(m => m.role === "user");
  if (turns.length !== 1 || turns[0].content !== "retry this exact message") throw new Error("retry duplicated or changed the user turn");
  if (await page.locator(".message.user").count() !== 1) throw new Error("retry duplicated the visible user message");
}

async function testDashboardMeasurements({ page, origin }) {
  await page.goto(`${origin}/#system`);
  await page.waitForFunction(() => document.querySelector("#measurements")?.textContent.includes("24.8"));
  const content = await page.locator("#measurements").innerText();
  if (!content.includes("tok/s") || content.includes("n/a")) throw new Error("generation rates have incorrect units or values");
}

async function testMemoryFitAndSavedSettings({ page, origin }) {
  await openChat(page, origin);
  await page.click("#settings-toggle");
  let fit = ADMIN_STATUS.models[0].vram_estimate;
  await page.route("**/admin/status", route => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({ ...ADMIN_STATUS, models: ADMIN_STATUS.models.map(m => m.name === "qwen" ? { ...m, vram_estimate: fit } : m) }),
  }));
  for (const [estimate, expected] of [
    [fit, "MiB headroom on the assigned GPU"],
    [{ estimated_mb: 25000, headroom_mb: -424, fit: false }, "Reduce context or KV size, or pick a smaller model"],
    [{ estimated_mb: 5200, fit: null }, "GPU capacity unknown; no fit verdict"],
    [null, "not estimated yet"],
  ]) {
    fit = estimate;
    await page.evaluate(() => fetchStatus());
    if (!(await page.locator("#s-fit").isVisible())) throw new Error("memory fit line is hidden");
    if (!(await page.textContent("#s-fit")).includes(expected)) throw new Error(`incorrect memory fit for ${JSON.stringify(fit)}`);
  }
  let edits;
  await page.route("**/admin/models/qwen/edit", route => {
    edits = route.request().postDataJSON();
    fit = { estimated_mb: 6000, headroom_mb: 18576, fit: true };
    return route.fulfill({ status: 200, contentType: "application/json", body: "{}" });
  });
  await page.fill("#s-ctx", "16384");
  await page.click("#s-apply");
  await page.waitForFunction(() => document.querySelector("#s-fit").textContent.includes("18,576"));
  if (edits?.ctx !== 16384) throw new Error("settings were not sent to the server");
  if (await page.locator("#s-apply").isDisabled()) throw new Error("apply button was not restored");
}

async function testDashboardFitAndEmptyMeasurements({ page, origin }) {
  await page.goto(origin);
  await page.waitForSelector("#readiness-card .readiness-metrics");
  if (!(await page.locator("#readiness-card .tone-ok").first().innerText()).includes("headroom")) throw new Error("fitting model has no visible fit verdict");
  await page.route("**/admin/status", route => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({ ...ADMIN_STATUS, models: ADMIN_STATUS.models.map(m => ({ ...m, vram_estimate: { estimated_mb: 25000, fit: false } })) }),
  }));
  await page.evaluate(() => fetchStatus(true));
  if (!(await page.locator("#readiness-card .tone-warn").first().innerText()).includes("Does not fit")) throw new Error("oversized model has no warning");
  await page.route("**/admin/metrics", route => route.fulfill({
    status: 200, contentType: "application/json", body: JSON.stringify({ ...METRICS, timings: { models: {}, queue_wait: null } }),
  }));
  await page.evaluate(() => fetchMeasurements());
  await page.locator('[data-view-link="system"]').click();
  const empty = await page.locator("#measurements").innerText();
  if (!empty.includes("No measurements yet") || empty.includes("24.8")) throw new Error("dashboard retained stale rates in empty state");
}

async function testDashboardPluginLayoutController({ page, origin }) {
  let saved = { layout: { toolbar: ["demo.open"], plugins: [], chat: [] }, hidden: [] };
  let layoutWrite = null;
  await page.route("**/admin/plugins", route => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({ plugins: [{
      name: "demo", status: "active", version: "1.2", description: "Browser fixture plugin",
      api: ["/plugins/demo/run"], ui: { pages: [{ path: "/plugins/demo/", label: "Open demo" }], actions: [{ id: "demo.open", label: "Open demo action", route: "/plugins/demo/" }] },
    }] }),
  }));
  await page.route("**/admin/ui/layout", route => {
    if (route.request().method() === "PUT") {
      layoutWrite = route.request().postDataJSON();
      saved = layoutWrite;
    }
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(saved) });
  });
  await page.goto(`${origin}/#system`);
  await page.getByRole("heading", { name: "demo" }).waitFor();
  if (!(await page.locator(".plugin-page-link").getAttribute("href")).includes("/plugins/demo/")) throw new Error("plugin page link was not rendered");
  await page.locator(".plugin-action").waitFor();
  await page.click("#customize-ui");
  const actionRow = page.locator("#ui-layout-list .plugin-card").filter({ hasText: "Open demo action" });
  await actionRow.locator("select").selectOption("plugins");
  await page.click("#ui-layout-save");
  await page.waitForFunction(() => !document.querySelector("#ui-layout-dialog")?.open);
  if (!layoutWrite?.layout.plugins.includes("demo.open")) throw new Error("layout save did not move action into Plugins");
  await page.locator("#plugin-action-list").getByRole("button", { name: "Open demo action" }).waitFor();
  await page.click("#customize-ui");
  const reopened = page.locator("#ui-layout-list .plugin-card").filter({ hasText: "Open demo action" });
  if ((await reopened.locator("select").inputValue()) !== "plugins") throw new Error("saved layout was not restored on reopen");
}

async function testFrontendIntegrationController({ page, origin }) {
  let integrationReads = 0;
  let authHeader = null;
  await page.addInitScript(() => {
    window.__copiedText = null;
    window.__failClipboard = false;
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: async text => {
        if (window.__failClipboard) throw new Error("clipboard unavailable");
        window.__copiedText = text;
      } },
    });
  });
  await page.route("**/admin/integration", route => {
    integrationReads += 1;
    authHeader = route.request().headers().authorization;
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({
      base_url: "http://127.0.0.1:11437/v1",
      api_key_guidance: "any non-empty string",
      curl_example: "curl http://127.0.0.1:11437/v1/models",
      curl_example_windows: "curl.exe http://127.0.0.1:11437/v1/models",
      server: { port: 11437 },
      ollama: { reachable: true, version: "0.6.1", already_registered: false, upstream_add_command: "arc-llama upstream add ollama http://127.0.0.1:11434" },
    }) });
  });
  await page.goto(`${origin}/#system`);
  await page.getByRole("button", { name: "Connect a frontend" }).click();
  await page.waitForFunction(() => document.querySelector("#openwebui-url")?.textContent === "http://127.0.0.1:11437/v1");
  if (authHeader !== "Bearer browser-test-token") throw new Error("integration request omitted admin authentication");
  if ((await page.locator("#openwebui-key-note").innerText()) !== "API Key: any non-empty string") throw new Error("API key guidance was not rendered");
  await page.click("#tab-generic");
  if (await page.locator("#tab-generic").getAttribute("aria-selected") !== "true" || !(await page.locator("#panel-generic").isVisible())) throw new Error("generic tab did not activate accessibly");
  await page.click("#copy-generic-url");
  await page.waitForFunction(() => window.__copiedText === "http://127.0.0.1:11437/v1");
  await page.waitForFunction(() => document.querySelector("#copy-generic-url")?.textContent === "Copied");
  await page.evaluate(() => {
    window.__failClipboard = true;
    document.execCommand = command => {
      window.__fallbackCopiedText = document.querySelector("textarea[readonly]")?.value;
      return command === "copy";
    };
  });
  await page.click("#copy-generic-curl");
  await page.waitForFunction(() => window.__fallbackCopiedText === "curl http://127.0.0.1:11437/v1/models");
  await page.waitForFunction(() => document.querySelector("#copy-generic-curl")?.textContent === "Copied");
  await page.click("#frontend-close");
  await page.getByRole("button", { name: "Connect a frontend" }).click();
  if (integrationReads !== 1) throw new Error("opening the cached integration dialog repeated its request");
  if (!(await page.locator("#ollama-command").textContent()).includes("upstream add ollama")) throw new Error("copyable Ollama guidance was lost");
}

async function testRegenerateAndEdit({ page, origin }) {
  const requests = [];
  await page.route("**/v1/chat/completions", route => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({ status: 200, contentType: "text/event-stream", body: ASSISTANT_REPLY });
  });
  await openChat(page, origin);
  await page.fill("#message-input", "first question");
  await page.press("#message-input", "Enter");
  await page.waitForSelector(".message.assistant .turn-action");
  await page.locator(".message.assistant .turn-action", { hasText: "Regenerate" }).click();
  await page.waitForFunction(() => document.querySelectorAll(".message.assistant .turn-action").length === 1);
  if (requests.length !== 2) throw new Error(`expected 2 completions, got ${requests.length}`);
  const second = requests[1].messages;
  if (second.length !== 1 || second[0].content !== "first question") throw new Error("regenerate resent the wrong transcript");
  if (await page.locator(".message.assistant").count() !== 1) throw new Error("regenerate left the old reply visible");
  await page.locator(".message.user .turn-action", { hasText: "Edit" }).click();
  if (await page.inputValue("#message-input") !== "first question") throw new Error("edit did not restore the prompt");
  if (await page.locator(".message.user, .message.assistant").count() !== 0) throw new Error("edit left the old turn visible");
}

async function testStopKeepsPartialAnswer({ page, origin }) {
  // A stream that sends one chunk and then waits until it is aborted.
  await page.addInitScript(() => {
    const realFetch = window.fetch.bind(window);
    window.fetch = (input, init = {}) => {
      const url = typeof input === "string" ? input : input.url;
      if (!url.endsWith("/v1/chat/completions")) return realFetch(input, init);
      const encoder = new TextEncoder();
      const body = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"Partial answer"}}]}\n\n'));
          init.signal?.addEventListener("abort", () => controller.error(new DOMException("aborted", "AbortError")));
        },
      });
      return Promise.resolve(new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } }));
    };
  });
  await openChat(page, origin);
  await page.fill("#message-input", "tell me a long story");
  await page.press("#message-input", "Enter");
  await page.waitForFunction(() => document.querySelector(".message.assistant .content")?.textContent.includes("Partial answer"));
  if (!(await page.getAttribute("#send-button", "class") || "").includes("stop")) throw new Error("send button did not become a stop button");
  await page.click("#send-button");
  await page.waitForFunction(() => {
    const stopped = document.querySelector(".message.assistant.stopped .stopped-note");
    const send = document.getElementById("send-button");
    const regenerate = [...document.querySelectorAll(".message.assistant .turn-action")]
      .some(action => action.textContent.includes("Regenerate"));
    return stopped && send && !send.classList.contains("stop") && regenerate;
  });
  const text = await page.textContent(".message.assistant .content");
  if (!text.includes("Partial answer")) throw new Error("stopping discarded the partial answer");
  if ((await page.getAttribute("#send-button", "class") || "").includes("stop")) throw new Error("stop mode did not reset");
  if (await page.locator(".message.assistant .turn-action", { hasText: "Regenerate" }).count() !== 1) throw new Error("stopped reply has no Regenerate action");
}

async function testBudgetShownBeforeSending({ page, origin }) {
  await openChat(page, origin);
  await page.fill("#message-input", "x".repeat(40000));
  await page.waitForFunction(() => document.getElementById("ctx-meter").classList.contains("over-budget"));
  const label = await page.textContent("#ctx-label-left");
  if (!label.startsWith("~") || !label.includes("8,192")) throw new Error(`unexpected budget label: ${label}`);
  await page.fill("#message-input", "short");
  await page.waitForFunction(() => !document.getElementById("ctx-meter").classList.contains("over-budget"));
}

async function testPresetShapesRequest({ page, origin }) {
  await page.addInitScript(() => {
    localStorage.setItem("arc-llama-presets", JSON.stringify({ qwen: { system: "Answer like a pirate.", temperature: 0.2 } }));
  });
  const requests = [];
  await page.route("**/v1/chat/completions", route => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({ status: 200, contentType: "text/event-stream", body: ASSISTANT_REPLY });
  });
  await openChat(page, origin);
  await page.fill("#message-input", "ahoy");
  await page.press("#message-input", "Enter");
  await page.waitForSelector(".message.assistant .turn-action");
  const body = requests[0];
  if (body.messages[0].role !== "system" || body.messages[0].content !== "Answer like a pirate.") throw new Error("preset system prompt missing");
  if (body.temperature !== 0.2) throw new Error(`preset temperature missing: ${body.temperature}`);
  if (await page.locator(".message.system").count() !== 0) throw new Error("system prompt should not render as a message");
}

const PNG_1PX = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
  "base64",
);

async function testTextAttachmentExtensionFallback({ page, origin }) {
  const requests = [];
  await page.route("**/v1/chat/completions", route => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({ status: 200, contentType: "text/event-stream", body: ASSISTANT_REPLY });
  });
  await openChat(page, origin);
  await page.setInputFiles("#pdf-input", {
    name: "notes.md", mimeType: "", buffer: Buffer.from("# Field notes\nKeep the blue folder."),
  });
  await page.waitForFunction(() => {
    const chip = document.querySelector(".attachment-chip");
    return chip && !chip.classList.contains("processing") && !chip.classList.contains("error");
  });
  await page.fill("#message-input", "Summarize this file");
  await page.press("#message-input", "Enter");
  await page.waitForSelector(".message.assistant .turn-action");
  const user = requests[0]?.messages.find(message => message.role === "user");
  if (!user?.content.includes("[Attachment: notes.md]") || !user.content.includes("Keep the blue folder.")) {
    throw new Error("text extension fallback did not include the attachment content");
  }
}

async function testImagesNeedVisionModel({ page, origin }) {
  await openChat(page, origin);
  await page.setInputFiles("#pdf-input", { name: "cat.png", mimeType: "image/png", buffer: PNG_1PX });
  await page.waitForSelector(".attachment-chip.error");
  await page.click(".attachment-chip .remove");

  const requests = [];
  await page.route("**/v1/models", route => {
    const models = JSON.parse(JSON.stringify(MODELS));
    models.data[0].metadata.capabilities = ["chat", "completion", "embedding", "vision"];
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(models) });
  });
  await page.route("**/v1/chat/completions", route => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({ status: 200, contentType: "text/event-stream", body: ASSISTANT_REPLY });
  });
  await openChat(page, origin);
  await page.setInputFiles("#pdf-input", { name: "cat.png", mimeType: "image/png", buffer: PNG_1PX });
  await page.waitForFunction(() => {
    const chip = document.querySelector(".attachment-chip");
    return chip && !chip.classList.contains("processing") && !chip.classList.contains("error");
  });
  await page.fill("#message-input", "what is this?");
  await page.press("#message-input", "Enter");
  await page.waitForSelector(".message.user .message-images img");
  await page.waitForSelector(".message.assistant .turn-action");
  const parts = requests[0].messages[0].content;
  if (!Array.isArray(parts) || parts[1]?.type !== "image_url" || !parts[1].image_url.url.startsWith("data:image/png;base64,")) {
    throw new Error("image was not sent as an image_url part");
  }
}

async function testDashboardGenerationTrend({ page, origin }) {
  const now = 1_760_000_000;
  await page.route("**/admin/metrics/history**", route => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      bucket_seconds: 3600,
      points: [
        { t: now - 7200, model: "qwen", metric: "generation_tok_s", median: 26.0, n: 4 },
        { t: now - 3600, model: "qwen", metric: "generation_tok_s", median: 21.5, n: 3 },
        { t: now, model: "qwen", metric: "generation_tok_s", median: 24.8, n: 2, partial: true },
      ],
    }),
  }));
  await page.goto(`${origin}/#system`);
  await page.waitForSelector("#measurements .measure-sparkline polyline");
  const text = await page.locator("#measurements .measure-trend").innerText();
  if (!text.includes("3 hours") || !text.includes("now 24.8 tok/s")) throw new Error(`unexpected trend row: ${text}`);
}

async function testRuntimeCompatibilityGuidance({ page, origin }) {
  const calls = [];
  let outcome = "incompatible";
  let identity = "old-runtime";
  await page.route("**/admin/status", route => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ...ADMIN_STATUS, runtime_compatibility_identity: identity }) }));
  await page.route("**/admin/library/search**", route => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ results: [{ repo: "test/Model-GGUF" }] }) }));
  await page.route("**/admin/library/repo**", route => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ options: [{ file: "model-Q4_K_M.gguf", quant: "Q4_K_M", size_mb: 1024, fit: "fits" }] }) }));
  await page.route("**/admin/library/compatibility", route => {
    calls.push(route.request().postDataJSON());
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: outcome, label: outcome === "incompatible" ? "Requires a different runtime" : "Architecture recognized", detail: "Runtime architecture assessment.", action: "Review runtime requirements.", scope: "Encoding and inference remain unverified.", runtime: "llama-server", runtime_fingerprint: identity, architecture: "llama" }) });
  });
  await page.goto(origin);
  if (calls.length) throw new Error("compatibility check ran automatically");
  await page.fill("#library-query", "model");
  await page.click("#library-search button[type=submit]");
  await page.getByRole("button", { name: "Compare versions" }).click();
  const option = page.locator(".library-option");
  await option.getByRole("button", { name: "Check runtime compatibility" }).click();
  await page.waitForFunction(() => document.querySelector(".library-option .library-compatibility")?.textContent.includes("Requires a different runtime"));
  if (calls[0].repo !== "test/Model-GGUF" || calls[0].file !== "model-Q4_K_M.gguf") throw new Error("checked the wrong model file");
  if (!(await option.innerText()).includes("Likely to fit")) throw new Error("compatibility replaced the memory-fit estimate");
  if (await option.getByRole("button", { name: "Download", exact: true }).isDisabled()) throw new Error("compatibility prevented a deliberate download");
  await option.getByText("What was checked", { exact: true }).click();
  if (!(await option.innerText()).includes("inference remain unverified")) throw new Error("evidence limits were hidden");
  outcome = "recognized";
  await page.locator("#readiness-card").getByRole("button", { name: "Check runtime compatibility" }).click();
  await page.waitForFunction(() => document.querySelector("#readiness-card .library-compatibility")?.textContent.includes("Architecture recognized"));
  if (calls[1].name !== ADMIN_STATUS.models[0].name) throw new Error("local check used the wrong registration");
  await page.evaluate(() => fetchStatus(true));
  if (!(await page.locator("#readiness-card .library-compatibility").innerText()).includes("Architecture recognized")) throw new Error("unchanged polling lost assessment");
  identity = "new-runtime";
  await page.evaluate(() => fetchStatus(true));
  if (!(await page.locator("#readiness-card .library-compatibility").innerText()).includes("has not been checked")) throw new Error("runtime change retained stale assessment");
  if (!(await option.locator(".library-compatibility").innerText()).includes("Runtime changed")) throw new Error("remote result retained a stale runtime assessment");
}

async function testLibraryPublisherAvatars({ page, origin }) {
  const avatarRequests = [];
  await page.route("https://huggingface.co/api/avatars/**", route => {
    avatarRequests.push(route.request().url());
    if (route.request().url().endsWith("/missing")) return route.abort();
    return route.fulfill({ status: 200, contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36"><rect width="36" height="36" fill="green"/></svg>' });
  });
  await page.route("**/admin/library/search**", route => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({ results: [
      { repo: "unsloth/model-GGUF" }, { repo: "missing/model-GGUF" }, { repo: "<script>/model-GGUF" },
    ] }),
  }));
  await page.goto(origin);
  await page.fill("#library-query", "model");
  await page.click("#library-search button[type=submit]");
  await page.waitForFunction(() => document.querySelector(".library-publisher-avatar img")?.naturalWidth > 0);
  await page.waitForFunction(() => document.querySelectorAll(".library-repo")[1]?.querySelector(".library-publisher-avatar img") === null);
  const cards = page.locator(".library-repo");
  if (!(await cards.first().innerText()).includes("Published by unsloth on Hugging Face")) throw new Error("publisher attribution missing");
  if ((await cards.nth(1).locator(".library-publisher-avatar").innerText()) !== "MI") throw new Error("failed image did not retain initials");
  if (await cards.nth(2).locator("img, script").count()) throw new Error("invalid publisher generated active markup or image URL");
  const img = cards.first().locator("img");
  if ((await img.getAttribute("referrerpolicy")) !== "no-referrer") throw new Error("avatar leaked page referrer");
  if (avatarRequests.length !== 2) throw new Error(`unexpected avatar requests: ${avatarRequests}`);
  await page.setViewportSize({ width: 390, height: 844 });
  if (await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)) throw new Error("publisher identity overflowed mobile layout");
}

async function testLibrarySearchAndDownload({ page, origin }) {
  const downloads = [];
  let postDownloadStatusReads = 0;
  let showDownloadedModels = false;
  let includeUnregisteredJob = false;
  await page.route("**/admin/status", route => {
    if (downloads.length && ++postDownloadStatusReads > 1) showDownloadedModels = true;
    const downloaded = showDownloadedModels ? [
      { ...ADMIN_STATUS.models[1], name: "qwen3-q4_k_m", display_name: "Qwen 3 Q4" },
      { ...ADMIN_STATUS.models[1], name: "gemma-q8", display_name: "Gemma Q8" },
    ] : [];
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ...ADMIN_STATUS, models: [...ADMIN_STATUS.models, ...downloaded] }) });
  });
  await page.route("**/admin/library/search**", route => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({ results: [{ repo: "unsloth/<b>Qwen3</b>-GGUF", downloads: 1200, likes: 40 }] }),
  }));
  await page.route("**/admin/library/repo**", route => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({ repo: "unsloth/<b>Qwen3</b>-GGUF", vision: true, vram_mb: 24576, options: [
      { file: "Qwen3-Q4_K_M.gguf", quant: "Q4_K_M", size_mb: 5000, shards: 1, fit: "fits" },
      { file: "Q8/Qwen3-Q8_0-00001-of-00002.gguf", quant: "Q8_0", size_mb: 30000, shards: 2, fit: "too_big" },
    ] }),
  }));
  await page.route("**/admin/library/download", route => {
    downloads.push(route.request().postDataJSON());
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ id: "j1", status: "queued" }) });
  });
  await page.route("**/admin/library/jobs", route => {
    const jobs = downloads.length ? [{ id: "j1", repo: "unsloth/Qwen3-GGUF", file: "Qwen3-Q4_K_M.gguf", status: "done", registered: ["qwen3-q4_k_m", "gemma-q8"], bytes_done: 1, bytes_total: 1, started_at: 1 }] : [];
    if (includeUnregisteredJob) jobs.push({ id: "j2", repo: "unsloth/Other-GGUF", file: "other-Q4.gguf", status: "done", registered: [], bytes_done: 1, bytes_total: 1, started_at: 2 });
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ jobs }) });
  });
  await page.goto(origin);
  await page.fill("#library-query", "qwen3");
  await page.click("#library-search button[type=submit]");
  await page.waitForSelector(".library-repo h3");
  if (await page.locator(".library-repo b").count() !== 0) throw new Error("repo name was rendered as HTML");
  await page.locator(".library-repo button", { hasText: "Compare versions" }).click();
  await page.waitForSelector(".library-option .status-pill");
  const badges = await page.locator(".library-option .status-pill").allInnerTexts();
  if (badges[0] !== "Likely to fit" || badges[1] !== "Likely too large") throw new Error(`unexpected badges ${badges}`);
  const firstRow = await page.locator(".library-option").first().innerText();
  if (!firstRow.includes("4-bit compressed") || !firstRow.includes("4.9 GiB") || !firstRow.includes("less memory")) throw new Error(`unexpected row ${firstRow}`);
  if (await page.locator(".library-technical").first().evaluate(node => node.open)) throw new Error("technical detail was expanded by default");
  await page.locator(".library-technical summary").first().click();
  if (!(await page.locator(".library-technical").first().innerText()).includes("Q4_K_M")) throw new Error("exact encoding missing from expandable details");
  if (!(await page.locator(".library-options").innerText()).includes("extra size is not included")) throw new Error("projector download cost was not explained");
  await page.locator(".library-option button", { hasText: "Download" }).first().click();
  await page.waitForFunction(() => document.querySelector("#library-jobs")?.textContent.includes("downloaded and added"));
  if (downloads.length !== 1 || downloads[0].file !== "Qwen3-Q4_K_M.gguf" || downloads[0].size_mb !== 5000) throw new Error("download request was wrong");
  if (!(await page.locator("#library-jobs").innerText()).includes("Registration is not visible yet")) throw new Error("pending registration was not explained");
  if ((await page.locator(".model-card.selected h3").innerText()) !== "Qwen 3") throw new Error("job completion changed the current model selection");
  await page.locator("#library-jobs").getByRole("button", { name: "Refresh to review" }).click();
  await page.getByRole("button", { name: "Review model qwen3-q4_k_m" }).waitFor();
  await page.getByRole("button", { name: "Review model gemma-q8" }).waitFor();

  const qwenCard = page.locator(".model-card", { has: page.locator("h3", { hasText: "Qwen 3 Q4" }) });
  const details = qwenCard.locator("details");
  await details.locator("summary").click();
  if (!(await details.evaluate(node => node.open))) throw new Error("Advanced details did not open before review");
  await page.getByRole("button", { name: "Review model qwen3-q4_k_m" }).focus();
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelector(".model-card.selected h3")?.textContent === "Qwen 3 Q4" && document.activeElement?.id === "readiness-title");
  if (!page.url().endsWith("/")) throw new Error("review did not expose Models view");
  if ((await page.locator("#chat-primary").getAttribute("href")) !== "/chat?model=qwen3-q4_k_m") throw new Error("review did not set the exact chat model");
  if (!(await page.locator(".model-card.selected details").evaluate(node => node.open))) throw new Error("Advanced details state was lost during review");

  await page.getByRole("button", { name: "Review model gemma-q8" }).click();
  await page.waitForFunction(() => document.querySelector(".model-card.selected h3")?.textContent === "Gemma Q8");
  if ((await page.locator("#chat-primary").getAttribute("href")) !== "/chat?model=gemma-q8") throw new Error("second review action did not select its exact model");
  includeUnregisteredJob = true;
  await page.evaluate(() => pollLibraryJobs({ refreshStatus: false }));
  await page.locator("#library-jobs").getByRole("button", { name: "View your models" }).waitFor();
  await page.locator("#library-jobs").getByRole("button", { name: "View your models" }).click();
  await page.waitForFunction(() => document.activeElement?.id === "choose-title");
  if ((await page.locator("#chat-primary").getAttribute("href")) !== "/chat?model=gemma-q8") throw new Error("generic route guessed a model or changed selection");
}


async function testDashboardNavigationAndBlockedModel({ page, origin }) {
  await page.goto(origin);
  await page.waitForSelector("#model-list .model-card");
  if (!(await page.locator("#library-title").isVisible())) throw new Error("Models view did not expose library");
  if (await page.locator("#system-readiness").isVisible()) throw new Error("System content flashed into default Models view");
  await page.locator('[data-view-link="system"]').click();
  await page.waitForFunction(() => document.querySelector(".dashboard")?.dataset.dashboardView === "system");
  if (!(await page.locator("#plugin-list").isVisible())) throw new Error("System view did not expose plugins");
  await page.goBack();
  await page.waitForFunction(() => document.querySelector(".dashboard")?.dataset.dashboardView === "models");
  await page.goto(`${origin}/#library-title`);
  if (!(await page.locator("#library-title").isVisible())) throw new Error("deep library link did not expose model library");

  let statusModels = [{ ...ADMIN_STATUS.models[1], loaded: false, state: "idle", path: "/models/missing.gguf", file_readiness: { status: "missing", available: false, detail: "Missing file: /models/missing.gguf" } }];
  let statusUnavailable = false;
  await page.route("**/admin/status", route => statusUnavailable
    ? route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "temporarily unavailable" }) })
    : route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ...ADMIN_STATUS, models: statusModels }) }));
  const refresh = () => page.evaluate(() => fetchStatus(true));
  await refresh();
  const brokenCard = page.locator("#model-list .model-card");
  await brokenCard.locator("details summary").click();
  if (!(await brokenCard.locator("details").evaluate(node => node.open))) throw new Error("Advanced details did not open for a broken model");
  await brokenCard.getByRole("button", { name: "Review model Gemma" }).focus();
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelector(".model-card.selected h3")?.textContent === "Gemma" && document.activeElement?.id === "readiness-title");
  if (!(await page.locator("#model-list").innerText()).includes("Missing file")) throw new Error("broken model was not inspectable");
  if (!(await page.locator(".model-card.selected details").evaluate(node => node.open))) throw new Error("Advanced details interaction changed after review");
  if (await page.locator("#chat-primary").getAttribute("href")) throw new Error("reviewing a broken model enabled chat");
  if ((await page.locator("#readiness-card").innerText()).indexOf("/models/missing.gguf") < 0) throw new Error("selected-model panel omitted broken path");

  statusUnavailable = true;
  await page.locator(".model-card.selected .choose-button").click();
  await page.waitForFunction(() => document.querySelector("#model-review-status")?.textContent.includes("Could not refresh") && document.activeElement?.id === "model-review-status");
  if (!(await page.locator("#model-review-status").innerText()).includes("Could not refresh")) throw new Error("failed review refresh had no visible retry message");
  if ((await page.locator(".model-card.selected h3").innerText()) !== "Gemma") throw new Error("failed refresh changed model selection");
  statusUnavailable = false;
  statusModels = [];
  await page.locator(".model-card.selected .choose-button").click();
  await page.waitForFunction(() => document.querySelector("#model-review-status")?.textContent.includes("no longer registered") && document.activeElement?.id === "model-review-status");
  if (!(await page.locator("#model-review-status").innerText()).includes("no longer registered")) throw new Error("missing exact model did not get a visible message");
  if (await page.locator(".model-card.selected").count()) throw new Error("missing model review fell back to another model");

  statusModels = [{ ...ADMIN_STATUS.models[1], loaded: false, state: "idle", file_readiness: { status: "available", available: true, detail: "Files are present and readable" } }];
  await refresh();
  await page.getByRole("button", { name: "Review model Gemma" }).click();
  await page.waitForFunction(() => document.activeElement?.id === "readiness-title");
  if (!(await page.locator("#chat-primary").getAttribute("href"))) throw new Error("reviewed available idle model could not start chat");
  if (!(await page.locator("#model-list").innerText()).includes("Files available")) throw new Error("available idle model was mislabeled");
  for (const state of ["loading", "draining"]) {
    statusModels = [{ ...statusModels[0], state }];
    await refresh();
    if (!(await page.locator("#model-list").innerText()).includes(state === "loading" ? "Loading" : "Draining")) throw new Error(`${state} state was mislabeled`);
  }
  statusModels = [{ ...ADMIN_STATUS.models[1], loaded: true, state: "ready", file_readiness: { status: "missing", available: false, detail: "Missing file: /models/gemma.gguf" } }];
  await refresh();
  if (!(await page.locator("#chat-primary").getAttribute("href"))) throw new Error("loaded model was disabled after its file disappeared");
  if (!(await page.locator("#readiness-card").innerText()).includes("This model remains loaded")) throw new Error("missing file warning did not explain loaded model state");

  statusModels = [];
  await refresh();
  await page.getByRole("button", { name: "Browse model library" }).click();
  if (!page.url().endsWith("#library-title") || !(await page.locator("#library-title").isVisible())) throw new Error("empty model list did not lead to discovery");
  await page.goto(`${origin}/chat`);
  await page.getByRole("link", { name: "System", exact: true }).click();
  await page.waitForFunction(() => document.querySelector(".dashboard")?.dataset.dashboardView === "system");
  await page.reload();
  await page.waitForFunction(() => document.querySelector(".dashboard")?.dataset.dashboardView === "system");
  if (await page.locator("#library-title").isVisible()) throw new Error("System deep link showed Models content");
  await page.locator('[data-view-link="models"]').click();
  await page.waitForFunction(() => document.querySelector(".dashboard")?.dataset.dashboardView === "models");
}

async function testFirstRunJourney({ page, origin }) {
  let status = { ...ADMIN_STATUS, models: [] };
  let jobs = [];
  let jobsUnavailable = false;
  const inference = [];
  await page.route("**/admin/load", route => { inference.push(route.request().url()); return route.abort(); });
  await page.route("**/v1/chat/completions", route => { inference.push(route.request().url()); return route.abort(); });
  await page.route("**/admin/status", route => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(status) }));
  await page.route("**/admin/library/jobs", route => route.fulfill({ status: jobsUnavailable ? 503 : 200, contentType: "application/json", body: JSON.stringify({ jobs }) }));
  const waitTitle = text => page.waitForFunction(text => document.querySelector("#next-step-title")?.textContent === text, text);
  const refreshJobs = () => page.evaluate(() => libraryController.poll({ refreshStatus: false }));
  const refreshStatus = () => page.evaluate(() => fetchStatus(true));
  await page.goto(`${origin}/`);
  await waitTitle("Get your first model");
  const placement = await page.evaluate(() => {
    const library = document.querySelector(".library-section");
    const models = document.querySelector("#choose-title").closest("section");
    return {
      libraryFirst: !!(library.compareDocumentPosition(models) & Node.DOCUMENT_POSITION_FOLLOWING),
      emptyReviewHidden: document.querySelector(".readiness-section").hidden,
      chatInsideReview: document.querySelector(".readiness-section").contains(document.querySelector("#chat-primary")),
    };
  });
  if (!placement.libraryFirst || !placement.emptyReviewHidden || !placement.chatInsideReview) throw new Error("discovery/review layout does not follow the first-run order");
  await page.locator("#next-step-action").click();
  if (!(await page.locator("#library-title").isVisible()) || await page.locator("#library-title").evaluate(node => node !== document.activeElement)) throw new Error("first model action did not focus discovery");
  jobs = [{ id: "new-download", repo: "test/New-GGUF", file: "new.gguf", status: "downloading", bytes_done: 25, bytes_total: 100, started_at: 10 }];
  await refreshJobs(); await waitTitle("Your model download is in progress");
  if (!(await page.locator("#next-step").innerText()).includes("25%")) throw new Error("download progress not visible on Models");
  await page.locator("#next-step-action").focus();
  await refreshStatus();
  if (await page.locator("#next-step-action").evaluate(node => node !== document.activeElement)) throw new Error("unchanged status polling stole next-step action focus");
  jobs[0] = { ...jobs[0], status: "error", error: "Network interrupted" };
  await refreshJobs(); await waitTitle("Your download needs attention");
  if (!(await page.locator("#library-jobs .library-technical").textContent()).includes("Network interrupted")) throw new Error("failed download diagnostics were lost");
  jobs[0] = { ...jobs[0], status: "done", registered: ["exact-new"] };
  await refreshJobs(); await waitTitle("Review your downloaded model");
  if ((await page.locator("#next-step-action").innerText()) !== "Refresh to review") throw new Error("pending registration offered wrong model");
  jobsUnavailable = true;
  await refreshJobs(); await waitTitle("Get your first model");
  if (!(await page.locator("#next-step").innerText()).includes("status is unavailable")) throw new Error("job lookup failure advertised stale completion");
  jobsUnavailable = false;
  await refreshJobs();
  status = { ...status, models: [{ ...ADMIN_STATUS.models[1], name: "exact-new", display_name: "New model", loaded: false, file_readiness: { available: true } }] };
  await page.locator("#next-step-action").click();
  await page.waitForFunction(() => document.querySelector("#next-step-action")?.textContent === "Review New model");
  if (await page.locator(".model-card.selected").count()) throw new Error("download polling automatically selected new model");
  await page.locator("#next-step-action").click();
  await waitTitle("Open chat with New model");
  if ((await page.locator("#next-step-action").getAttribute("href")) !== "/chat?model=exact-new") throw new Error("review did not target exact downloaded registration");
  if (!(await page.locator("#next-step").innerText()).includes("does not verify inference")) throw new Error("estimated readiness overstated inference verification");
  // Opposite arrival order: registry status appears before the completion job.
  jobs = [];
  await page.reload();
  await waitTitle("Open chat with New model");
  jobs = [{ id: "status-first", repo: "test/New-GGUF", file: "new.gguf", status: "done", registered: ["exact-new"], started_at: 11 }];
  await refreshJobs(); await waitTitle("Review your downloaded model");
  if ((await page.locator("#next-step-action").innerText()) !== "Review New model") throw new Error("status-first completion did not offer exact model review");
  if (inference.length) throw new Error("first-run navigation triggered loading or inference");
}

async function testFirstRunBlockers({ page, origin }) {
  let status = { ...ADMIN_STATUS, models: [], gpus: [] };
  let responseCode = 200;
  await page.route("**/admin/status", route => route.fulfill({ status: responseCode, contentType: "application/json", body: JSON.stringify(status) }));
  const waitTitle = text => page.waitForFunction(text => document.querySelector("#next-step-title")?.textContent === text, text);
  await page.goto(`${origin}/`);
  await waitTitle("Check your GPU setup");
  if (!(await page.locator("#next-step").innerText()).includes("arc-llama doctor")) throw new Error("GPU blocker had no actionable diagnostic");
  await page.locator("#next-step-action").click();
  await page.waitForFunction(() => document.querySelector(".dashboard")?.dataset.dashboardView === "system");
  await page.locator('[data-view-link="models"]').click();
  status = { ...ADMIN_STATUS, models: [{ ...ADMIN_STATUS.models[1], file_readiness: { available: false, detail: "Missing file: /models/gemma.gguf" } }] };
  await page.locator("#refresh").click();
  await page.locator(".model-card").first().click();
  await waitTitle("Restore or replace this model's files");
  if (await page.locator("#chat-primary").getAttribute("href")) throw new Error("missing file enabled chat");
  await page.locator("#next-step-action").click();
  if (await page.locator("#readiness-title").evaluate(node => node !== document.activeElement)) throw new Error("missing-file action did not focus review");
  status = { ...status, gpus: [], models: [{ ...status.models[0], loaded: true }] };
  await page.locator("#refresh").click(); await waitTitle("Open chat with Gemma");
  if (!(await page.locator("#chat-primary").getAttribute("href"))) throw new Error("loaded model disabled after source or GPU configuration disappeared");
  responseCode = 503;
  await page.locator("#refresh").click(); await waitTitle("Reconnect to Arc Llama");
  if (await page.locator("#chat-primary").getAttribute("href") || await page.locator("#next-step a").count()) throw new Error("offline snapshot exposed stale chat action");
  responseCode = 401;
  await page.locator("#next-step-action").click();
  await page.waitForFunction(() => document.querySelector("#next-step")?.textContent.includes("Local admin access is unavailable"));
  responseCode = 200;
  await page.locator("#next-step-action").click(); await waitTitle("Open chat with Gemma");
  if (!(await page.locator("#chat-primary").getAttribute("href"))) throw new Error("recovery did not restore chat");
}

async function testModelChoiceExplanations({ page, origin }) {
  await page.route("**/admin/library/search**", route => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ results: [{ repo: "test/Choices-GGUF" }] }) }));
  await page.route("**/admin/library/repo**", route => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ vram_mb: null, vision: false, options: [
    { file: "unknown-<img>.gguf", quant: "unknown", size_mb: 0, shards: 1, fit: "unknown" },
    { file: "small-IQ3_S.gguf", quant: "IQ3_S", size_mb: 2000, shards: 1, fit: "tight" },
    { file: "medium-Q6_K.gguf", quant: "Q6_K", size_mb: 4000, shards: 1, fit: "fits" },
    { file: "large-BF16.gguf", quant: "BF16", size_mb: 10000, shards: 1, fit: "too_big" },
  ] }) }));
  await page.goto(origin);
  await page.fill("#library-query", "choices");
  await page.locator("#library-search button").click();
  await page.getByRole("button", { name: "Compare versions" }).click();
  await page.waitForSelector(".library-technical");
  const choices = await page.locator(".library-options").innerText();
  for (const phrase of ["Compression not reported", "Size not reported", "Fit unknown", "GPU memory capacity is unavailable", "3-bit compressed", "Limited memory headroom", "shorter conversation", "6-bit compressed", "middle ground", "16-bit precision", "larger download"]) {
    if (!choices.includes(phrase)) throw new Error(`missing plain-language explanation: ${phrase}`);
  }
  if (await page.locator(".library-options img").count()) throw new Error("remote filename was rendered as HTML");
  if (await page.locator(".library-technical").evaluateAll(nodes => nodes.some(node => node.open))) throw new Error("technical details were expanded automatically");
  await page.locator(".library-technical summary").first().click();
  if (!(await page.locator(".library-technical").first().innerText()).includes("unknown-<img>.gguf")) throw new Error("exact unknown filename not preserved safely");
  if ((await page.locator(".library-option > button").count()) !== 4) throw new Error("model options lack clear download actions");
  await page.setViewportSize({ width: 390, height: 844 });
  if (await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)) throw new Error("mobile model choices overflow horizontally");
}

async function testPollingEfficiency({ page, origin }) {
  let status = JSON.parse(JSON.stringify(ADMIN_STATUS));
  let statusReads = 0;
  let jobReads = 0;
  let failStatus = false;
  let jobs = [];
  await page.route("**/admin/status", route => { statusReads++; return route.fulfill({ status: failStatus ? 503 : 200, contentType: "application/json", body: JSON.stringify(status) }); });
  await page.route("**/admin/library/jobs", route => { jobReads++; return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ jobs }) }); });
  await page.goto(origin);
  await page.waitForFunction(() => document.querySelector(".model-card") && typeof libraryController !== "undefined");
  await page.evaluate(async () => { await fetchStatus(); await pollLibraryJobs(); });
  await page.locator(".model-card").first().locator("details summary").click();
  await page.locator(".model-card").first().locator(".choose-button").focus();
  await page.evaluate(() => { window.optimizationCard = document.querySelector(".model-card"); });
  const before = statusReads;
  await page.evaluate(() => Promise.all([fetchStatus(), fetchStatus(), fetchStatus()]));
  if (statusReads !== before + 1) throw new Error("concurrent background status reads did not coalesce");
  const unchanged = await page.evaluate(() => ({ same: window.optimizationCard === document.querySelector(".model-card"), open: document.querySelector(".model-card details").open, focused: document.activeElement.classList.contains("choose-button") }));
  if (!unchanged.same || !unchanged.open || !unchanged.focused) throw new Error("unchanged poll rebuilt the card or lost interaction state");
  status.models[0].ctx = 16384;
  await page.evaluate(() => fetchStatus());
  if (!(await page.locator(".model-card").first().innerText()).includes("16,384")) throw new Error("changed model configuration did not render");
  if (!(await page.locator(".model-card details").first().evaluate(node => node.open))) throw new Error("changed poll closed Advanced details");
  if (!(await page.locator(".model-card .choose-button").first().evaluate(node => node === document.activeElement))) throw new Error("changed poll lost keyboard focus");
  jobs = [{ id: "completion", repo: "test/Qwen", file: "qwen.gguf", status: "done", registered: ["qwen"], started_at: 1 }];
  const initialCompletionReads = statusReads;
  await page.evaluate(() => pollLibraryJobs());
  if (statusReads !== initialCompletionReads + 1) throw new Error("newly completed download did not refresh registration once");
  const afterCompletionReads = statusReads;
  const afterCompletionJobs = jobReads;
  await page.evaluate(() => Promise.all([pollLibraryJobs(), pollLibraryJobs(), pollLibraryJobs()]));
  if (statusReads !== afterCompletionReads || jobReads !== afterCompletionJobs + 1) throw new Error("handled completion caused repeated status reads or overlapping job reads");
  jobs.push({ id: "retry-completion", repo: "test/Gemma", file: "gemma.gguf", status: "done", registered: ["gemma"], started_at: 2 });
  failStatus = true;
  await page.evaluate(() => pollLibraryJobs());
  const failedReads = statusReads;
  failStatus = false;
  await page.evaluate(() => pollLibraryJobs());
  if (statusReads !== failedReads + 1) throw new Error("failed completion refresh was incorrectly considered handled");
  const recoveredReads = statusReads;
  await page.evaluate(() => pollLibraryJobs());
  if (statusReads !== recoveredReads) throw new Error("successful retry kept refreshing completed jobs");
}

async function testQueuedStatusRefresh({ page, origin }) {
  let reads = 0;
  let hold = false;
  let release;
  let signalStarted;
  let status = JSON.parse(JSON.stringify(ADMIN_STATUS));
  const started = new Promise(resolve => { signalStarted = resolve; });
  await page.route("**/admin/status", async route => {
    reads++;
    const response = JSON.stringify(status);
    if (hold) { signalStarted(); await new Promise(resolve => { release = resolve; }); }
    return route.fulfill({ status: 200, contentType: "application/json", body: response });
  });
  await page.goto(origin);
  await page.waitForSelector(".model-card");
  await page.evaluate(() => fetchStatus());
  await page.locator(".model-card .choose-button").first().focus();
  hold = true;
  const before = reads;
  const pending = page.evaluate(() => Promise.all([fetchStatus(), fetchStatus(), fetchStatus(true), fetchStatus(true)]));
  await started;
  if (reads !== before + 1) throw new Error("forced refresh raced the older in-flight response");
  await page.locator("#library-query").focus();
  status.models[0].ctx = 32768;
  hold = false;
  release();
  await pending;
  if (reads !== before + 2) throw new Error("forced refreshes did not share one fresh follow-up read");
  if (!(await page.locator("#readiness-card").innerText()).includes("32,768")) throw new Error("older status overwrote the fresh follow-up");
  if (!(await page.locator("#library-query").evaluate(node => node === document.activeElement))) throw new Error("slow poll restored stale focus after the user moved elsewhere");
  // A download started during an older jobs read must get a fresh follow-up.
  let jobReads = 0;
  let holdJobs = true;
  let releaseJobs;
  let signalJobs;
  let jobs = [];
  const jobsStarted = new Promise(resolve => { signalJobs = resolve; });
  await page.route("**/admin/library/jobs", async route => {
    jobReads++;
    const response = JSON.stringify({ jobs });
    if (holdJobs) { signalJobs(); await new Promise(resolve => { releaseJobs = resolve; }); }
    return route.fulfill({ status: 200, contentType: "application/json", body: response });
  });
  const jobPending = page.evaluate(() => Promise.all([pollLibraryJobs(), pollLibraryJobs({ force: true }), pollLibraryJobs({ force: true })]));
  await jobsStarted;
  if (jobReads !== 1) throw new Error("new-download refresh raced an older jobs read");
  jobs = [{ id: "fresh-download", repo: "test/Qwen", file: "qwen.gguf", status: "done", registered: ["qwen"], started_at: 1 }];
  holdJobs = false;
  releaseJobs();
  await jobPending;
  if (jobReads !== 2 || !(await page.locator("#library-jobs").innerText()).includes("downloaded and added")) throw new Error("fresh follow-up missed the newly completed download");
}

async function testHiddenTabPolling({ page, origin }) {
  await page.addInitScript(() => {
    window.optimizationHidden = false;
    window.optimizationDelays = [];
    Object.defineProperty(document, "hidden", { configurable: true, get: () => window.optimizationHidden });
    const nativeTimeout = window.setTimeout.bind(window);
    window.setTimeout = (callback, delay, ...args) => { window.optimizationDelays.push(delay); return nativeTimeout(callback, delay, ...args); };
  });
  const counts = { status: 0, jobs: 0, metrics: 0, history: 0 };
  await page.route("**/admin/status", route => { counts.status++; return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(ADMIN_STATUS) }); });
  await page.route("**/admin/library/jobs", route => { counts.jobs++; return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ jobs: [{ id: "active", repo: "test/Qwen", file: "qwen.gguf", status: "downloading", bytes_done: 1, bytes_total: 10, started_at: 1 }] }) }); });
  await page.route("**/admin/metrics", route => { counts.metrics++; return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(METRICS) }); });
  await page.route("**/admin/metrics/history**", route => { counts.history++; return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ points: [] }) }); });
  await page.goto(origin);
  await page.waitForFunction(() => typeof statusPolling !== "undefined" && statusPolling);
  await page.evaluate(async () => { await Promise.all([fetchStatus(), pollLibraryJobs(), fetchMeasurements()]); });
  const before = { ...counts };
  const delays = await page.evaluate(() => { window.optimizationHidden = true; window.optimizationDelays = []; document.dispatchEvent(new Event("visibilitychange")); return window.optimizationDelays; });
  if (delays.length !== 3 || delays.some(delay => delay !== 30000)) throw new Error(`hidden polling did not slow all three controllers: ${delays}`);
  if (Object.keys(counts).some(key => counts[key] !== before[key])) throw new Error("hiding the tab triggered unnecessary requests");
  await page.evaluate(async () => { window.optimizationHidden = false; document.dispatchEvent(new Event("visibilitychange")); await Promise.all([fetchStatus(), pollLibraryJobs(), fetchMeasurements()]); });
  for (const key of Object.keys(counts)) if (counts[key] !== before[key] + 1) throw new Error(`returning to the tab did not perform exactly one ${key} refresh`);
  const activeDelays = await page.evaluate(() => window.optimizationDelays.slice(3));
  for (const delay of [2000, 5000, 15000]) if (!activeDelays.includes(delay)) throw new Error(`active polling cadence not restored: ${delay}`);
}

async function testDownloadRecovery({ page, origin }) {
  let jobs = [{ id: "failed", repo: "test/Recover", file: "nested/model-Q4.gguf", status: "error", error: "Connection interrupted <script>bad()</script>", bytes_total: 5000 * 1048576, started_at: 1 }];
  let rejectRetry = true;
  let releaseRejected;
  let signalRejected;
  const rejectedStarted = new Promise(resolve => { signalRejected = resolve; });
  let releaseRetry;
  let signalRetry;
  const startedRetry = new Promise(resolve => { signalRetry = resolve; });
  const submissions = [];
  await page.route("**/admin/library/jobs", route => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ jobs }) }));
  await page.route("**/admin/library/download", async route => {
    submissions.push(route.request().postDataJSON());
    if (rejectRetry) {
      signalRejected();
      await new Promise(resolve => { releaseRejected = resolve; });
      return route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Try again later" }) });
    }
    signalRetry();
    await new Promise(resolve => { releaseRetry = resolve; });
    jobs = [{ ...jobs[0], id: "retried", status: "queued", error: null, started_at: 2 }, ...jobs];
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(jobs[0]) });
  });
  await page.goto(origin);
  await page.waitForSelector("#library-jobs .library-technical");
  await page.fill("#library-query", "keep my search");
  await page.locator(".model-card").nth(1).click();
  const originalSelection = await page.evaluate(() => selectedModel);
  if (!(await page.locator("#library-jobs .library-technical").isVisible()) || await page.locator("#library-jobs script").count()) throw new Error("unsafe download diagnostic rendering");
  if (await page.locator("#library-jobs details").evaluate(node => node.open)) throw new Error("download diagnostics expanded automatically");
  await page.locator("#library-jobs details summary").click();
  if (!(await page.locator("#library-jobs pre").innerText()).includes("<script>bad()</script>")) throw new Error("exact error not preserved safely");
  const retry = page.locator("#library-jobs button").filter({ hasText: "Retry download" });
  await retry.click();
  await rejectedStarted;
  await page.evaluate(() => pollLibraryJobs({ refreshStatus: false }));
  if (!(await retry.isDisabled())) throw new Error("polling replaced a busy retry with an enabled button");
  releaseRejected();
  await page.waitForFunction(() => document.querySelector("#library-recovery-status")?.textContent.includes("Could not start"));
  await page.waitForFunction(() => !document.querySelector('#library-jobs button[aria-label^="Retry download"]')?.disabled);
  if (await retry.isDisabled()) throw new Error("failed retry did not re-enable its replaced button");
  rejectRetry = false;
  const retryPending = page.evaluate(() => Promise.all([libraryController.retry(libraryJobs[0]), libraryController.retry(libraryJobs[0])]));
  await startedRetry;
  if (submissions.length !== 2) throw new Error("duplicate retries submitted overlapping downloads");
  releaseRetry();
  await retryPending;
  if (JSON.stringify(submissions[1]) !== JSON.stringify({ repo: "test/Recover", file: "nested/model-Q4.gguf", size_mb: 5000 })) throw new Error("retry changed the requested file or size");
  if ((await page.locator("#library-query").inputValue()) !== "keep my search" || (await page.evaluate(() => selectedModel)) !== originalSelection) throw new Error("retry lost the search or model selection");
  if ((await page.locator("#library-jobs button").filter({ hasText: "Retry download" }).count()) !== 0) throw new Error("superseded failure still offers another retry");
  jobs[0] = { ...jobs[0], status: "done", registered: ["gemma"] };
  await page.evaluate(() => pollLibraryJobs());
  if (!(await page.locator("#library-recovery-status").innerText()).includes("Download finished")) throw new Error("queued retry notice did not reflect completion");
  await page.locator("#library-jobs").getByRole("button", { name: "Review model gemma" }).click();
  if ((await page.locator("#chat-primary").getAttribute("href")) !== "/chat?model=gemma") throw new Error("successful retry did not lead to the exact model review");
}

async function testMissingFileRecovery({ page, origin }) {
  let scans = 0;
  const model = { ...ADMIN_STATUS.models[1], ctx: 8192, file_readiness: { available: false, detail: "Missing file: /models/gemma.gguf" } };
  await page.route("**/admin/status", route => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ...ADMIN_STATUS, models: [model] }) }));
  await page.route("**/admin/scan", route => {
    scans++;
    model.file_readiness = { available: true, detail: "File restored" };
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ found: 1, added: [] }) });
  });
  await page.goto(origin);
  await page.waitForSelector("#readiness-card .toolbar");
  await page.fill("#library-query", "my replacement query");
  await page.getByRole("button", { name: "Find replacement", exact: true }).click();
  if (!(await page.locator("#library-title").isVisible()) || (await page.locator("#library-query").inputValue()) !== "my replacement query") throw new Error("replacement action did not preserve search");
  await page.getByRole("button", { name: "Scan again", exact: true }).click();
  await page.waitForFunction(() => document.querySelector("#chat-primary")?.getAttribute("href") === "/chat?model=gemma");
  if (scans !== 1 || (await page.evaluate(() => selectedModel)) !== "gemma") throw new Error("file recovery changed selection or repeated scanning");
  if (!(await page.locator("#readiness-card").innerText()).includes("8,192")) throw new Error("file recovery discarded existing model settings");
  if ((await page.locator("#library-query").inputValue()) !== "my replacement query") throw new Error("scanning discarded the search");
}

const TESTS = [
  ["recovery-download", testDownloadRecovery],
  ["recovery-missing-file", testMissingFileRecovery],
  ["polling-efficiency", testPollingEfficiency],
  ["polling-queued-status", testQueuedStatusRefresh],
  ["polling-hidden-tab", testHiddenTabPolling],
  ["model-choice-explanations", testModelChoiceExplanations],
  ["first-run-journey", testFirstRunJourney],
  ["first-run-blockers", testFirstRunBlockers],
  ["chat-selection", testChatSelection],
  ["chat-history-components", testChatHistoryComponents],
  ["chat-send", testChatSend],
  ["structured-load-failure", testStructuredLoadFailure],
  ["refresh-preserves-edits", testRefreshPreservesEdits],
  ["keyboard-navigation", testKeyboardNavigation],
  ["settings-survive-status-poll", testSettingsSurviveStatusPoll],
  ["retry-original-turn-once", testRetrySendsOriginalTurnOnce],
  ["dashboard-measurements", testDashboardMeasurements],
  ["memory-fit-and-saved-settings", testMemoryFitAndSavedSettings],
  ["dashboard-fit-and-empty-measurements", testDashboardFitAndEmptyMeasurements],
  ["dashboard-navigation-and-blocked-model", testDashboardNavigationAndBlockedModel],
  ["dashboard-generation-trend", testDashboardGenerationTrend],
  ["runtime-compatibility-guidance", testRuntimeCompatibilityGuidance],
  ["library-publisher-avatars", testLibraryPublisherAvatars],
  ["library-search-and-download", testLibrarySearchAndDownload],
  ["dashboard-plugin-layout-controller", testDashboardPluginLayoutController],
  ["frontend-integration-controller", testFrontendIntegrationController],
  ["regenerate-and-edit", testRegenerateAndEdit],
  ["stop-keeps-partial-answer", testStopKeepsPartialAnswer],
  ["budget-before-sending", testBudgetShownBeforeSending],
  ["preset-shapes-request", testPresetShapesRequest],
  ["text-attachment-extension-fallback", testTextAttachmentExtensionFallback],
  ["images-need-vision-model", testImagesNeedVisionModel],
];

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

function parseFilter(argv) {
  const idx = argv.indexOf("--only");
  if (idx === -1 || idx + 1 >= argv.length) return null;
  return argv[idx + 1];
}

async function main() {
  const filter = parseFilter(process.argv);
  const { server, port } = await startStaticServer();
  const origin = `http://127.0.0.1:${port}`;
  const launchOptions = { headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"] };
  let browser;
  const results = [];
  try {
    browser = await chromium.launch({ ...launchOptions, channel: "chrome" }).catch(() =>
      chromium.launch(launchOptions)
    );
    for (const [name, fn] of TESTS) {
      if (filter && !name.includes(filter)) continue;
      const context = await browser.newContext();
      await installApiMocks(context);
      const page = await context.newPage();
      page.setDefaultTimeout(5000);
      // Chromium cold navigation can outlast interaction waits on busy hosts.
      page.setDefaultNavigationTimeout(30000);
      const pageErrors = [];
      page.on("pageerror", error => pageErrors.push(error.message));
      recordSessionTokenCalls(page);
      const started = Date.now();
      try {
        await fn({ page, origin });
        if (pageErrors.length) throw new Error(pageErrors.join("; "));
        results.push({ name, ok: true, ms: Date.now() - started });
        console.log(`ok   ${name} (${Date.now() - started}ms)`);
      } catch (err) {
        results.push({ name, ok: false, ms: Date.now() - started, error: String(err && err.message || err) });
        console.error(`FAIL ${name}: ${err && err.message ? err.message : err}`);
      } finally {
        await context.close().catch(() => {});
      }
    }
  } catch (err) {
    console.error(`fatal: ${err && err.message ? err.message : err}`);
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close().catch(() => {});
    server.close();
  }
  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} passed`);
  if (failed.length) process.exitCode = 1;
}

main();