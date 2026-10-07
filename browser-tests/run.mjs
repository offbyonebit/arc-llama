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
  await page.goto(origin);
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
  const empty = await page.locator("#measurements").innerText();
  if (!empty.includes("No measurements yet") || empty.includes("24.8")) throw new Error("dashboard retained stale rates in empty state");
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
  await page.waitForSelector(".message.assistant.stopped .stopped-note");
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
  await page.goto(origin);
  await page.waitForSelector("#measurements .measure-sparkline polyline");
  const text = await page.locator("#measurements .measure-trend").innerText();
  if (!text.includes("3 hours") || !text.includes("now 24.8 tok/s")) throw new Error(`unexpected trend row: ${text}`);
}

const TESTS = [
  ["chat-selection", testChatSelection],
  ["chat-send", testChatSend],
  ["structured-load-failure", testStructuredLoadFailure],
  ["refresh-preserves-edits", testRefreshPreservesEdits],
  ["keyboard-navigation", testKeyboardNavigation],
  ["settings-survive-status-poll", testSettingsSurviveStatusPoll],
  ["retry-original-turn-once", testRetrySendsOriginalTurnOnce],
  ["dashboard-measurements", testDashboardMeasurements],
  ["memory-fit-and-saved-settings", testMemoryFitAndSavedSettings],
  ["dashboard-fit-and-empty-measurements", testDashboardFitAndEmptyMeasurements],
  ["dashboard-generation-trend", testDashboardGenerationTrend],
  ["regenerate-and-edit", testRegenerateAndEdit],
  ["stop-keeps-partial-answer", testStopKeepsPartialAnswer],
  ["budget-before-sending", testBudgetShownBeforeSending],
  ["preset-shapes-request", testPresetShapesRequest],
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