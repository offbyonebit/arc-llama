// Existing regression cases; shared fixtures and browser setup live in harness.mjs.
import { ADMIN_STATUS, METRICS } from "./harness.mjs";

export async function testDashboardMeasurements({ page, origin }) {
  await page.goto(`${origin}/#system`);
  await page.waitForFunction(() => document.querySelector("#measurements")?.textContent.includes("24.8"));
  const content = await page.locator("#measurements").innerText();
  if (!content.includes("tok/s") || content.includes("n/a")) throw new Error("generation rates have incorrect units or values");
}

export async function testDashboardFitAndEmptyMeasurements({ page, origin }) {
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

export async function testDashboardPluginLayoutController({ page, origin }) {
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

export async function testFrontendIntegrationController({ page, origin }) {
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

export async function testDashboardGenerationTrend({ page, origin }) {
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

export async function testDashboardNavigationAndBlockedModel({ page, origin }) {
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

export async function testFirstRunJourney({ page, origin }) {
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

export async function testFirstRunBlockers({ page, origin }) {
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

export async function testPollingEfficiency({ page, origin }) {
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

export async function testQueuedStatusRefresh({ page, origin }) {
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

export async function testHiddenTabPolling({ page, origin }) {
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
