// Existing regression cases; shared fixtures and browser setup live in harness.mjs.
import { ADMIN_STATUS } from "./harness.mjs";

export async function testRuntimeCompatibilityGuidance({ page, origin }) {
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

export async function testLibraryPublisherAvatars({ page, origin }) {
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

export async function testLibrarySearchAndDownload({ page, origin }) {
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


export async function testModelChoiceExplanations({ page, origin }) {
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

export async function testDownloadRecovery({ page, origin }) {
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

export async function testMissingFileRecovery({ page, origin }) {
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



export async function testCompatibilityAbandonedChecks({ page, origin }) {
  let identity = "old-runtime";
  await page.route("**/admin/status", route => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ...ADMIN_STATUS, runtime_compatibility_identity: identity }) }));
  await page.route("**/admin/library/search**", route => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ results: [{ repo: "test/Model-GGUF" }] }) }));
  await page.route("**/admin/library/repo**", route => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ options: [{ file: "model-Q4.gguf", quant: "Q4", size_mb: 1024, fit: "fits" }] }) }));
  await page.goto(origin);
  await page.fill("#library-query", "model");
  await page.click("#library-search button[type=submit]");
  await page.getByRole("button", { name: "Compare versions" }).click();
  await page.evaluate(() => {
    const fetch = window.fetch.bind(window);
    const setTimeout = window.setTimeout.bind(window);
    window.compatAttempts = [];
    window.compatAborts = 0;
    window.fetch = (url, options) => {
      if (url !== "/admin/library/compatibility") return fetch(url, options);
      return new Promise((resolve, reject) => {
        const number = window.compatAttempts.length;
        window.compatAttempts.push(resolve);
        options.signal.addEventListener("abort", () => {
          window.compatAborts++;
          if (number === 0) reject(new DOMException("Aborted", "AbortError"));
        });
        if (number >= 2) resolve(new Response(JSON.stringify({ status: "recognized", label: "Architecture recognized", detail: "Current result.", action: "Review requirements." })));
      });
    };
    // Only the first compatibility deadline is accelerated. Other app timers
    // retain their normal cadence; the subsequent stale request ignores abort.
    window.setTimeout = (fn, ms, ...args) => setTimeout(fn, ms === 30000 && !window.compatAttempts.length ? 20 : ms, ...args);
  });
  const host = page.locator(".library-option .library-compatibility");
  const check = host.getByRole("button", { name: "Check runtime compatibility" });
  await check.click();
  await page.waitForFunction(() => document.querySelector(".library-option .library-compatibility").textContent.includes("timed out"));
  if (await check.isDisabled()) throw new Error("deadline left the check disabled");
  await check.click();
  identity = "new-runtime";
  await page.evaluate(() => fetchStatus(true));
  if (!(await host.innerText()).includes("Runtime changed") || await check.isDisabled()) throw new Error("runtime change did not cancel/reset pending check");
  await check.click();
  await page.waitForFunction(() => document.querySelector(".library-option .library-compatibility").textContent.includes("Current result"));
  await page.evaluate(() => window.compatAttempts[1](new Response(JSON.stringify({ status: "incompatible", label: "Old result", detail: "Stale result." }))));
  if (!(await host.innerText()).includes("Current result") || (await host.innerText()).includes("Stale result")) throw new Error("abandoned response overwrote the current result");
  if (await page.evaluate(() => window.compatAborts) !== 2) throw new Error("timeout/runtime change did not abort fetches");
}
