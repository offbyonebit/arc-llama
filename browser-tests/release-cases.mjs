import { choosePaperModel, openContext } from "./harness.mjs";
import assert from "node:assert/strict";
import { openChat, ASSISTANT_REPLY } from "./harness.mjs";

const json = body => ({ status: 200, contentType: "application/json", body: JSON.stringify(body) });

async function savedChat(page) {
  let chat = { id: "release-chat", title: "Release chat", messages: [], compaction: null, created_at: 1, updated_at: 1 };
  await page.route("**/v1/chats", route => route.fulfill(json(route.request().method() === "POST" ? chat : { data: [chat] })));
  await page.route("**/v1/chats/release-chat", route => {
    const req = route.request();
    if (req.method() === "PUT") chat = { ...chat, ...req.postDataJSON() };
    if (req.method() === "PATCH") chat.messages.push(...(req.postDataJSON().messages || []));
    return route.fulfill(json(chat));
  });
  return () => chat;
}

export async function testInterruptedAnswerPersists({ page, origin }) {
  const chat = await savedChat(page);
  let eof = false;
  await page.route("**/v1/chat/completions", route => route.fulfill({ status: 200, contentType: "text/event-stream",
    body: 'data: {"choices":[{"delta":{"content":"Keep this partial answer"}}]}\n\n' +
      (eof ? "" : 'data: {"error":{"code":"inference_interrupted","message":"Model stalled"}}\n\n') }));
  await openChat(page, origin);
  for (const mode of ["error event", "EOF"]) {
    await page.fill("#message-input", mode);
    await page.press("#message-input", "Enter");
    await page.locator(".message.assistant .turn-action").last().waitFor();
    assert.equal(chat().messages.at(-1).status, "interrupted");
    assert.equal(chat().messages.at(-1).content, "Keep this partial answer");
    eof = true;
  }
  await page.reload();
  await page.click("#history-toggle");
  await page.locator('.h-card[data-id="release-chat"]').click();
  await page.locator(".message.assistant .stopped-note").first().waitFor();
  assert.equal(await page.locator(".message.assistant .stopped-note").count(), 2);
  assert.match(await page.locator(".message.assistant .stopped-note").first().innerText(), /Interrupted/);
  await page.unroute("**/v1/chat/completions");
  await page.fill("#message-input", "continue now");
  await page.press("#message-input", "Enter");
  await page.waitForFunction(() => [...document.querySelectorAll(".message.assistant .content")].some(n => n.textContent.includes("Hello from arc-llama")));
}

export async function testStoppedAnswerPersists({ page, origin }) {
  const chat = await savedChat(page);
  await page.addInitScript(() => {
    const real = window.fetch.bind(window);
    window.fetch = (url, options = {}) => {
      if (!String(url).endsWith("/v1/chat/completions")) return real(url, options);
      const body = new ReadableStream({ start(controller) {
        controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"reasoning_content":"Keep reasoning","content":"Keep partial"}}]}\n\n'));
        options.signal?.addEventListener("abort", () => controller.error(new DOMException("stopped", "AbortError")));
      } });
      return Promise.resolve(new Response(body));
    };
  });
  await openChat(page, origin);
  await page.fill("#message-input", "long answer");
  await page.press("#message-input", "Enter");
  await page.waitForFunction(() => document.querySelector(".message.assistant .content")?.textContent.includes("Keep partial"));
  await page.click("#send-button");
  await page.locator(".message.assistant .turn-action").waitFor();
  assert.equal(chat().messages.at(-1).status, "stopped");
  assert.equal(chat().messages.at(-1).thinking, "Keep reasoning");
  await page.reload();
  await page.click("#history-toggle");
  await page.locator('.h-card[data-id="release-chat"]').click();
  await page.locator(".message.assistant .stopped-note").waitFor();
  assert.equal(await page.locator(".stopped-note").innerText(), "Stopped");
}

export async function testExactBudgetAndReplyLimit({ page, origin }) {
  const budgets = [], generations = [];
  await page.route("**/v1/chat/budget", route => {
    const body = route.request().postDataJSON();
    budgets.push(body);
    return route.fulfill(json({ prompt_tokens: 60, reserved_tokens: body.max_tokens, total_tokens: 60 + body.max_tokens,
      context_tokens: 512, exact: true, over_budget: 60 + body.max_tokens > 512, reason: "Resident tokenizer" }));
  });
  await page.route("**/v1/chat/completions", route => {
    generations.push(route.request().postDataJSON());
    return route.fulfill({ status: 200, contentType: "text/event-stream", body: ASSISTANT_REPLY });
  });
  await openChat(page, origin);
  await page.fill("#message-input", "日本語 ```python``` ");
  await page.waitForFunction(() => document.querySelector("#ctx-label-left").textContent.startsWith("1,084 / 512"));
  await page.press("#message-input", "Enter");
  await page.locator(".budget-warning").waitFor();
  assert.equal(generations.length, 0);
  await page.getByRole("button", { name: "Adjust context or reply limit" }).click();
  await page.fill("#s-system", "Keep the user's code.");
  await page.fill("#s-max-tokens", "128");
  await page.click("#s-preset-save");
  await page.click("#settings-toggle");
  await page.getByRole("button", { name: "Retry message", exact: true }).click();
  await page.locator(".message.assistant .turn-action").waitFor();
  assert.equal(generations.length, 1);
  await openContext(page);
  assert.match(await page.locator("#ctx-label-left").innerText(), /512/);
  assert.equal(generations[0].max_tokens, 128);
  assert.equal(generations[0].messages[0].role, "system");
  const last = budgets.at(-1);
  assert.equal(last.max_tokens, generations[0].max_tokens);
  assert.deepEqual(last.messages, generations[0].messages);
}

export async function testBudgetDiscardsStaleResponses({ page, origin }) {
  await page.route("**/v1/chat/budget", async route => {
    const body = route.request().postDataJSON();
    if (body.model === "qwen") await new Promise(resolve => setTimeout(resolve, 900));
    await route.fulfill(json({ prompt_tokens: 7, reserved_tokens: 1024, total_tokens: 1031,
      context_tokens: body.model === "qwen" ? 1111 : 2222, exact: true, over_budget: false, reason: "Counted" })).catch(() => {});
  });
  await openChat(page, origin);
  await page.fill("#message-input", "first");
  await page.waitForTimeout(450);
  await choosePaperModel(page, "gemma");
  await page.fill("#message-input", "second");
  await page.waitForFunction(() => document.querySelector("#ctx-label-left").textContent.includes("2,222"));
  await page.waitForTimeout(1000);
  await openContext(page);
  assert.match(await page.locator("#ctx-label-left").innerText(), /2,222/);
}

export async function testDownloadFolderAndRetry({ page, origin }) {
  const downloads = [];
  let jobs = [];
  await page.route("**/admin/library/search**", route => route.fulfill(json({ results: [{ repo: "test/Model" }] })));
  await page.route("**/admin/library/repo**", route => route.fulfill(json({ repo: "test/Model", options: [{ file: "model-UD-Q5_K_M.gguf", quant: "UD-Q5_K_M", size_mb: 1000, fit: "fits" }] })));
  await page.route("**/admin/library/folder**", route => {
    const path = new URL(route.request().url()).searchParams.get("path");
    return route.fulfill(path === "relative" ? { ...json({ detail: "Choose a full path" }), status: 400 } : json({ path, free_mb: 10000, exists: false }));
  });
  await page.route("**/admin/library/jobs", route => route.fulfill(json({ jobs })));
  await page.route("**/admin/library/download", route => {
    const data = route.request().postDataJSON();
    downloads.push(data);
    jobs = [{ ...data, target_dir: `${data.dest_dir}/Model`, id: String(downloads.length), started_at: downloads.length, bytes_total: 1048576000,
      status: "error", error: "Disk filled while downloading" }];
    return route.fulfill(json(jobs[0]));
  });
  await page.goto(origin);
  await page.locator('.paper-rail a[aria-label="Discover"]').click();
  await page.fill("#library-query", "model");
  await page.locator("#library-search button[type=submit]").click();
  await page.getByRole("button", { name: "Compare versions" }).click();
  await page.locator(".library-folder-input").fill("relative");
  await page.waitForFunction(() => document.querySelector(".library-folder p").textContent.includes("full path"));
  await page.locator(".library-folder-input").fill("D:/");
  await page.waitForFunction(() => document.querySelector(".library-folder p").textContent.includes("Will be created"));
  assert.match(await page.locator(".library-option").innerText(), /5-bit compressed/);
  await page.getByRole("button", { name: "Download", exact: true }).click();
  await page.evaluate(() => pollLibraryJobs({ refreshStatus: false }));
  await page.getByRole("button", { name: /Retry download/ }).click();
  assert.equal(downloads[0].dest_dir, "D:/");
  assert.equal(downloads[1].dest_dir, "D:/");
  await page.reload();
  await page.locator('.paper-rail a[aria-label="Discover"]').click();
  await page.fill("#library-query", "model");
  await page.locator("#library-search button[type=submit]").click();
  await page.getByRole("button", { name: "Compare versions" }).click();
  assert.equal(await page.locator(".library-folder-input").inputValue(), "D:/");
}
