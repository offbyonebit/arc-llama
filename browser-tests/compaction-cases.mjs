import { openContext } from "./harness.mjs";
import assert from "node:assert/strict";
import { ASSISTANT_REPLY, openChat, MODELS, ADMIN_STATUS } from "./harness.mjs";

const json = body => ({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
const summary = text => ({ choices: [{ message: { content: text }, finish_reason: "stop" }] });

async function fixture(page, origin, { messages, ctx = 8192, respond, failSave = false, legacyServer = false } = {}) {
  const initial = messages || [
    { role: "user", content: "Earlier project request. " + "Preserve the benchmark constraints. ".repeat(45) },
    { role: "assistant", content: "Earlier project answer. " + "Run five repetitions and keep results. ".repeat(40) },
    { role: "user", content: "Earlier follow-up. " + "Keep the original files and reports. ".repeat(25) },
    { role: "assistant", content: "Earlier follow-up answer. " + "Use the B60 with the saved recipe. ".repeat(20) },
    { role: "user", content: "Recent question one" },
    { role: "assistant", content: "Recent answer one" },
    { role: "user", content: "Recent question two" },
    { role: "assistant", content: "Recent answer two" },
  ];
  let chat = { id: "compact-chat", title: "Compaction test", folder: "", messages: initial,
    created_at: 1, updated_at: 1, compaction: null };
  const requests = [];
  const saves = [];
  await page.route("**/v1/models", route => route.fulfill(json({ ...MODELS,
    data: MODELS.data.map(m => ({ ...m, metadata: { ...m.metadata, ctx } })) })));
  await page.route("**/admin/status", route => route.fulfill(json({ ...ADMIN_STATUS,
    models: ADMIN_STATUS.models.map(m => ({ ...m, ctx })) })));
  await page.route("**/v1/chats", route => route.fulfill(json({ object: "list", data: [chat] })));
  await page.route("**/v1/chats/compact-chat", async route => {
    const req = route.request();
    if (req.method() === "PUT") {
      saves.push(req.postDataJSON());
      if (typeof failSave === "function" ? failSave() : failSave) return route.fulfill({ ...json({ detail: "Disk is full" }), status: 503 });
      chat = { ...chat, ...req.postDataJSON() };
      if (legacyServer) delete chat.compaction;
    } else if (req.method() === "PATCH") {
      chat = { ...chat, messages: [...chat.messages, ...(req.postDataJSON().messages || [])] };
    }
    return route.fulfill(json(chat));
  });
  await page.route("**/v1/chat/completions", async route => {
    const body = route.request().postDataJSON();
    requests.push({ body, auth: route.request().headers().authorization });
    if (body.stream) return route.fulfill({ status: 200, contentType: "text/event-stream", body: ASSISTANT_REPLY });
    if (respond) return respond(route, body, requests.length);
    return route.fulfill(json(summary("We are optimizing Arc inference. Preserve original files, B60 recipes, and five benchmark repetitions.")));
  });
  await openChat(page, origin);
  await page.click("#history-toggle");
  await page.locator(".h-card").filter({ hasText: "Compaction test" }).click();
  await page.waitForFunction(() => document.querySelectorAll(".message.user").length >= 2);
  return { initial, requests, saves, chat: () => chat };
}

async function compact(page) {
  await openContext(page);
  await page.click("#compact-button");
  await page.waitForFunction(() => !document.querySelector("#compact-undo").hidden);
}

async function send(page, text) {
  const response = page.waitForResponse(r => r.url().includes("/v1/chat/completions") && r.request().postDataJSON().stream);
  await page.fill("#message-input", text);
  await page.press("#message-input", "Enter");
  await response;
  await page.waitForFunction(() => !document.querySelector("#send-button").classList.contains("stop") &&
    [...document.querySelectorAll(".message.assistant .content")].some(el => el.textContent.includes("Hello from arc-llama")));
  await page.waitForFunction(() => !document.querySelector("#compact-undo").disabled);
}

export async function testCompactionRoundTrip({ page, origin }) {
  const f = await fixture(page, origin);
  await compact(page);
  assert.equal(f.requests[0].auth, "Bearer browser-test-token");
  const summaryInput = f.requests[0].body.messages.map(m => m.content).join("\n");
  assert.match(summaryInput, /Earlier project request/);
  assert.doesNotMatch(summaryInput, /Recent question/);
  assert.equal(f.chat().compaction.boundary, 4);
  assert.deepEqual(f.chat().messages, f.initial);
  assert.equal(await page.locator(".message.user").count(), 4, "full transcript remains visible");
  await openContext(page);
  assert.match(await page.locator("#compact-status").innerText(), /tokens.*full transcript kept/);
  await send(page, "Continue the measurements");
  const body = f.requests.find(r => r.body.stream).body;
  assert.match(body.messages[0].content, /Summary of earlier conversation/);
  assert.deepEqual(body.messages.slice(1, 5), f.initial.slice(4));
  assert.equal(body.messages.at(-1).content, "Continue the measurements");
  assert.equal(f.chat().messages.length, 10);
  assert.equal(f.chat().compaction.boundary, 4);
  await openContext(page);
  await page.click("#compact-undo");
  await page.waitForFunction(() => document.querySelector("#compact-undo").hidden);
  assert.equal(f.chat().compaction, null);
  assert.equal(f.chat().messages.length, 10, "undo retains later exchanges");
  await send(page, "Continue with full context");
  assert.equal(f.requests.at(-1).body.messages[0].content, f.initial[0].content);
}

export async function testCompactionReloadAndClear({ page, origin }) {
  const f = await fixture(page, origin);
  await compact(page);
  await page.reload();
  await page.waitForFunction(() => document.querySelector("#model-select").options.length > 0);
  await page.click("#history-toggle");
  await page.locator(".h-card").filter({ hasText: "Compaction test" }).click();
  await page.waitForFunction(() => !document.querySelector("#compact-undo").hidden);
  assert.equal(await page.locator(".message.user").count(), 4);
  await send(page, "After reload");
  assert.match(f.requests.at(-1).body.messages[0].content, /Summary of earlier/);
  await page.fill("#message-input", "/clear ");
  await page.press("#message-input", "Enter");
  await page.waitForFunction(() => document.querySelector("#compact-undo").hidden);
  assert.equal(f.chat().compaction, null);
  assert.deepEqual(f.chat().messages, []);
}

export async function testCompactionFailures({ page, origin }) {
  let mode = "empty";
  const f = await fixture(page, origin, { respond: route => {
    if (mode === "http") return route.fulfill({ ...json({ detail: "Runtime failed" }), status: 500 });
    const body = summary(mode === "empty" ? "" : "Short summary");
    if (mode === "truncated") body.choices[0].finish_reason = "length";
    return route.fulfill(json(body));
  } });
  for (mode of ["empty", "http", "truncated"]) {
    await openContext(page);
  await page.click("#compact-button");
    await page.waitForFunction(n => document.querySelectorAll(".message.error").length >= n, f.requests.length);
    await page.waitForFunction(() => !document.querySelector("#compact-button").disabled);
    assert.equal(f.saves.length, 0);
    assert.equal(f.chat().compaction, null);
    assert.deepEqual(f.chat().messages, f.initial);
  }
}

export async function testCompactionSaveFailure({ page, origin }) {
  const f = await fixture(page, origin, { failSave: true });
  await openContext(page);
  await page.click("#compact-button");
  await page.waitForFunction(() => document.querySelector(".message.error .content")?.textContent.includes("Disk is full"));
  assert.equal(f.chat().compaction, null);
  assert.deepEqual(f.chat().messages, f.initial);
  assert.equal(await page.locator("#compact-undo").isVisible(), false);
  await send(page, "Still full context");
  assert.equal(f.requests.at(-1).body.messages[0].content, f.initial[0].content);
}

export async function testCompactionCancelAndGuards({ page, origin }) {
  let release;
  const waiting = new Promise(resolve => { release = resolve; });
  const f = await fixture(page, origin, { respond: async route => {
    await waiting;
    await route.fulfill(json(summary("Delayed summary"))).catch(() => {});
  } });
  await openContext(page);
  await page.click("#compact-button");
  await page.waitForFunction(() => document.querySelector("#compact-status").textContent.includes("Summarizing"));
  assert.equal(await page.locator("#model-select").isDisabled(), true);
  await page.fill("#message-input", "/new ");
  await page.press("#message-input", "Enter");
  await page.waitForFunction(() => document.querySelector(".message.error .content")?.textContent.includes("Finish or stop"));
  assert.equal(await page.locator(".message.user").count(), 4);
  await page.fill("#message-input", "/compact another focus");
  await page.press("#message-input", "Enter");
  await page.waitForFunction(() => document.querySelectorAll(".message.error").length === 2);
  await openContext(page);
  await page.click("#compact-cancel");
  release();
  await page.waitForFunction(() => document.querySelector("#compact-cancel").hidden);
  assert.equal(f.requests.length, 1);
  assert.equal(f.saves.length, 0);
  assert.deepEqual(f.chat().messages, f.initial);
  assert.equal(f.chat().compaction, null);
}

export async function testCompactionOversizedAndImages({ page, origin }) {
  const large = "START-FACT " + "漢字🚀 Keep the benchmark evidence. ".repeat(550) + " END-FACT";
  const image = "data:image/png;base64," + "x".repeat(15000);
  const messages = [
    { role: "system", content: "Always answer in English." },
    { role: "user", content: large, images: [image] },
    { role: "assistant", content: "Old answer" },
    { role: "user", content: "Second old request" },
    { role: "assistant", content: "Second old answer" },
    { role: "user", content: "Recent request one" },
    { role: "assistant", content: "Recent answer one" },
    { role: "user", content: "Recent request two" },
    { role: "assistant", content: "Recent answer two" },
  ];
  const f = await fixture(page, origin, { messages, ctx: 4096 });
  await page.fill("#message-input", "/compact benchmark evidence");
  await page.press("#message-input", "Enter");
  await page.waitForFunction(() => !document.querySelector("#compact-undo").hidden);
  assert.ok(f.requests.length > 2, "oversized history must use bounded requests");
  const excerpts = f.requests.map(r => r.body.messages[1].content.split("Next excerpt:\n")[1]).join("");
  assert.ok(excerpts.includes(large), "all of the oversized message was summarized, including Unicode");
  assert.match(excerpts, /image contents are not included/);
  assert.doesNotMatch(excerpts, /base64/);
  for (const r of f.requests) {
    const estimated = r.body.messages.reduce((n, m) => n + Buffer.byteLength(m.content), 0);
    assert.ok(estimated + r.body.max_tokens + 256 + 64 <= 4096);
    assert.match(r.body.messages[0].content, /benchmark evidence/);
  }
  assert.deepEqual(f.chat().messages, messages);
  assert.deepEqual(f.chat().messages[1].images, [image]);
  await send(page, "Next turn");
  assert.equal(f.requests.at(-1).body.messages[0].content, "Always answer in English.");
  assert.match(f.requests.at(-1).body.messages[1].content, /Summary of earlier/);
}

export async function testCompactionRecentTooLarge({ page, origin }) {
  const f = await fixture(page, origin, { messages: [
    { role: "user", content: "Old request ".repeat(1000) },
    { role: "assistant", content: "Old answer" },
    { role: "user", content: "Recent huge request ".repeat(250) },
    { role: "assistant", content: "Answer" },
  ], ctx: 4096 });
  await openContext(page);
  await page.click("#compact-button");
  await page.waitForFunction(() => document.querySelector(".message.error .content")?.textContent.includes("Recent turns still exceed"));
  assert.equal(f.saves.length, 0);
  assert.deepEqual(f.chat().messages, f.initial);
}


export async function testCompactionUndoFailureAndEdit({ page, origin }) {
  let fail = false;
  const f = await fixture(page, origin, { failSave: () => fail });
  await compact(page);
  await send(page, "New turn after compaction");
  const regenerateResponse = page.waitForResponse(r => r.url().includes("/v1/chat/completions") && r.request().postDataJSON().stream);
  await page.getByRole("button", { name: "Regenerate", exact: true }).click();
  await regenerateResponse;
  await page.waitForFunction(() => !document.querySelector("#compact-undo").disabled);
  assert.equal(f.chat().compaction.boundary, 4);
  assert.equal(f.chat().messages.length, 10);
  const editSave = page.waitForResponse(r => r.url().includes("/v1/chats/compact-chat") && r.request().method() === "PUT");
  await page.getByRole("button", { name: "Edit", exact: true }).click();
  await editSave;
  assert.equal(f.chat().compaction.boundary, 4);
  assert.equal(f.chat().messages.length, 8);
  assert.equal(await page.inputValue("#message-input"), "New turn after compaction");
  fail = true;
  await openContext(page);
  await page.click("#compact-undo");
  await page.waitForFunction(() => document.querySelector(".message.error .content")?.textContent.includes("Undo failed"));
  assert.equal(f.chat().compaction.boundary, 4);
  assert.equal(await page.locator("#compact-undo").isVisible(), true);
  fail = false;
  await page.fill("#message-input", "/uncompact ");
  await page.press("#message-input", "Enter");
  await page.waitForFunction(() => document.querySelector("#compact-undo").hidden);
  assert.equal(f.chat().compaction, null);
}


export async function testCompactionRequiresPersistenceAck({ page, origin }) {
  const f = await fixture(page, origin, { legacyServer: true });
  await openContext(page);
  await page.click("#compact-button");
  await page.waitForFunction(() => document.querySelector(".message.error .content")?.textContent.includes("Restart Arc Llama"));
  assert.equal(await page.locator("#compact-undo").isVisible(), false);
  assert.deepEqual(f.chat().messages, f.initial);
  await send(page, "Full context remains active");
  assert.equal(f.requests.at(-1).body.messages[0].content, f.initial[0].content);
}
