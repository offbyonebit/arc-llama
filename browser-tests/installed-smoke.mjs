// Real installed-wheel validation; invoked by scripts/published_smoke.py --ui-check.
import assert from "node:assert/strict";
import { chromium } from "playwright";

const origin = process.argv[2];
if (!origin?.startsWith("http://127.0.0.1:")) throw new Error("Expected a temporary loopback server URL");
const headers = { "Content-Type": "application/json", Authorization: `Bearer ${process.env.ARC_LLAMA_SMOKE_TOKEN}` };
async function api(path, data, method = "POST") {
  const response = await fetch(origin + path, { method, headers, ...(data ? { body: JSON.stringify(data) } : {}) });
  assert(response.ok, `${path}: ${response.status} ${await response.clone().text()}`);
  return response.json();
}
const model = "published-smoke";
const chatId = `installed-${Date.now()}`;
const payload = { model, messages: [{ role: "system", content: "Answer briefly." }, { role: "user", content: "café 日本語 ```python\nprint(1)\n``` Say hello." }], max_tokens: 32, temperature: 0 };
const count = await api("/v1/chat/budget", payload);
const completion = await api("/v1/chat/completions", payload);
assert.equal(count.exact, true);
assert.equal(count.prompt_tokens, completion.usage.prompt_tokens, "Budget must match native inference tokenization");
assert.equal(count.context_tokens, 4096, "Use the per-slot context, not the 8192-token pool");
const toolPayload = { ...payload, tools: [{ type: "function", function: { name: "lookup", description: "Look up release notes", parameters: { type: "object", properties: {} } } }] };
const toolCount = await api("/v1/chat/budget", toolPayload);
const toolCompletion = await api("/v1/chat/completions", toolPayload);
if (toolCount.exact) assert.equal(toolCount.prompt_tokens, toolCompletion.usage.prompt_tokens, "Tool definitions must be included in the count");
const older = "Project Arc Llama uses a blue release folder. Preserve the user's files, keep CPU tests isolated, and record release checks. ".repeat(18);
const messages = [
  { role: "user", content: older }, { role: "assistant", content: older },
  { role: "user", content: older }, { role: "assistant", content: older },
  { role: "user", content: "Keep the blue folder." }, { role: "assistant", content: "I will keep it." },
  { role: "user", content: "Finish the release checks." }, { role: "assistant", content: "Ready to continue." },
];
await api("/v1/chats", { id: chatId, title: "Installed wheel compaction" });
await api(`/v1/chats/${chatId}`, { messages, compaction: null }, "PUT");
const launch = { headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"] };
// Compaction controls live in the Context popover of the current UI.
async function openContext(page) {
  if (await page.locator("#paper-context-pop").isHidden()) await page.locator("#paper-context-toggle").click();
}
const browser = await chromium.launch({ ...launch, channel: "chrome" }).catch(() => chromium.launch(launch));
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", err => errors.push(err.message));
  await page.addInitScript(() => localStorage.setItem("arc-llama-presets", JSON.stringify({ "published-smoke": { max_tokens: 128 } })));
  await page.goto(origin + "/chat?model=" + model);
  await page.click("#history-toggle");
  await page.locator(`.h-card[data-id="${chatId}"]`).click();
  await page.locator(".message.user").last().waitFor();
  await openContext(page);
  await page.click("#compact-button");
  await page.waitForFunction(() => !document.querySelector("#compact-undo").hidden, null, { timeout: 120000 });
  const compacted = await api(`/v1/chats/${chatId}`, null, "GET");
  assert.equal(compacted.messages.length, messages.length);
  assert.equal(compacted.compaction.boundary, 4);
  assert(compacted.compaction.summary.trim());
  await page.fill("#message-input", "Continue the project in one short sentence.");
  await page.press("#message-input", "Enter");
  await page.locator(".message.assistant .turn-action").last().waitFor({ timeout: 120000 });
  const continued = await api(`/v1/chats/${chatId}`, null, "GET");
  assert.equal(continued.messages.length, messages.length + 2);
  assert(continued.messages.at(-1).content.trim());
  assert.equal(continued.messages.at(-1).status || "completed", "completed");
  await openContext(page);
  await page.click("#compact-undo");
  await page.waitForFunction(() => document.querySelector("#compact-undo").hidden);
  const restored = await api(`/v1/chats/${chatId}`, null, "GET");
  assert.equal(restored.compaction, null);
  assert.equal(restored.messages.length, messages.length + 2);
  assert.deepEqual(restored.messages.slice(0, messages.length).map(({ role, content }) => ({ role, content })), messages);
  assert.deepEqual(errors, []);
  if (process.env.ARC_LLAMA_SMOKE_SCREENSHOT) await page.screenshot({ path: process.env.ARC_LLAMA_SMOKE_SCREENSHOT, fullPage: true });
  console.log(JSON.stringify({ native_budget: count.prompt_tokens, runtime_context: count.context_tokens,
    compaction: "pass", continue_after_compaction: "pass", undo_preserves_full_transcript: "pass" }));
} finally { await browser.close(); }
