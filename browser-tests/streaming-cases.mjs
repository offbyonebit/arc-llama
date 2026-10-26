import assert from "node:assert/strict";
import { openChat } from "./harness.mjs";

const event = delta => `data: ${JSON.stringify({ choices: [{ delta }] })}\n\n`;
const done = "data:[DONE]\n\n";

async function setup(page, origin) {
  let saved = { id: "stream-fixture", title: "Synthetic stream", messages: [], created_at: 1, updated_at: 1 };
  const json = body => ({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  await page.route("**/v1/chats", route => route.fulfill(json(route.request().method() === "POST" ? saved : { data: [saved] })));
  await page.route("**/v1/chats/stream-fixture", route => {
    const request = route.request();
    if (request.method() === "PUT") saved = { ...saved, ...request.postDataJSON() };
    if (request.method() === "PATCH") saved.messages.push(...(request.postDataJSON().messages || []));
    return route.fulfill(json(saved));
  });
  await page.addInitScript(() => {
    const realFetch = window.fetch.bind(window);
    window.__streamRequests = 0;
    window.fetch = (url, options = {}) => {
      if (!String(url).endsWith("/v1/chat/completions")) return realFetch(url, options);
      window.__streamRequests++;
      const fixture = window.__streamFixture;
      let index = 0, ended = false;
      const body = new ReadableStream({
        start(controller) {
          options.signal?.addEventListener("abort", () => {
            if (!ended) { ended = true; controller.error(new DOMException("Stopped", "AbortError")); }
          }, { once: true });
        },
        pull(controller) {
          if (index < fixture.chunks.length) controller.enqueue(new Uint8Array(fixture.chunks[index++]));
          else if (!fixture.keepOpen) { ended = true; controller.close(); }
        },
        cancel() { ended = true; window.__streamCancelled++; },
      });
      return Promise.resolve(new Response(body, { headers: { "Content-Type": "text/event-stream" } }));
    };
  });
  await openChat(page, origin);
  return () => saved;
}

async function send(page, name, text, { fragment = 0, keepOpen = true } = {}) {
  const bytes = [...Buffer.from(text)];
  const chunks = fragment ? Array.from({ length: Math.ceil(bytes.length / fragment) }, (_, i) => bytes.slice(i * fragment, (i + 1) * fragment)) : [bytes];
  await page.evaluate(fixture => { window.__streamFixture = fixture; window.__streamCancelled = 0; }, { chunks, keepOpen });
  const before = await page.evaluate(() => window.__streamRequests);
  await page.fill("#message-input", name);
  await page.press("#message-input", "Enter");
  await page.waitForFunction(before => window.__streamRequests > before && !generating && document.querySelectorAll(".message.assistant .turn-action").length > 0, before);
}

export async function testChatSseFraming({ page, origin }) {
  const chat = await setup(page, origin);
  const content = "Synthetic 日本語 🦙 [DONE] text";
  const standard = event({ reasoning_content: "Synthetic reasoning", content }) +
    'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":7,"total_tokens":17}}\n\n' + done;
  const cases = [
    ["single-byte UTF-8", standard, 1],
    ["CR only", standard.replaceAll("\n", "\r"), 3],
    ["split CRLF", standard.replaceAll("\n", "\r\n"), 1],
    ["BOM and multiline data", '\ufeff: keepalive\r\nevent: message\r\ndata: {"choices":\r\ndata: [{"delta":{"content":' + JSON.stringify(content) + '}}]}\r\n\r\ndata:[DONE]\r\n\r\n', 2],
    ["ignore data after completion", event({ content }) + done + 'data: {malformed}\n\n', 0],
  ];
  for (const [name, text, fragment] of cases) {
    await send(page, name, text, { fragment });
    assert.equal(chat().messages.at(-1).content, content, name);
    assert.equal(chat().messages.at(-1).status, undefined, name);
    assert.equal(await page.locator(".message.assistant").last().locator(".stopped-note").count(), 0, name);
    assert.equal(await page.evaluate(() => window.__streamCancelled), 1, `${name}: response was not cancelled at [DONE]`);
    if (name === "single-byte UTF-8") {
      assert.equal(chat().messages.at(-1).thinking, "Synthetic reasoning");
      assert.equal(await page.evaluate(() => lastUsage), null); // reset after finishing
    }
  }
  assert.equal(await page.evaluate(() => window.__streamRequests), cases.length);
}

export async function testChatSseErrorsKeepPartial({ page, origin }) {
  const chat = await setup(page, origin);
  const prefix = event({ reasoning_content: "Retain reasoning", content: "Retain partial answer" });
  const cases = [
    ["malformed JSON", "data: {malformed}\n\n" + done],
    ["null chunk", "data: null\n\n" + done],
    ["null choice", 'data: {"choices":[null]}\n\n' + done],
    ["null delta", 'data: {"choices":[{"delta":null}]}\n\n' + done],
    ["array delta", 'data: {"choices":[{"delta":[]}]}\n\n' + done],
    ["numeric content", event({ content: 123 }) + done],
    ["invalid tool delta", event({ tool_calls: [null] }) + done],
    ["invalid usage", 'data: {"choices":[],"usage":[]}\n\n' + done],
    ["oversized line", "data:" + " ".repeat(1024 * 1024) + "\n\n" + done],
    ["oversized multiline event", ("data:" + " ".repeat(600000) + "\n").repeat(2) + "\n" + done],
    ["model finish without transport completion", 'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n', false],
    ["unterminated final event", "data:[DONE]", false],
  ];
  for (const [name, suffix, keepOpen = true] of cases) {
    await send(page, name, prefix + suffix, { keepOpen });
    assert.equal(chat().messages.at(-1).status, "interrupted", name);
    assert.equal(chat().messages.at(-1).content, "Retain partial answer", name);
    assert.equal(chat().messages.at(-1).thinking, "Retain reasoning", name);
    assert.match(await page.locator(".message.assistant").last().locator(".stopped-note").innerText(), /Interrupted/, name);
    if (keepOpen) assert.equal(await page.evaluate(() => window.__streamCancelled), 1, name);
  }
  assert.equal(await page.evaluate(() => window.__streamRequests), cases.length, "errors must not automatically replay requests");
  await send(page, "recover after malformed stream", event({ content: "Recovered normally" }) + done);
  assert.equal(chat().messages.at(-1).content, "Recovered normally");
  assert.equal(chat().messages.at(-1).status, undefined);
}
