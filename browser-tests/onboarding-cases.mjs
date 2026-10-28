import assert from "node:assert/strict";
import { ADMIN_STATUS } from "./harness.mjs";

const starter = { id: "dynamic-catalog-id", name: "Synthetic Newly Discovered Chat", license: "mit", size_mb: 4465, source: "https://huggingface.co/fixture/new-chat/tree/accepted-commit" };
const initial = () => ({ phase: "idle", starter, gpu: { name: "Intel Arc B580", vram_mb: 12288 }, runtime_ready: false, pause_requested: false });
const field = name => `[data-setup="${name}"]`;

async function setup({ page, origin }) {
  let state = initial(), registry = [], unavailable = false;
  const actions = [], inference = [];
  await page.route("**/admin/status", route => route.fulfill({ json: { ...ADMIN_STATUS, models: registry } }));
  await page.route("**/admin/onboarding", route => {
    if (route.request().method() === "POST") {
      const body = route.request().postDataJSON(); actions.push(body);
      if (body.action === "start") state = { ...state, phase: "checking", error: null, pause_requested: false };
      if (body.action === "pause") state = { ...state, pause_requested: true };
    }
    return route.fulfill({ status: unavailable ? 503 : 200, json: unavailable ? { detail: "offline" } : state });
  });
  await page.route("**/admin/library/jobs", route => route.fulfill({ json: { jobs: [] } }));
  await page.route("**/admin/load/*", route => { inference.push(route.request().url()); return route.fulfill({ json: { loaded: true } }); });
  await page.route("**/v1/chat/completions", route => { inference.push(route.request().url()); return route.fallback(); });
  await page.goto(`${origin}/`);
  await page.locator("#first-answer").waitFor();
  return {
    actions, inference,
    async state(value) { state = { ...state, ...value }; await page.evaluate(() => onboardingController.poll(true)); },
    async unavailable(value) { unavailable = value; await page.evaluate(() => onboardingController.poll(true)); },
    registry(value) { registry = value; },
  };
}

export async function testGuidedFirstAnswer({ page, origin }) {
  const f = await setup({ page, origin });
  assert.match(await page.locator(field("hardware")).innerText(), /B580.*12.0 GiB/);
  assert.match(await page.locator(field("description")).innerText(), /estimate|Estimated/);
  assert.equal(f.actions.length, 0); assert.equal(f.inference.length, 0);
  await page.locator(field("start")).click();
  assert.deepEqual(f.actions, [{ action: "start", starter: "dynamic-catalog-id" }]);
  await f.state({ phase: "runtime", runtime_bytes_done: 25, runtime_bytes_total: 100 });
  assert.match(await page.locator(field("bytes")).innerText(), /25%/);
  await f.state({ phase: "downloading", download: { bytes_done: 1048576, bytes_total: 4194304 } });
  assert.match(await page.locator(field("bytes")).innerText(), /1 MiB of 4 MiB.*25%/);
  await page.reload();
  await page.locator(field("pause")).waitFor();
  assert.match(await page.locator(field("status")).innerText(), /Downloading/);
  assert.equal(f.actions.length, 1, "reload started duplicate setup");
  await f.state({ phase: "loading" });
  assert.equal(await page.locator(field("progress")).isVisible(), false, "fake loading percentage");
  assert.match(await page.locator(field("status")).innerText(), /Loading/);
  f.registry([{ ...ADMIN_STATUS.models[0], name: "qwen", loaded: true }]);
  const navigated = page.waitForURL("**/chat?model=qwen&welcome=1");
  await f.state({ phase: "ready", model: "qwen", chat_url: "/chat?model=qwen&welcome=1" });
  await navigated;
  await page.locator("#first-question").waitFor();
  assert.equal(await page.locator("#model-select").inputValue(), "qwen");
  assert.equal(f.inference.length, 0, "setup sent an unsolicited question");
  await page.getByRole("button", { name: "Help me write something", exact: true }).click();
  assert.match(await page.locator("#message-input").inputValue(), /thank-you/);
  assert.equal(f.inference.length, 0, "suggestion automatically sent");
  await page.locator("#send-button").click();
  await page.getByRole("heading", { name: "Your first answer is here" }).waitFor();
  assert.match(await page.locator("#chat-log .message.assistant .content").last().innerText(), /Hello from arc-llama/);
  assert.equal(new URL(page.url()).searchParams.has("welcome"), false);
}

export async function testGuidedPauseRetryAndErrors({ page, origin }) {
  const f = await setup({ page, origin });
  await page.locator(field("start")).click();
  await f.state({ phase: "downloading", download: { bytes_done: 1, bytes_total: 10 } });
  await page.locator(field("pause")).click();
  await page.waitForFunction(() => document.querySelector('[data-setup="status"]')?.textContent.includes("Pausing after this step"));
  assert.equal(await page.locator(field("pause")).isDisabled(), true);
  await f.state({ phase: "paused" });
  assert.equal(await page.locator(field("start")).innerText(), "Continue setup");
  f.registry([{ ...ADMIN_STATUS.models[0], name: "qwen", loaded: false }]);
  await page.reload();
  await page.getByRole("button", { name: "Continue setup", exact: true }).waitFor();
  await page.locator(field("start")).click();
  await f.state({ phase: "error", error: "Synthetic network interruption" });
  assert.match(await page.locator(field("error")).innerText(), /network interruption/);
  assert.equal(await page.locator(field("start")).innerText(), "Retry setup");
  await f.unavailable(true);
  assert.equal(await page.locator(field("start")).isDisabled(), true);
  assert.equal(await page.locator(field("chat")).isVisible(), false);
  await f.unavailable(false);
  await page.locator(field("start")).click();
  assert.equal(f.actions.filter(a => a.action === "start").length, 3);
  assert.equal(f.inference.length, 0);
}

export async function testGuidedHardwareAndSmallScreen({ page, origin }) {
  await page.setViewportSize({ width: 390, height: 844 });
  const f = await setup({ page, origin });
  await f.state({ starter: null, gpu: null });
  assert.equal(await page.locator(field("start")).innerText(), "Check hardware again");
  await page.locator(field("start")).click();
  assert.equal(f.actions[0].action, "detect");
  await f.state(initial());
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await page.locator(field("another")).click();
  await page.locator("#library-title").waitFor();
  assert.equal(f.actions.filter(a => a.action === "start").length, 0);
}

export async function testWelcomePreservesDraftAndDoesNotClaimFailureAsSuccess({ page, origin }) {
  await page.addInitScript(() => sessionStorage.setItem("arc-llama-draft-qwen", "My own question"));
  await page.route("**/v1/chat/completions", route => route.fulfill({ status: 503, json: { error: { message: "synthetic generation failure" } } }));
  await page.goto(`${origin}/chat?model=qwen&welcome=1`);
  await page.locator("#first-question").waitFor();
  assert.equal(await page.locator("#message-input").inputValue(), "My own question");
  await page.locator("#message-input").fill("My own question");
  await page.getByRole("button", { name: "Help me write something", exact: true }).click();
  assert.equal(await page.locator("#message-input").inputValue(), "My own question");
  await page.locator("#send-button").click();
  await page.locator(".message.error").waitFor();
  assert.equal(await page.getByRole("heading", { name: "Your first answer is here" }).count(), 0);
  assert.equal(new URL(page.url()).searchParams.get("welcome"), "1");
}


export async function testWelcomeNeedsCompleteNonemptyReply({ page, origin }) {
  let finish = "length", content = "An incomplete answer";
  await page.route("**/v1/chat/completions", route => route.fulfill({ contentType: "text/event-stream", body: [
    "data: " + JSON.stringify({ choices: [{ delta: { content } }] }),
    "data: " + JSON.stringify({ choices: [{ delta: {}, finish_reason: finish }] }),
    "data: [DONE]",
  ].join("\n\n") + "\n\n" }));
  for (const mode of ["truncated", "empty"]) {
    if (mode === "empty") { finish = "stop"; content = ""; }
    await page.goto(`${origin}/chat?model=qwen&welcome=1`);
    await page.locator("#first-question").waitFor();
    await page.locator("#message-input").fill("A synthetic question");
    const response = page.waitForResponse("**/v1/chat/completions");
    await page.locator("#send-button").click();
    await response;
    await page.waitForFunction(() => document.querySelector(".message.assistant") && !document.querySelector("#send-button").classList.contains("stop"));
    assert.equal(await page.getByRole("heading", { name: "Your first answer is here" }).count(), 0);
    assert.equal(new URL(page.url()).searchParams.get("welcome"), "1");
  }
}


export async function testGuidedDynamicCatalogStates({ page, origin }) {
  const f = await setup({ page, origin });
  assert.match(await page.locator(field("source")).innerText(), /License: mit/);
  await f.state({ starter: null, discovery: { status: "searching" } });
  assert.match(await page.locator(field("description")).innerText(), /Searching the current/);
  assert.equal(await page.locator(field("start")).isDisabled(), true);
  assert.equal(await page.locator(field("source")).isVisible(), false);
  for (const status of ["unavailable", "no_match"]) {
    await f.state({ discovery: { status } });
    assert.equal(await page.locator(field("start")).innerText(), "Retry model search");
    await page.locator(field("start")).click();
    assert.equal(f.actions.at(-1).action, "refresh");
    assert.equal(f.actions.filter(a => a.action === "start").length, 0);
  }
  await f.state({ starter: { ...starter, id: "brand-new-id", name: "Another unseen candidate", license: "custom-license" }, discovery: { status: "ready" } });
  assert.equal(await page.locator(field("model")).innerText(), "Another unseen candidate");
  assert.match(await page.locator(field("source")).innerText(), /custom-license/);
  await page.locator(field("start")).click();
  assert.deepEqual(f.actions.at(-1), { action: "start", starter: "brand-new-id" });
  assert.equal(f.inference.length, 0);
}
