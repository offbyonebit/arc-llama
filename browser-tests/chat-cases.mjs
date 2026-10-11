// Existing regression cases; shared fixtures and browser setup live in harness.mjs.
import {
  ASSISTANT_REPLY,
  STRUCTURED_LOAD_FAILURE,
  MODELS,
  ADMIN_STATUS,
  openChat,
} from "./harness.mjs";

export async function testChatSelection({ page, origin }) {
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

export async function testChatHistoryComponents({ page, origin }) {
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

export async function testChatSend({ page, origin }) {
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

export async function testStructuredLoadFailure({ page, origin }) {
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

export async function testRefreshPreservesEdits({ page, origin }) {
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

export async function testKeyboardNavigation({ page, origin }) {
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

export async function testSettingsSurviveStatusPoll({ page, origin }) {
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

export async function testRetrySendsOriginalTurnOnce({ page, origin }) {
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

export async function testMemoryFitAndSavedSettings({ page, origin }) {
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

export async function testRegenerateAndEdit({ page, origin }) {
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

export async function testStopKeepsPartialAnswer({ page, origin }) {
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

export async function testBudgetShownBeforeSending({ page, origin }) {
  await openChat(page, origin);
  await page.fill("#message-input", "x".repeat(40000));
  await page.waitForFunction(() => document.getElementById("ctx-meter").classList.contains("over-budget"));
  const label = await page.textContent("#ctx-label-left");
  if (!label.startsWith("~") || !label.includes("8,192")) throw new Error(`unexpected budget label: ${label}`);
  await page.fill("#message-input", "short");
  await page.waitForFunction(() => !document.getElementById("ctx-meter").classList.contains("over-budget"));
}

export async function testPresetShapesRequest({ page, origin }) {
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

export async function testTextAttachmentExtensionFallback({ page, origin }) {
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

export async function testImagesNeedVisionModel({ page, origin }) {
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
