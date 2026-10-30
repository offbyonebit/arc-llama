// Status polls and the Work clock must not rebuild what the user is using.
// Same contract as the other case files: no model, no server.
import assert from "node:assert/strict";
import { openChat } from "./harness.mjs";

const json = (body) => ({ status: 200, contentType: "application/json", body: JSON.stringify(body) });

export async function testChatPollKeepsExactBudget({ page, origin }) {
  let counts = 0;
  await page.route("**/v1/chat/budget", route => {
    counts += 1;
    return route.fulfill(json({ total_tokens: 1312, reserved_tokens: 1024, context_tokens: 8192, exact: true, reason: "Counted." }));
  });
  await openChat(page, origin);
  await page.fill("#message-input", "hello");
  await page.waitForFunction(() => /^1,312 \//.test(document.getElementById("ctx-label-left").textContent));
  const before = counts;
  for (let i = 0; i < 3; i++) await page.evaluate(() => fetchStatus());
  assert.match(await page.textContent("#ctx-label-left"), /^1,312 \//, "poll replaced the exact count with an estimate");
  await page.waitForTimeout(450);
  assert.equal(counts, before, "poll re-counted an unchanged conversation");
  await page.fill("#message-input", "hello again");
  await page.waitForTimeout(450);
  assert.ok(counts > before, "an edited draft was not re-counted");
}

export async function testChatPollKeepsSettingsOpen({ page, origin }) {
  await openChat(page, origin);
  await page.click("#settings-toggle");
  await page.locator(".s-runtime summary").click();
  await page.fill("#s-system", "Unsaved instructions");
  await page.evaluate(() => document.activeElement?.blur());
  await page.evaluate(() => fetchStatus());
  await page.waitForTimeout(100);
  assert.equal(await page.locator(".s-runtime").evaluate(n => n.open), true, "poll closed the runtime section");
  assert.equal(await page.inputValue("#s-system"), "Unsaved instructions", "poll discarded typed preset text");
}

export async function testModelPanelKeepsFocusAcrossPolls({ page, origin }) {
  await page.goto(`${origin}/`);
  await page.locator(".model-card").first().click();
  const action = page.locator("#paper-model-actions button").first();
  await action.waitFor();
  await action.focus();
  await page.evaluate(() => fetchStatus(true));
  await page.evaluate(() => fetchStatus(true));
  assert.equal(await action.evaluate(n => n.isConnected && document.activeElement === n), true, "poll moved focus off the model action");
}

export async function testActionResultSurvivesNextPoll({ page, origin }) {
  await page.goto(`${origin}/`);
  await page.locator(".model-card").first().waitFor();
  await page.evaluate(() => setFooter("Delete failed (500)", true));
  await page.evaluate(() => fetchStatus(true));
  assert.equal(await page.textContent("#last-updated"), "Delete failed (500)");
}

export async function testWorkClockKeepsApprovalFocus({ page, origin }) {
  await page.addInitScript(() => {
    const real = window.fetch.bind(window);
    window.fetch = (input, init = {}) => {
      const url = typeof input === "string" ? input : input.url;
      if (url.endsWith("/v1/agent") && (init.method || "GET") === "POST") {
        const enc = new TextEncoder();
        const stream = new ReadableStream({ start(c) { window.__push = e => c.enqueue(enc.encode("data: " + JSON.stringify(e) + "\n\n")); } });
        return Promise.resolve(new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } }));
      }
      return real(input, init);
    };
  });
  await page.route("**/v1/agent/info", route => route.fulfill(json({
    experimental: true, auto_approve: true, root: "/work/demo", home: "/work/demo",
    harnesses: [{ id: "builtin", label: "Built-in", available: true }],
  })));
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.fill("#work-task", "Create hello.txt");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__push);
  await page.evaluate(() => {
    window.__push({ type: "tool_call", id: "c1", name: "bash", arguments: { command: "printf hi > hello.txt" } });
    window.__push({ type: "confirm_required", id: "per_1", run_id: "run-1", tool: "bash", arguments: { patterns: ["printf hi > hello.txt"] } });
  });
  const allow = page.getByRole("button", { name: "Allow", exact: true });
  await allow.waitFor();
  await allow.focus();
  const clock = await page.textContent("#work-activity-time");
  await page.waitForTimeout(2300);
  assert.equal(await allow.evaluate(n => n.isConnected && document.activeElement === n), true, "the clock rebuilt the approval");
  assert.notEqual(await page.textContent("#work-activity-time"), clock, "elapsed time stopped updating");
}
