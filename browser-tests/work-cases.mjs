// Work view regression cases. Same contract as the other case files: no model, no server.
import { openChat } from "./harness.mjs";

const json = (body) => ({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
const INFO = {
  experimental: true,
  auto_approve: true,
  root: "/work/demo",
  home: "/home/demo",
  harnesses: [
    { id: "builtin", label: "Built-in", available: true },
    { id: "opencode", label: "OpenCode", available: true },
  ],
};

// Replaces POST /v1/agent with a stream the test drives by hand, so approvals can be
// observed while a run is still in flight.
async function installAgentStream(page) {
  await page.addInitScript(() => {
    const real = window.fetch.bind(window);
    window.fetch = (input, init = {}) => {
      const url = typeof input === "string" ? input : input.url;
      if (url.endsWith("/v1/agent") && (init.method || "GET") === "POST") {
        window.__agentBody = JSON.parse(init.body);
        const enc = new TextEncoder();
        const stream = new ReadableStream({
          start(controller) {
            window.__push = (event) => controller.enqueue(enc.encode("data: " + JSON.stringify(event) + "\n\n"));
            window.__end = () => controller.close();
            // A real fetch rejects its body when the caller aborts; mirror that.
            if (init.signal) init.signal.addEventListener("abort", () => controller.error(new DOMException("Aborted", "AbortError")));
          },
        });
        return Promise.resolve(new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } }));
      }
      return real(input, init);
    };
  });
}

export async function testWorkViewHiddenUnlessExperimental({ page, origin }) {
  await page.route("**/v1/agent/info", route => route.fulfill(json({ ...INFO, experimental: false })));
  await openChat(page, origin);
  await page.waitForTimeout(200);
  if (!(await page.locator("#mode-switch").isHidden())) throw new Error("Work switch visible without the experimental flag");
  if (await page.evaluate(() => document.body.classList.contains("work-mode"))) throw new Error("work mode active without the flag");
}

export async function testWorkModeApprovalAndUndo({ page, origin }) {
  const calls = [];
  await page.addInitScript(() => {
    const NativeObserver = window.ResizeObserver;
    window.__workObserved = new Set();
    window.ResizeObserver = class extends NativeObserver {
      targets = new Set();
      observe(target, options) {
        if (target.classList.contains("work-steps")) {
          this.targets.add(target); window.__workObserved.add(target);
        }
        super.observe(target, options);
      }
      disconnect() {
        for (const target of this.targets) window.__workObserved.delete(target);
        this.targets.clear(); super.disconnect();
      }
    };
  });
  await installAgentStream(page);
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  await page.route("**/v1/agent/run-1/confirm", route => {
    calls.push({ kind: "confirm", body: route.request().postDataJSON() });
    return route.fulfill(json({ ok: true }));
  });
  await page.route("**/v1/agent/run-1/undo", route => {
    calls.push({ kind: "undo" });
    return route.fulfill(json({ restored: [{ path: "hello.txt", status: "added" }], skipped: [] }));
  });
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  if (!(await page.evaluate(() => document.body.classList.contains("work-mode")))) throw new Error("Work mode did not open");
  if (await page.locator("#chat-log").isVisible()) throw new Error("chat log still visible in Work mode");

  // The folder defaults to the user's home; clearing it means nothing may be sent, and the user is told why.
  if ((await page.inputValue("#work-root")) !== "/home/demo") throw new Error("folder did not default to home");
  await page.fill("#work-root", "");
  await page.fill("#work-task", "Create hello.txt containing hi. Then read it back.");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !document.querySelector("#work-problem").hidden);
  if (!(await page.locator("#work-problem").innerText()).includes("Choose a project folder first")) throw new Error("problem not explained in the main area");
  if (!(await page.locator("#work-root").evaluate((n) => n.closest(".work-chip").classList.contains("invalid")))) throw new Error("folder chip not highlighted");
  if (await page.evaluate(() => !!window.__push)) throw new Error("a run started without a project folder");
  await page.fill("#work-root", "/work/demo");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__push);
  const body = await page.evaluate(() => window.__agentBody);
  if (body.auto_confirm !== false) throw new Error("auto-approve must be off unless the user turned it on");
  if (body.harness !== "opencode" || body.local_only !== true || body.track_changes !== true || body.root !== "/work/demo") {
    throw new Error("run request is wrong: " + JSON.stringify(body));
  }

  const cmd = "printf 'hi' > hello.txt";
  await page.evaluate((command) => {
    window.__push({ type: "tool_call", id: "c1", name: "bash", arguments: { command } });
    window.__push({ type: "confirm_required", id: "per_1", run_id: "run-1", tool: "bash", arguments: { patterns: [command], metadata: { command } } });
  }, cmd);
  await page.waitForSelector(".work-sheet");
  if (!(await page.locator(".paper-nav-dot").isVisible())) throw new Error("Work rail did not indicate the pending permission");
  const sheet = await page.locator(".work-sheet").innerText();
  if (!sheet.includes("Allow Arc Llama to write a file?") || !sheet.includes("hello.txt")) throw new Error("approval sheet is not plain language: " + sheet);
  if (await page.locator(".work-sheet .work-code").count()) throw new Error("command shown before asking for it");
  await page.getByRole("button", { name: "Allow", exact: true }).click();
  await page.waitForFunction(() => !document.querySelector(".work-sheet"));
  if (await page.locator(".paper-nav-dot").isVisible()) throw new Error("Permission dot stayed on after approval");
  if (!calls.some(c => c.kind === "confirm" && c.body.approved === true)) throw new Error("approval was not sent");

  await page.evaluate(() => {
    window.__push({ type: "tool_result", id: "c1", name: "bash", content: "(no output)", error: false });
    window.__push({ type: "assistant", content: "Done." });
    window.__push({ type: "changes", run_id: "run-1", undoable: true, truncated: false,
      files: [{ path: "hello.txt", status: "added", additions: 1, deletions: 0, preview: ["hi"], binary: false }] });
    window.__push({ type: "done" });
    window.__end();
  });
  await page.waitForSelector(".work-result");
  const result = await page.locator(".work-result").innerText();
  if (!result.includes("1 file changed") || !result.includes("hello.txt")) throw new Error("result card missing: " + result);
  if (!(await page.locator(".work-steps").innerText()).includes("Wrote hello.txt")) throw new Error("step was not summarised in plain language");
  await page.getByRole("button", { name: "Review", exact: true }).click();
  await page.locator(".paper-work-review").waitFor();
  if (!(await page.locator(".paper-work-review").innerText()).includes("hello.txt")) throw new Error("review panel lost the file change");
  await page.getByRole("button", { name: "Close review", exact: true }).click();
  if (!(await page.locator("#work-run .work-result").isVisible())) throw new Error("closing review lost the result card");
  await page.getByRole("button", { name: "Review work", exact: true }).click();
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  await page.waitForFunction(() => document.querySelector(".work-result")?.textContent.includes("Undone: 1 file restored."));
  if (!calls.some(c => c.kind === "undo")) throw new Error("undo was not requested");
  if (await page.evaluate(() => [...window.__workObserved].some(target => !target.isConnected))) {
    throw new Error("Work updates retained observers on removed step trees");
  }
}

export async function testWorkModeDeniedAndStopped({ page, origin }) {
  const calls = [];
  await installAgentStream(page);
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  await page.route("**/v1/agent/run-2/confirm", route => {
    calls.push(route.request().postDataJSON());
    return route.fulfill(json({ ok: true }));
  });
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.fill("#work-root", "/work/demo");
  await page.fill("#work-task", "Delete the build folder");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__push);
  await page.evaluate(() => {
    window.__push({ type: "tool_call", id: "c1", name: "bash", arguments: { command: "rm -rf build" } });
    window.__push({ type: "confirm_required", id: "per_9", run_id: "run-2", tool: "bash", arguments: { patterns: ["rm -rf build"] } });
  });
  await page.waitForSelector(".work-sheet");
  const sheet = await page.locator(".work-sheet").innerText();
  if (!sheet.includes("delete") || !sheet.includes("rm -rf build")) throw new Error("risky command was not surfaced: " + sheet);
  await page.getByRole("button", { name: "Don’t Allow" }).click();
  await page.waitForFunction(() => !document.querySelector(".work-sheet"));
  if (calls.length !== 1 || calls[0].approved !== false) throw new Error("denial was not sent");
  // The send button doubles as Stop while a run is in flight.
  if ((await page.getAttribute("#work-send", "aria-label")) !== "Stop") throw new Error("no stop control while running");
  await page.click("#work-send");
  await page.waitForFunction(() => document.querySelector(".work-eyebrow")?.textContent.includes("stopped"));
}

export async function testWorkModeServerRejectsMissingFolder({ page, origin }) {
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  await page.route("**/v1/agent", route => route.request().method() === "POST"
    ? route.fulfill({ status: 400, contentType: "application/json", body: JSON.stringify({ detail: "Project folder not found on the server: /nope" }) })
    : route.continue());
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.fill("#work-root", "/nope");
  await page.fill("#work-task", "do something");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => document.querySelector("#work-run .work-error")?.textContent.includes("Project folder not found"));
  if (!(await page.locator(".work-eyebrow").innerText()).includes("error")) throw new Error("run was not marked as failed");
}

export async function testWorkModeEndedWithoutResultIsAnError({ page, origin }) {
  await installAgentStream(page);
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.fill("#work-root", "/work/demo");
  await page.fill("#work-task", "do something");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__push);
  await page.evaluate(() => { window.__push({ type: "status", message: "Working…" }); window.__end(); });  // server died silently
  await page.waitForFunction(() => document.querySelector("#work-run .work-error")?.textContent.includes("ended before the task finished"));
}


export async function testWorkModeOffersClosestFolder({ page, origin }) {
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  await page.route("**/v1/agent", route => route.request().method() === "POST"
    ? route.fulfill({ status: 400, contentType: "application/json",
        body: JSON.stringify({ detail: "Project folder not found on the server: /home/demo/home", closest: "/home/demo" }) })
    : route.continue());
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.fill("#work-root", "~/home");
  await page.fill("#work-task", "list my files");
  await page.press("#work-task", "Enter");
  await page.getByRole("button", { name: "Use /home/demo" }).click();
  if ((await page.inputValue("#work-root")) !== "/home/demo") throw new Error("folder was not corrected");
  if ((await page.inputValue("#work-task")) !== "list my files") throw new Error("the task was not restored");
  if (await page.locator("#work-run").isVisible()) throw new Error("the failed run was left on screen");
}

export async function testWorkModeAutoApprove({ page, origin }) {
  await installAgentStream(page);
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  await page.route("**/v1/agent/run-3/confirm", route => route.fulfill(json({ ok: true })));
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  const chip = page.locator("#work-auto");
  if ((await chip.getAttribute("aria-pressed")) !== "false" || !(await chip.innerText()).includes("Ask before changes")) throw new Error("auto-approve must default to off");
  await chip.click();
  if ((await chip.getAttribute("aria-pressed")) !== "true" || (await chip.innerText()) !== "Auto-approve") throw new Error("toggle did not turn on");
  if (!(await page.locator("#work-hint").innerText()).includes("Risky commands still ask")) throw new Error("the safety net was not explained");
  if ((await page.evaluate(() => sessionStorage.getItem("arc-llama-work-auto"))) !== "1") throw new Error("not remembered for this tab");
  if ((await page.evaluate(() => localStorage.getItem("arc-llama-work-auto"))) !== null) throw new Error("auto-approve must not outlive the tab");
  await page.fill("#work-root", "/work/demo");
  await page.fill("#work-task", "clean up the build output");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__push);
  if ((await page.evaluate(() => window.__agentBody.auto_confirm)) !== true) throw new Error("auto_confirm was not sent");
  if (!(await page.locator(".work-eyebrow").innerText()).includes("auto-approve")) throw new Error("run does not say it is auto-approving");
  await page.evaluate(() => {
    window.__push({ type: "tool_call", id: "c1", name: "bash", arguments: { command: "rm -rf build" } });
    window.__push({ type: "confirm_required", id: "per_3", run_id: "run-3", tool: "bash", arguments: { patterns: ["rm -rf build"] } });
  });
  await page.waitForSelector(".work-sheet");
  if (!(await page.locator(".work-sheet").innerText()).includes("still asks")) throw new Error("a risky command in auto mode did not explain itself");
}

export async function testWorkModeAutoApproveHiddenOnOldServer({ page, origin }) {
  await page.addInitScript(() => { try { sessionStorage.setItem("arc-llama-work-auto", "1"); } catch (e) {} });
  await page.route("**/v1/agent/info", route => route.fulfill(json({ ...INFO, auto_approve: undefined })));
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  if (await page.locator("#work-auto").isVisible()) throw new Error("auto-approve offered by a server that cannot keep risky commands asking");
  if ((await page.evaluate(() => sessionStorage.getItem("arc-llama-work-auto"))) !== null) throw new Error("a stale auto-approve setting was kept");
}

export async function testWorkModeKeepsTheConversation({ page, origin }) {
  await installAgentStream(page);
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.fill("#work-root", "/work/demo");

  await page.fill("#work-task", "Create hello.txt containing hi");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__push);
  await page.evaluate(() => { window.__push({ type: "assistant", content: "First answer." }); window.__push({ type: "done" }); window.__end(); window.__push = null; });
  await page.waitForFunction(() => document.querySelector(".work-eyebrow")?.textContent.includes("finished"));

  // Second message: the first stays on screen, and the agent is told what came before.
  await page.fill("#work-task", "Now add a second file");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__push);
  if ((await page.locator(".work-turn").count()) !== 2) throw new Error("the first message disappeared");
  const body = await page.evaluate(() => window.__agentBody);
  if (!body.history || body.history.length !== 2 || body.history[0].content !== "Create hello.txt containing hi" || body.history[1].content !== "First answer.") {
    throw new Error("earlier turns were not sent: " + JSON.stringify(body.history));
  }

  // While it works, Enter must not stop it.
  await page.fill("#work-task", "a third message too early");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => document.querySelector("#work-hint")?.textContent.includes("Still working"));
  if ((await page.getAttribute("#work-send", "aria-label")) !== "Stop") throw new Error("the running task was interrupted by Enter");
  if ((await page.locator(".work-turn").count()) !== 2) throw new Error("a message was dropped or duplicated");

  await page.evaluate(() => { window.__push({ type: "assistant", content: "Second answer." }); window.__push({ type: "done" }); window.__end(); });
  await page.waitForFunction(() => document.querySelectorAll(".work-eyebrow")[1]?.textContent.includes("finished"));
  const text = await page.locator("#work-run").innerText();
  if (!text.includes("First answer.") || !text.includes("Second answer.")) throw new Error("earlier answers are not on the page: " + text);
}

export async function testWorkModeOutsideFolderIsSaidPlainly({ page, origin }) {
  await installAgentStream(page);
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  await page.route("**/v1/agent/run-4/confirm", route => route.fulfill(json({ ok: true })));
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.fill("#work-root", "/work/demo");
  await page.fill("#work-task", "create fruit.txt");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__push);
  await page.evaluate(() => {
    window.__push({ type: "tool_call", id: "c1", name: "write", arguments: { filePath: "/work/fruit.txt", content: "apple" } });
    window.__push({ type: "confirm_required", id: "per_4", run_id: "run-4", tool: "external_directory", arguments: { patterns: ["/work/*"] } });
  });
  await page.waitForSelector(".work-sheet");
  const sheet = await page.locator(".work-sheet").innerText();
  if (!sheet.includes("outside your project folder") || !sheet.includes("/work/fruit.txt")) throw new Error("sheet hides that the path is outside: " + sheet);
  if (sheet.includes("in your project folder")) throw new Error("sheet still claims the path is in the project folder");
  await page.getByRole("button", { name: "Allow", exact: true }).click();
  await page.evaluate(() => {
    window.__push({ type: "tool_result", id: "c1", name: "write", content: "ok", error: false });
    window.__push({ type: "changes", run_id: "run-4", undoable: false, truncated: false, files: [] });
    window.__push({ type: "done" }); window.__end();
  });
  await page.waitForFunction(() => document.querySelector("#work-run")?.textContent.includes("Written outside your project folder"));
  if (await page.locator(".work-result").count()) throw new Error("an empty '0 files changed' card was shown");
}

export async function testWorkModeCompactReplacesEarlierTurns({ page, origin }) {
  await installAgentStream(page);
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  let summaryRequests = 0;
  await page.route("**/v1/chat/completions", route => {
    summaryRequests += 1;
    route.fulfill(json({ choices: [{ message: { content: "Made hello.txt." }, finish_reason: "stop" }] }));
  });
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.fill("#work-root", "/work/demo");

  await page.fill("#work-task", "/compact");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => document.querySelector("#work-hint")?.textContent.includes("Nothing to compact"));
  if (summaryRequests !== 0 || (await page.evaluate(() => !!window.__agentBody))) throw new Error("/compact with no history made a request");

  await page.fill("#work-task", "Create hello.txt containing hi");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__push);
  await page.evaluate(() => { window.__push({ type: "assistant", content: "First answer." }); window.__push({ type: "done" }); window.__end(); window.__push = null; });
  await page.waitForFunction(() => document.querySelector(".work-eyebrow")?.textContent.includes("finished"));

  await page.fill("#work-task", "/compact keep file names");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => document.querySelector("#work-hint")?.textContent.includes("Compacted 1 earlier task"));
  if (summaryRequests !== 1) throw new Error("expected one summary request, saw " + summaryRequests);
  if (await page.evaluate(() => window.__agentBody.task) !== "Create hello.txt containing hi") throw new Error("/compact was sent to the agent as a task");

  await page.fill("#work-task", "Now add a second file");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => window.__agentBody.task === "Now add a second file");
  const history = await page.evaluate(() => window.__agentBody.history);
  if (history.length !== 1 || !history[0].content.includes("Made hello.txt.") || history[0].content.includes("First answer.")) {
    throw new Error("history was not replaced by the summary: " + JSON.stringify(history));
  }
  if ((await page.locator(".work-turn").count()) !== 2) throw new Error("the visible conversation changed");
}

const COMMANDS = [
  { name: "review", description: "review changes [commit|branch|pr]", source: "command", takes_arguments: true },
  { name: "pdf", description: "Work with PDF files", source: "skill", takes_arguments: false },
];

export async function testWorkSlashMenuListsHarnessCommands({ page, origin }) {
  await installAgentStream(page);
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  const asked = [];
  await page.route("**/v1/agent/commands*", route => { asked.push(route.request().url()); route.fulfill(json({ commands: COMMANDS })); });
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.fill("#work-root", "/work/demo");

  await page.click("#work-task");
  await page.keyboard.type("/");
  await page.waitForFunction(() => document.querySelectorAll("#work-commands .work-command").length === 4);
  const names = await page.locator("#work-commands .work-command b").allInnerTexts();
  if (names.join(",") !== "/compact,/review,/sessions,/pdf") throw new Error("menu shows " + names);
  if (!asked.length || !asked[0].includes("harness=opencode") || !asked[0].includes("%2Fwork%2Fdemo")) throw new Error("commands were not requested for this agent and folder: " + asked);

  await page.keyboard.type("rev");
  await page.waitForFunction(() => document.querySelectorAll("#work-commands .work-command").length === 1);
  await page.keyboard.press("Enter");                         // picks the command, does not send it
  if ((await page.inputValue("#work-task")) !== "/review ") throw new Error("Enter did not complete the command");
  if (await page.evaluate(() => !!window.__agentBody)) throw new Error("picking a command started a run");
  await page.keyboard.press("Escape");
}

export async function testWorkSlashCommandRunsInTheHarness({ page, origin }) {
  await installAgentStream(page);
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  await page.route("**/v1/agent/commands*", route => route.fulfill(json({ commands: COMMANDS })));
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.fill("#work-root", "/work/demo");

  await page.fill("#work-task", "/review main");
  await page.press("#work-task", "Enter");                    // a complete command name sends
  await page.waitForFunction(() => !!window.__agentBody);
  const body = await page.evaluate(() => window.__agentBody);
  if (body.command !== "review" || body.arguments !== "main" || body.harness !== "opencode") throw new Error("command not sent: " + JSON.stringify(body));
  await page.evaluate(() => { window.__push({ type: "assistant", content: "Reviewed." }); window.__push({ type: "done" }); window.__end(); window.__push = null; window.__agentBody = null; });
  await page.waitForFunction(() => document.querySelector(".work-eyebrow")?.textContent.includes("finished"));

  // Text that merely starts with a slash, but is not a known command, is an ordinary task.
  await page.fill("#work-task", "/etc/hosts has a typo");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__agentBody);
  const plain = await page.evaluate(() => window.__agentBody);
  if (plain.command !== undefined || plain.task !== "/etc/hosts has a typo") throw new Error("unknown slash text was treated as a command: " + JSON.stringify(plain));
}

export async function testWorkSessionsAreSavedAndReopened({ page, origin }) {
  await installAgentStream(page);
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  const saved = new Map();
  await page.route("**/v1/agent/sessions**", route => {
    const request = route.request();
    const id = decodeURIComponent(new URL(request.url()).pathname.split("/v1/agent/sessions")[1] || "").replace(/^\//, "");
    if (request.method() === "PUT") {
      const body = JSON.parse(request.postData());
      saved.set(id, { id, ...body, updated: Date.now() / 1000 });
      return route.fulfill(json({ id }));
    }
    if (request.method() === "DELETE") { saved.delete(id); return route.fulfill(json({ deleted: true })); }
    if (id) return route.fulfill(saved.has(id) ? json(saved.get(id)) : { status: 404, contentType: "application/json", body: "{}" });
    return route.fulfill(json({ sessions: [...saved.values()].map(s => ({ id: s.id, title: s.title, folder: s.folder, updated: s.updated, tasks: s.runs.length })) }));
  });
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.fill("#work-root", "/work/demo");
  if (!(await page.locator("#work-recent").isHidden())) throw new Error("Recent shown with no saved sessions");

  await page.fill("#work-task", "Create hello.txt containing hi");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__push);
  await page.evaluate(() => { window.__push({ type: "assistant", content: "## Done\n\nCreated `hello.txt`:\n\n```js\nconsole.log('hi')\n```" }); window.__push({ type: "done" }); window.__end(); window.__push = null; });
  await page.waitForFunction(() => document.querySelector(".work-eyebrow")?.textContent.includes("finished"));
  // The answer is formatted, not shown as raw markdown.
  await page.waitForSelector(".work-say h2");
  await page.waitForSelector(".work-say .code-block-wrapper pre code");
  if ((await page.locator(".work-say").innerText()).includes("```")) throw new Error("markdown fences are showing");
  await page.waitForFunction(() => document.querySelector("#work-hint") !== null);
  await page.waitForTimeout(150);
  if (saved.size !== 1) throw new Error("the session was not saved");

  // Start over, find it under Recent, and reopen it.
  await page.click("#work-new");
  await page.waitForSelector("#work-recent:not([hidden]) .work-recent-item");
  if (!(await page.locator(".work-recent-item").innerText()).includes("Create hello.txt containing hi")) throw new Error("Recent does not name the task");
  await page.click(".work-recent-item");
  await page.waitForSelector(".work-turn");
  if (!(await page.locator("#work-run").innerText()).includes("Create hello.txt containing hi")) throw new Error("the task is missing after reopening");
  await page.waitForSelector(".work-say h2");

  // A follow-up in the reopened session carries the earlier turn and saves into the same session.
  await page.fill("#work-task", "Now add a second file");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => window.__agentBody && window.__agentBody.task === "Now add a second file");
  const history = await page.evaluate(() => window.__agentBody.history);
  if (history.length !== 2 || history[0].content !== "Create hello.txt containing hi") throw new Error("reopened history was not sent: " + JSON.stringify(history));
  await page.waitForTimeout(150);
  if (saved.size !== 1 || [...saved.values()][0].runs.length !== 2) throw new Error("the follow-up did not save into the same session");

  // Delete from Recent.
  await page.evaluate(() => { window.__push({ type: "done" }); window.__end(); });
  await page.waitForFunction(() => document.querySelectorAll(".work-eyebrow")[1]?.textContent.includes("finished"));
  await page.click("#work-new");
  await page.waitForSelector(".work-recent-item");
  await page.hover(".work-recent-row");
  await page.click(".work-recent-remove");
  await page.waitForSelector("#work-recent", { state: "hidden" });
  if (saved.size !== 0) throw new Error("delete did not reach the server");
}

export async function testWorkToolProfilesLimitWhatARunCanUse({ page, origin }) {
  await installAgentStream(page);
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  const profiles = new Map();
  const commandRequests = [];
  await page.route("**/v1/agent/capabilities*", route => route.fulfill(json({
    mcp: ["searxng", "github"], skills: [{ name: "pdf", description: "Work with PDFs" }, { name: "docs", description: "Write documents" }], web: ["webfetch"],
    tools: [
      { id: "bash", group: "builtin" }, { id: "read", group: "builtin" }, { id: "edit", group: "builtin" }, { id: "write", group: "builtin" }, { id: "apply_patch", group: "builtin" },
      { id: "searxng_web_search", group: "searxng" }, { id: "github_create_pull_request", group: "github" }, { id: "github_list_issues", group: "github" },
      { id: "quota_status", group: "plugin" },
    ],
  })));
  await page.route("**/v1/agent/commands*", route => { commandRequests.push(route.request().url()); route.fulfill(json({ commands: [] })); });
  await page.route("**/v1/agent/profiles**", route => {
    const request = route.request();
    const id = new URL(request.url()).pathname.split("/v1/agent/profiles/")[1];
    if (request.method() === "PUT") { const body = JSON.parse(request.postData()); profiles.set(id, { id, description: "", ...body }); return route.fulfill(json(profiles.get(id))); }
    if (request.method() === "DELETE") { profiles.delete(id); return route.fulfill(json({ deleted: true })); }
    return route.fulfill(json({ profiles: [...profiles.values()] }));
  });
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.fill("#work-root", "/work/demo");
  await page.waitForSelector("#work-profile-chip:not([hidden])");
  if ((await page.locator("#work-profile option").allInnerTexts()).join("|") !== "All tools|Manage profiles…") throw new Error("unexpected tool choices");

  // With no profile yet, choosing Manage opens a new one.
  await page.selectOption("#work-profile", "__manage");
  await page.waitForSelector("#work-manager:not([hidden]) input[type=text]");
  const boxes = await page.locator("#work-manager input[type=checkbox]").count();
  // 2 servers, 2 skills, web, 3 built-in (edit/write counts as one), 3 server tools, 1 plugin tool
  if (boxes !== 12) throw new Error(`expected 12 boxes, saw ${boxes}`);
  await page.locator("#work-manager label.tick", { hasText: "pdf" }).first().waitFor();
  if (!(await page.locator(".work-count").innerText()).includes("6 of 9 tools")) throw new Error("count is wrong before choosing: " + await page.locator(".work-count").innerText());
  await page.fill("#work-manager input[type=text]", "Research");
  await page.locator("#work-manager label.tick", { hasText: "searxng" }).locator("input").check();
  await page.locator("#work-manager label.tick", { hasText: "pdf" }).locator("input").check();
  await page.locator("#work-manager label.tick", { hasText: "Web search" }).locator("input").check();
  // Individual tools: turn off shell commands, file editing, one GitHub-less plugin tool.
  await page.locator("#work-manager fieldset", { hasText: "Built-in tools" }).locator("label.tick", { hasText: "bash" }).locator("input").uncheck();
  await page.locator("#work-manager fieldset", { hasText: "Built-in tools" }).locator("label.tick", { hasText: "edit and write files" }).locator("input").uncheck();
  await page.locator("#work-manager fieldset", { hasText: "Plugin tools" }).locator("label.tick", { hasText: "quota_status" }).locator("input").uncheck();
  await page.waitForFunction(() => document.querySelector(".work-count").textContent.includes("2 of 9 tools"));
  await page.click("#work-manager .work-btn.primary");
  await page.waitForFunction(() => document.querySelector("#work-profile").selectedOptions[0]?.textContent === "Research");
  const saved = [...profiles.values()][0];
  if (JSON.stringify([saved.mcp, saved.skills, saved.web]) !== JSON.stringify([["searxng"], ["pdf"], true])) throw new Error("saved the wrong tools: " + JSON.stringify(saved));
  if (JSON.stringify([...saved.tools_off].sort()) !== JSON.stringify(["apply_patch", "bash", "edit", "quota_status", "write"])) throw new Error("tools_off is wrong: " + JSON.stringify(saved.tools_off));
  await page.waitForSelector("#work-manager[hidden]", { state: "attached" });

  // The run is limited to that profile, and the turn says so.
  await page.fill("#work-task", "Find recent papers");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__agentBody);
  const body = await page.evaluate(() => window.__agentBody);
  if (body.tool_profile_id !== saved.id) throw new Error("the run did not carry the profile: " + JSON.stringify(body));
  if (!(await page.locator(".work-eyebrow").innerText()).includes("Research")) throw new Error("the turn does not name its profile");
  await page.evaluate(() => { window.__push({ type: "done" }); window.__end(); window.__push = null; window.__agentBody = null; });
  await page.waitForFunction(() => document.querySelector(".work-eyebrow")?.textContent.includes("finished"));

  // Command suggestions are for the same profile; "All tools" sends no profile.
  await page.fill("#work-task", "/");
  await page.waitForFunction(() => true);
  await page.waitForTimeout(150);
  if (!commandRequests.some(url => url.includes("profile_id=" + encodeURIComponent(saved.id)))) throw new Error("commands were not requested for the profile: " + commandRequests);
  await page.fill("#work-task", "");
  await page.selectOption("#work-profile", "");
  await page.fill("#work-task", "Another task");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__agentBody);
  if ((await page.evaluate(() => window.__agentBody)).tool_profile_id !== undefined) throw new Error("All tools still sent a profile");
}

export async function testWorkResumeAndVisibleContext({page,origin}) {
  await installAgentStream(page);
  await page.route('**/v1/agent/info',route=>route.fulfill(json(INFO)));
  const records=Array.from({length:10},(_,i)=>({id:`saved-${i}`,title:`Saved task ${i}`,folder:'/work/saved',model:'gemma',harness:'opencode',updated:10-i,tasks:1,
    runs:[{id:`old-${i}`,task:`Saved task ${i}`,model:'gemma',harness:'opencode',folder:'/work/saved',folderName:'saved',status:'done',startedAt:1000,endedAt:2000,steps:[],answer:[`Original saved answer ${i}`],error:''}]}));
  await page.route('**/v1/agent/sessions**',route=>{
    const id=new URL(route.request().url()).pathname.split('/v1/agent/sessions/')[1];
    if(route.request().method()==='PUT')return route.fulfill(json({id}));
    return route.fulfill(json(id?records.find(r=>r.id===id):{sessions:records}));
  });
  await page.route('**/v1/chat/budget',route=>{
    const body=route.request().postDataJSON();
    const summarized=body.messages.some(m=>m.content.includes('Summary of the earlier work'));
    return route.fulfill(json({prompt_tokens:summarized?200:1000,reserved_tokens:1,total_tokens:summarized?201:1001,context_tokens:body.model==='gemma'?4096:8192,exact:true}));
  });
  let summaryBody;
  await page.route('**/v1/chat/completions',route=>{
    summaryBody=route.request().postDataJSON();
    return route.fulfill(json({choices:[{message:{content:'We saved the earlier task and answer. Continue the saved project.'},finish_reason:'stop'}]}));
  });
  await openChat(page,origin);
  if(!(await page.locator('#history-toggle').innerText()).includes('History'))throw new Error('chat history is still an unlabeled icon');
  await page.locator('#mode-switch:not([hidden])').waitFor({state:'attached'});
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.getByRole('button',{name:'Work sessions',exact:true}).click();
  await page.waitForFunction(()=>document.querySelectorAll('.paper-work-review .work-recent-item').length===10);
  await page.locator('.paper-work-review .work-recent-item').filter({hasText:'Saved task 9'}).click();
  await page.waitForSelector('.work-turn');
  if(await page.locator('.paper-work-review').isVisible())throw new Error('sessions panel stayed over the resumed task');
  if((await page.inputValue('#work-model'))!=='gemma')throw new Error('saved model was not restored');
  if(!(await page.locator('#work-run').innerText()).includes('Original saved answer 9'))throw new Error('older session did not resume');
  await page.waitForFunction(()=>document.querySelector('#work-context-label').textContent.includes('1,000 / 4,096'));
  await page.fill('#work-task','Keep this unsent draft');
  await page.getByRole('button',{name:'Compact session',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#work-hint').textContent.startsWith('Compacted'));
  if(summaryBody.max_tokens!==512||summaryBody.model!=='gemma')throw new Error('summary used a hard-coded model context');
  if((await page.inputValue('#work-task'))!=='Keep this unsent draft')throw new Error('compaction erased the draft');
  await page.waitForFunction(()=>document.querySelector('#work-context-label').textContent.includes('200 / 4,096'));
  await page.press('#work-task','Enter');await page.waitForFunction(()=>!!window.__push);
  if(!(await page.getByRole('button',{name:'Compact session',exact:true}).isDisabled()))throw new Error('compaction enabled during a task');
  await page.evaluate(()=>window.__push({type:'usage',prompt_tokens:3230,completion_tokens:370}));
  await page.waitForFunction(()=>document.querySelector('#work-context-label').textContent.includes('Last model call: 3,600 / 4,096 tokens (88%)'));
  if(!(await page.locator('#work-context').getAttribute('class')).includes('near-limit'))throw new Error('no near-limit warning');
  await page.evaluate(()=>{window.__push({type:'done'});window.__end();});
  await page.waitForFunction(()=>document.querySelector('#work-context-label').textContent.startsWith('Request text'));
}

export async function testWorkContextIgnoresStaleBudgets({page,origin}) {
  await page.route('**/v1/agent/info',route=>route.fulfill(json(INFO)));
  let release;const held=new Promise(r=>{release=r;});let began=false;
  await page.route('**/v1/chat/budget',async route=>{
    const body=route.request().postDataJSON();
    if(body.messages.some(m=>m.content==='Hold old count')){began=true;await held;return route.fulfill(json({prompt_tokens:7000,context_tokens:8192})).catch(()=>{});}
    return route.fulfill(json({prompt_tokens:123,context_tokens:body.model==='gemma'?4096:8192}));
  });
  await openChat(page,origin);await page.locator('#mode-switch:not([hidden])').waitFor({state:'attached'});
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.fill('#work-task','Hold old count');
  await page.waitForFunction(()=>document.querySelector('#work-task').value==='Hold old count');
  for(let i=0;i<30&&!began;i++)await page.waitForTimeout(20);
  if(!began)throw new Error('budget request did not start');
  await page.fill('#work-task','New draft');
  await page.locator('#paper-model-pill').click();await page.locator('.paper-model-option[data-model="gemma"]').click();
  release();
  await page.waitForFunction(()=>document.querySelector('#work-context-label').textContent.includes('123 / 4,096'));
  await page.waitForTimeout(100);
  if((await page.locator('#work-context-label').innerText()).includes('7,000'))throw new Error('stale response replaced the current context');
}

export async function testWorkLiveActivityAndFullPrompt({page,origin}) {
  await installAgentStream(page);
  await page.route('**/v1/agent/info',r=>r.fulfill(json(INFO)));
  await page.route('**/v1/chat/budget',r=>r.fulfill(json({prompt_tokens:200,context_tokens:8192,exact:true,scope:r.request().postDataJSON().work?.harness==='builtin'?'full_prompt':'request_only'})));
  await openChat(page,origin);await page.locator('#mode-switch:not([hidden])').waitFor({state:'attached'});
  await page.locator('.paper-rail a[aria-label="Work"]').click();await page.fill('#work-root','/work/demo');
  await page.waitForFunction(()=>document.querySelector('#work-context-label').textContent.includes('setup not measured'));
  if(await page.locator('#work-context-track').isVisible())throw new Error('partial history count displayed as full prompt percentage');
  await page.fill('#work-task','Inspect the project');await page.press('#work-task','Enter');await page.waitForFunction(()=>!!window.__push);
  await page.evaluate(()=>{window.__push({type:'activity',phase:'generating'});window.__push({type:'context',prompt_tokens:5200,context_tokens:8192,exact:true,setup_tokens:5000,source:'agent_request'});});
  await page.waitForFunction(()=>document.querySelector('#work-context-label').textContent.includes('Current prompt: 5,200 / 8,192'));
  await page.evaluate(()=>window.__push({type:'activity',phase:'thinking'}));
  await page.getByRole('status').filter({hasText:'Thinking'}).waitFor();
  if(!(await page.locator('.work-step-title').innerText()).includes('Thinking'))throw new Error('initial placeholder did not follow the live phase');
  await page.evaluate(()=>window.__push({type:'tool_call',id:'a',name:'read_file',arguments:{path:'README.md'}}));
  await page.waitForFunction(()=>document.querySelector('#work-activity-label').textContent==='Running tools');
  await page.evaluate(()=>window.__push({type:'tool_result',id:'a',name:'read_file',content:'Read it',error:false}));
  await page.waitForFunction(()=>document.querySelector('#work-activity-label').textContent==='Generating');
  if(!(await page.locator('#work-activity').isVisible()))throw new Error('activity disappeared after a tool completed');
  await page.evaluate(()=>{window.__push({type:'activity',phase:'generating'});window.__push({type:'context',prompt_tokens:7200,context_tokens:8192,exact:true,source:'agent_request'});window.__push({type:'activity',phase:'answering'});});
  await page.waitForFunction(()=>document.querySelector('#work-activity-label').textContent==='Writing answer');
  if(!(await page.locator('#work-context').getAttribute('class')).includes('near-limit'))throw new Error('actual full prompt did not warn near limit');
  await page.evaluate(()=>{window.__push({type:'assistant',content:'Inspected.'});window.__push({type:'usage',prompt_tokens:7200,completion_tokens:20});window.__push({type:'done'});window.__end();});
  await page.locator('#work-activity').waitFor({state:'hidden'});
  await page.waitForFunction(()=>document.querySelector('#work-context-label').textContent.includes('Next prompt estimate: ~5,200 / 8,192'));
  await page.selectOption('#work-harness','builtin');
  await page.waitForFunction(()=>document.querySelector('#work-context-label').textContent.includes('Next prompt estimate: ~200 / 8,192'));
}

export async function testChatLiveActivity({page,origin}) {
  await page.addInitScript(()=>{const real=window.fetch.bind(window);window.fetch=(input,init={})=>{
    if(typeof input==='string'&&input.endsWith('/v1/chat/completions')&&init.method==='POST'){
      const encoder=new TextEncoder();return Promise.resolve(new Response(new ReadableStream({start(c){window.__chatPush=event=>c.enqueue(encoder.encode('data: '+JSON.stringify(event)+'\n\n'));window.__chatEnd=()=>{c.enqueue(encoder.encode('data: [DONE]\n\n'));c.close();};}}),{headers:{'Content-Type':'text/event-stream'}}));
    }return real(input,init);};});
  await openChat(page,origin);await page.fill('#message-input','Explain it');await page.press('#message-input','Enter');
  await page.locator('.message.assistant .live-activity').filter({hasText:'Generating'}).waitFor();
  await page.evaluate(()=>window.__chatPush({choices:[{delta:{reasoning_content:'Reasoning from the test model'}}]}));
  await page.locator('.message.assistant .live-activity').filter({hasText:'Thinking'}).waitFor();
  await page.evaluate(()=>window.__chatPush({choices:[{delta:{content:'<think>Test reasoning in a tagged stream'}}]}));
  await page.locator('.message.assistant .live-activity').filter({hasText:'Thinking'}).waitFor();
  await page.evaluate(()=>window.__chatPush({choices:[{delta:{content:'</think>Here is the answer.'}}]}));
  await page.locator('.message.assistant .live-activity').filter({hasText:'Writing answer'}).waitFor();
  await page.evaluate(()=>{window.__chatPush({choices:[{delta:{},finish_reason:'stop'}],usage:{prompt_tokens:12,completion_tokens:8,total_tokens:20}});window.__chatEnd();});
  await page.locator('.message.assistant .live-activity').waitFor({state:'detached'});
  if(!(await page.locator('.message.assistant .content').innerText()).includes('Here is the answer'))throw new Error('activity interrupted answer streaming');
}

export async function testWorkHermesHarness({page,origin}) {
  await installAgentStream(page);
  await page.addInitScript(()=>sessionStorage.setItem('arc-llama-work-auto','1'));
  const hermes={id:'hermes',label:'Hermes',available:true,auto_approve:false,approval_note:'Hermes asks before file edits and commands flagged by its own checks. Auto-approve is unavailable.'};
  await page.route('**/v1/agent/info',route=>route.fulfill(json({...INFO,harnesses:[...INFO.harnesses,hermes]})));
  const saved=new Map(),confirmations=[];
  await page.route('**/v1/agent/sessions**',route=>{
    const id=new URL(route.request().url()).pathname.split('/sessions/')[1];
    if(route.request().method()==='PUT'){saved.set(id,{id,...route.request().postDataJSON()});return route.fulfill(json({id}));}
    return route.fulfill(json(id?saved.get(id):{sessions:[...saved.values()]}));
  });
  await page.route('**/v1/agent/hermes-1/confirm',route=>{confirmations.push(route.request().postDataJSON());return route.fulfill(json({ok:true}));});
  await page.route('**/v1/chat/budget',route=>route.fulfill(json({prompt_tokens:180,context_tokens:8192,reserved_tokens:1024,exact:false,scope:'request_only'})));
  await openChat(page,origin);
  await page.waitForSelector('#mode-switch:not([hidden])',{state:'attached'});
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.selectOption('#work-harness','hermes');
  if(!(await page.locator('#work-auto').isDisabled()))throw new Error('Hermes incorrectly offers auto-approve');
  if(!(await page.locator('#work-profile-chip').isHidden()))throw new Error('OpenCode-only profiles are shown for Hermes');
  if(!(await page.locator('#work-hint').innerText()).includes('flagged'))throw new Error('Hermes approval policy is not explained');
  await page.waitForFunction(()=>document.querySelector('#work-context-note')?.textContent.includes('partial count'));
  if((await page.locator('#work-context-note').innerText()).includes('OpenCode'))throw new Error('Hermes context note names another harness');
  await page.fill('#work-root','/work/demo');await page.fill('#work-task','Write a note');await page.press('#work-task','Enter');
  await page.waitForFunction(()=>!!window.__agentBody);
  const body=await page.evaluate(()=>window.__agentBody);
  if(body.harness!=='hermes'||body.auto_confirm!==false||body.local_only!==true||body.tool_profile_id!==undefined)throw new Error('wrong Hermes request: '+JSON.stringify(body));
  await page.evaluate(()=>{
    window.__push({type:'activity',phase:'thinking'});
    window.__push({type:'context',prompt_tokens:4600,context_tokens:8192,exact:true,source:'agent_request'});
  });
  await page.waitForFunction(()=>document.querySelector('#work-activity-label').textContent==='Thinking');
  await page.evaluate(()=>{window.__push({type:'assistant_delta',content:'I will '});window.__push({type:'assistant_delta',content:'write the note.'});});
  await page.waitForFunction(()=>document.querySelector('.work-say')?.textContent.includes('I will write the note.'));
  if(await page.locator('.work-say').count()!==1 || await page.locator('.work-step.running').count())throw new Error('Hermes streaming duplicated the answer or kept a placeholder');
  await page.evaluate(()=>{
    window.__push({type:'tool_call',id:'write-1',name:'write_file',arguments:{path:'/work/demo/note.txt',content:'hi'}});
    window.__push({type:'confirm_required',id:'write-1',run_id:'hermes-1',tool:'write_file',arguments:{path:'/work/demo/note.txt',content:'hi'}});
  });
  await page.waitForSelector('.work-sheet');
  await page.locator('.work-sheet .work-btn.primary').click();
  await page.waitForFunction(()=>document.querySelector('.work-sheet')===null);
  if(confirmations.length!==1||confirmations[0].approved!==true)throw new Error('Hermes approval was not sent');
  const savedDone=page.waitForResponse(response=>response.url().includes('/v1/agent/sessions/')&&response.request().method()==='PUT'&&response.request().postDataJSON().runs?.[0]?.status==='done');
  await page.evaluate(()=>{
    window.__push({type:'tool_result',id:'write-1',name:'write_file',content:'written',error:false});
    window.__push({type:'assistant_delta',content:'The note '});window.__push({type:'assistant_delta',content:'is ready.'});
    window.__push({type:'done'});window.__end();
  });
  await page.waitForFunction(()=>document.querySelector('.work-eyebrow')?.textContent.includes('finished'));
  if(!(await page.locator('.work-say').first().innerText()).includes('The note is ready.'))throw new Error('Hermes answer deltas were not combined');
  await savedDone;
  const record=[...saved.values()].find(r=>r.runs?.[0]?.status==='done');
  if(!record||record.harness!=='hermes'||JSON.stringify(record.runs[0].answer)!==JSON.stringify(['I will write the note.','The note is ready.']))throw new Error('Hermes conversation was not persisted correctly');
  await page.reload();await page.locator('#paper-sessions-toggle').click();await page.waitForSelector('#work-recent .work-recent-item');
  await page.locator('#work-recent .work-recent-item').first().click();
  await page.waitForFunction(()=>document.querySelector('.work-say')?.textContent.includes('The note is ready.'));
  if(await page.inputValue('#work-harness')!=='hermes'||!(await page.locator('#work-auto').isDisabled()))throw new Error('saved Hermes session reopened with the wrong harness controls');
  await page.selectOption('#work-harness','opencode');
  if(await page.locator('#work-auto').isDisabled())throw new Error('OpenCode auto-approve remained disabled after switching harness');
  if(await page.locator('#work-profile-chip').isHidden())throw new Error('OpenCode profiles did not return');
}

export async function testWorkHermesUnavailable({page,origin}) {
  await page.route('**/v1/agent/info',route=>route.fulfill(json({...INFO,harnesses:[...INFO.harnesses,{id:'hermes',label:'Hermes',available:false,auto_approve:false}]})));
  await openChat(page,origin);await page.waitForSelector('#work-harness option[value="hermes"]',{state:'attached'});
  if(!(await page.locator('#work-harness option[value="hermes"]').evaluate(node=>node.disabled)))throw new Error('missing Hermes was offered as runnable');
}

export async function testWorkNativeHarnessHistory({ page, origin }) {
  await installAgentStream(page);
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  const original = { id: "oc-ses_fixture", title: "Synthetic OpenCode conversation", folder: "/work/synthetic",
    harness: "opencode", origin: "opencode", native: true, model: "", updated: 1, tasks: 1,
    runs: [{ task: "Remember the synthetic fixture", answer: ["I remember the synthetic fixture."], steps: [],
      harness: "opencode", model: "", folder: "/work/synthetic", folderName: "synthetic", status: "done", imported: true,
      startedAt: 1000, endedAt: 2000 }] };
  let saved = null, deleted = 0;
  await page.route("**/v1/agent/sessions**", route => {
    const request = route.request();
    const id = new URL(request.url()).pathname.split("/sessions/")[1];
    if (request.method() === "DELETE") { deleted++; saved = null; return route.fulfill(json({ deleted: true })); }
    if (request.method() === "PUT") { saved = { ...request.postDataJSON(), id, updated: 2 }; return route.fulfill(json(saved)); }
    return route.fulfill(json(id ? (saved || original) : { sessions: [saved || original], warnings: ["Synthetic history source unavailable"] }));
  });
  await openChat(page, origin);
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  // A local command opens history without starting inference, even with no project folder.
  await page.fill("#work-root", "");
  await page.fill("#work-task", "/sessions");
  await page.press("#work-task", "Enter");
  await page.waitForSelector(".work-recent-item");
  if (await page.evaluate(() => !!window.__agentBody)) throw new Error("/sessions started inference");
  if (!(await page.locator(".work-recent-item").innerText()).includes("OpenCode · Harness history")) throw new Error("native source is not labeled");
  if (await page.locator(".work-recent-remove").count()) throw new Error("native history offers Delete");
  if (!(await page.locator("#work-sessions-warning").innerText()).includes("Synthetic history")) throw new Error("history warning missing");
  await page.locator(".work-recent-item").click();
  await page.waitForSelector(".work-say");
  if (!(await page.locator("#work-run").innerText()).includes("I remember the synthetic fixture.")) throw new Error("transcript missing");
  if ((await page.inputValue("#work-root")) !== "/work/synthetic") throw new Error("project folder missing");
  if (!(await page.locator("#work-hint").innerText()).includes("Work copy")) throw new Error("import behavior is not explained");
  await page.fill("#work-task", "Continue the synthetic task");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__agentBody);
  const body = await page.evaluate(() => window.__agentBody);
  if (body.history.length !== 2 || body.history[0].content !== original.runs[0].task || body.history[1].content !== original.runs[0].answer[0]) throw new Error("native text history not passed to follow-up");
  if (body.harness !== "opencode" || body.root !== "/work/synthetic" || body.local_only !== true) throw new Error("follow-up settings wrong");
  await page.evaluate(() => { window.__push({ type: "done" }); window.__end(); });
  await page.waitForFunction(() => document.querySelectorAll(".work-eyebrow")[1]?.textContent.includes("finished"));
  await page.waitForTimeout(150);
  if (!saved || saved.runs.length !== 2 || saved.origin !== "opencode") throw new Error("Work copy lost history/provenance");
  await page.click("#work-new");
  await page.locator("#paper-sessions-toggle").click();
  await page.waitForSelector(".work-recent-remove");
  await page.locator(".work-recent-remove").click();
  await page.waitForFunction(() => !document.querySelector(".work-recent-remove"));
  if (deleted !== 1 || !(await page.locator(".work-recent-item").innerText()).includes("Harness history")) throw new Error("deleting Work copy removed native original");
}

export async function testWorkGenerationSpeedCounter({ page, origin }) {
  await installAgentStream(page);
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  let saved = null;
  await page.route("**/v1/agent/sessions**", route => {
    const request = route.request(), id = new URL(request.url()).pathname.split("/sessions/")[1];
    if (request.method() === "PUT") saved = { ...request.postDataJSON(), id, updated: Date.now() / 1000 };
    return route.fulfill(json(id ? saved : { sessions: saved ? [saved] : [] }));
  });
  await openChat(page, origin);
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  const speed = page.locator("#work-speed");
  if ((await speed.innerText()) !== "N/A tok/s") throw new Error("speed invented before inference");
  await page.fill("#work-root", "/work/synthetic");
  await page.fill("#work-task", "Synthetic speed counter task");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => !!window.__push);
  await page.evaluate(() => window.__push({ type: "usage", prompt_tokens: 1000, completion_tokens: 100 }));
  if ((await speed.innerText()) !== "N/A tok/s") throw new Error("speed estimated from usage without timing");
  await page.evaluate(() => window.__push({ type: "performance", generation_tok_s: 42.56 }));
  await page.waitForFunction(() => document.querySelector("#work-speed").textContent === "42.6 tok/s");
  await page.evaluate(() => {
    window.__push({ type: "tool_call", id: "synthetic-tool", name: "read_file", arguments: { path: "fixture.txt" } });
    for (const value of [0, -4, "200", null]) window.__push({ type: "performance", generation_tok_s: value });
  });
  await page.waitForTimeout(1100);
  if ((await speed.innerText()) !== "42.6 tok/s") throw new Error("tool time or invalid samples changed measured speed");
  if (!(await speed.getAttribute("title")).includes("excludes loading")) throw new Error("speed scope not explained");
  await page.evaluate(() => {
    window.__push({ type: "performance", generation_tok_s: 35.25 });
    window.__push({ type: "assistant", content: "Synthetic finished answer" });
    window.__push({ type: "done" }); window.__end();
  });
  await page.waitForFunction(() => document.querySelector("#work-speed").textContent === "35.3 tok/s");
  await page.locator("#work-activity").waitFor({ state: "hidden" });
  if (!(await page.locator(".work-eyebrow").innerText()).includes("35.3 tok/s")) throw new Error("completed task lost speed");
  await page.waitForTimeout(100);
  if (saved?.runs[0]?.generationTokS !== 35.25) throw new Error("speed was not saved with the conversation");
  await page.click("#work-new");
  if ((await speed.innerText()) !== "N/A tok/s") throw new Error("new conversation inherited old speed");
  await page.locator("#paper-sessions-toggle").click();
  await page.locator(".work-recent-item").click();
  await page.waitForFunction(() => document.querySelector("#work-speed").textContent === "35.3 tok/s");
  await page.fill("#work-task", "Synthetic second request");
  await page.press("#work-task", "Enter");
  await page.waitForFunction(() => window.__agentBody?.task === "Synthetic second request");
  if ((await speed.innerText()) !== "N/A tok/s") throw new Error("new task displayed old task speed");
  await page.evaluate(() => { window.__push({ type: "done" }); window.__end(); });
}


export async function testWorkProfilesRecoverFromDiscoveryFailure({ page, origin }) {
  const profile = { id: "saved-profile", name: "Safe tools", description: "Keep restrictions", mcp: ["offline-server"], skills: ["offline-skill"], tools_off: ["offline_tool", "bash"], web: false };
  let saved, requests = 0, deleteRequests = 0;
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  await page.route("**/v1/agent/profiles**", route => {
    if (route.request().method() === "PUT") {
      saved = route.request().postDataJSON();
      return route.fulfill(json({ ...saved, id: profile.id }));
    }
    if (route.request().method() === "DELETE") { deleteRequests++; return route.fulfill({ ...json({ detail: "Synthetic delete failure" }), status: 503 }); }
    return route.fulfill(json({ profiles: [profile] }));
  });
  await page.route("**/v1/agent/capabilities*", route => {
    requests++;
    if (requests === 1) return route.abort("failed");
    if (requests === 2) return route.fulfill(json({ mcp: [], skills: [], tools: [], tools_complete: false, warnings: ["Synthetic incomplete catalog"] }));
    return route.fulfill(json({ mcp: [], skills: [], tools: [{ id: "read", group: "builtin" }, { id: "bash", group: "builtin" }], tools_complete: true }));
  });
  await openChat(page, origin);
  await page.locator('#mode-switch:not([hidden])').waitFor({ state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.waitForSelector('#work-profile option[value="saved-profile"]', { state: "attached" });
  await page.selectOption("#work-profile", "saved-profile");
  await page.selectOption("#work-profile", "__manage");
  await page.getByRole("button", { name: "Retry", exact: true }).waitFor();
  if (await page.locator("#work-manager input").count()) throw new Error("failed discovery exposed an editable catalog");
  await page.getByRole("button", { name: "Use selected profile", exact: true }).click();
  if (await page.inputValue("#work-profile") !== profile.id) throw new Error("saved profile unavailable after failure");
  await page.selectOption("#work-profile", "__manage");
  await page.getByRole("button", { name: "Retry", exact: true }).waitFor();
  if (await page.locator("#work-manager input").count()) throw new Error("partial discovery exposed an editable catalog");
  await page.getByRole("button", { name: "Retry", exact: true }).click();
  await page.getByRole("textbox", { name: "Profile name", exact: true }).waitFor();
  if (await page.locator('#work-manager label.tick', { hasText: 'bash' }).locator('input').isChecked()) throw new Error("discovery removed the bash restriction");
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await page.locator('#work-manager [role="alert"]').filter({ hasText: "Synthetic delete failure" }).waitFor();
  if (deleteRequests !== 1 || await page.inputValue("#work-profile") !== profile.id) throw new Error("failed delete changed selection");
  await page.getByRole("button", { name: "Save and use", exact: true }).click();
  await page.waitForSelector("#work-manager[hidden]", { state: "attached" });
  if (saved.description !== profile.description || JSON.stringify(saved.tools_off) !== JSON.stringify(profile.tools_off) || JSON.stringify(saved.mcp) !== JSON.stringify(profile.mcp) || JSON.stringify(saved.skills) !== JSON.stringify(profile.skills)) throw new Error("saving lost unavailable tool restrictions: " + JSON.stringify(saved));
}

export async function testWorkProfilesIgnoreClosedDiscovery({ page, origin }) {
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  await page.route("**/v1/agent/profiles", route => route.fulfill(json({ profiles: [] })));
  await page.addInitScript(() => {
    const real = window.fetch.bind(window);
    window.fetch = (input, init) => {
      if (String(input).includes("/v1/agent/capabilities")) {
        if (!window.__oldDiscovery) return new Promise(resolve => { window.__oldDiscovery = () => resolve(new Response(JSON.stringify({ mcp: ["stale-server"], skills: [], tools: [] }), { status: 200 })); });
        return Promise.resolve(new Response(JSON.stringify({ mcp: ["current-server"], skills: [], tools: [] }), { status: 200 }));
      }
      return real(input, init);
    };
  });
  await openChat(page, origin);
  await page.locator('#mode-switch:not([hidden])').waitFor({ state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.selectOption("#work-profile", "__manage");
  await page.waitForFunction(() => !!window.__oldDiscovery);
  await page.getByRole("button", { name: "Close", exact: true }).click();
  await page.fill("#work-root", "/work/new-folder");
  await page.selectOption("#work-profile", "__manage");
  await page.getByRole("textbox", { name: "Profile name", exact: true }).waitFor();
  await page.evaluate(() => window.__oldDiscovery());
  await page.waitForTimeout(50);
  if (!(await page.locator("#work-manager-body").innerText()).includes("current-server") || (await page.locator("#work-manager-body").innerText()).includes("stale-server")) throw new Error("closed discovery replaced the current folder catalog");
}


export async function testWorkSessionDuplicatesAreExpandable({ page, origin }) {
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  const repeated = Array.from({length: 6}, (_, i) => ({ id: `oc-ses_duplicate${i}`, title: "Synthetic repeated title", folder: "/work/one", harness: "opencode", native: true, updated: 100+i, tasks: 1 }));
  const otherFolder = { ...repeated[0], id: "oc-ses_otherfolder", folder: "/work/two" };
  let opened;
  await page.route("**/v1/agent/sessions**", route => {
    const id = new URL(route.request().url()).pathname.split("/sessions/")[1];
    if (id) { opened = id; return route.fulfill(json({ ...repeated.find(s => s.id === id), runs: [] })); }
    // An exact duplicate ID must also render only once, even if a source repeats it.
    return route.fulfill(json({ sessions: [...repeated, repeated[5], otherFolder] }));
  });
  await openChat(page, origin);
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.getByRole("button", { name: "Work sessions", exact: true }).click();
  await page.waitForSelector('.work-session-duplicates summary');
  const visible = page.locator('.work-recent-item:visible');
  if (await visible.count() !== 2) throw new Error("repeated titles were not grouped or distinct folders were collapsed");
  if (await page.locator('.work-recent-row[data-session-id="oc-ses_duplicate5"]').count() !== 1) throw new Error("duplicate ID rendered twice");
  const group = page.locator('.work-session-group');
  if (!(await group.locator('.work-recent-row').first().getAttribute('data-session-id')).endsWith('5')) throw new Error("group did not prefer latest session");
  await group.locator('summary').click();
  if (await visible.count() !== 7) throw new Error("older sessions were lost");
  await page.locator('[data-session-id="oc-ses_duplicate0"] .work-recent-item').click();
  await page.locator('.paper-work-review').waitFor({ state: "hidden" });
  if (opened !== "oc-ses_duplicate0") throw new Error("older session opened the wrong ID");
}

export async function testWorkRunCarriesItsStableConversationId({ page, origin }) {
  await installAgentStream(page);
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  let savedId;
  await page.route("**/v1/agent/sessions**", route => {
    const id = new URL(route.request().url()).pathname.split("/sessions/")[1];
    if (route.request().method() === "PUT") savedId = id;
    return route.fulfill(json(id ? { id } : { sessions: [] }));
  });
  await openChat(page, origin);
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.fill('#work-root', '/work/synthetic');
  await page.fill('#work-task', 'Synthetic first task');
  const firstSave = page.waitForResponse(r => r.request().method() === "PUT" && r.url().includes("/v1/agent/sessions/"));
  await page.press('#work-task', 'Enter');
  await page.waitForFunction(() => !!window.__agentBody);
  const first = (await page.evaluate(() => window.__agentBody)).work_session_id;
  await firstSave;
  if (!first || first !== savedId) throw new Error("run did not carry its saved Work ID");
  await page.evaluate(() => { window.__push({type:'done'}); window.__end(); window.__agentBody = null; });
  await page.waitForFunction(() => document.querySelector('.work-eyebrow')?.textContent.includes('finished'));
  await page.fill('#work-task', 'Synthetic follow-up');
  await page.press('#work-task', 'Enter');
  await page.waitForFunction(() => !!window.__agentBody);
  if ((await page.evaluate(() => window.__agentBody)).work_session_id !== first) throw new Error("follow-up changed Work identity");
}


export async function testWorkLinkedHarnessSessionsStayAccessible({ page, origin }) {
  await page.route("**/v1/agent/info", route => route.fulfill(json(INFO)));
  const original = { id: "oc-ses_linked", title: "Synthetic task title", harness: "opencode", native: true, updated: 10, folder: "/work/synthetic", tasks: 1 };
  const work = { id: "work-synthetic", title: "Synthetic Work conversation", harness: "opencode", updated: 11, folder: "/work/synthetic", tasks: 2, harness_history: [original] };
  let opened;
  await page.route("**/v1/agent/sessions**", route => {
    const id = new URL(route.request().url()).pathname.split("/sessions/")[1];
    if (id) { opened = id; return route.fulfill(json({ ...original, runs: [] })); }
    return route.fulfill(json({ sessions: [work] }));
  });
  await openChat(page, origin);
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.getByRole('button', { name: 'Work sessions', exact: true }).click();
  await page.locator('.work-session-linked summary').waitFor();
  if (await page.locator('.work-recent-item:visible').count() !== 1) throw new Error('linked originals duplicated top-level Work entry');
  await page.locator('.work-session-linked summary').click();
  await page.locator('[data-session-id="oc-ses_linked"] .work-recent-item').click();
  await page.locator('.paper-work-review').waitFor({ state: 'hidden' });
  if (opened !== original.id) throw new Error('original harness session no longer accessible');
}
