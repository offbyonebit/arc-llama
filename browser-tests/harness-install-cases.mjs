// Installing Work agents from the browser: Work's Agent menu and System's Agents list.
// Same contract as the other case files: no model, no server, no real download.
import { openChat } from "./harness.mjs";

const json = (body, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });
const OFFER = {
  installable: true, version: "v1.19.4", license: "MIT", source: "https://github.com/anomalyco/opencode",
  size_note: "about 80 MB download", location: "/state/harnesses/opencode", experimental: false,
};
const missing = { id: "opencode", label: "OpenCode", available: false, source: "missing", path: null, managed: [], install: OFFER };
const installed = { ...missing, available: true, source: "managed", path: "/state/harnesses/opencode/opencode-v1.19.4-linux-x64/opencode",
  managed: [{ version: "v1.19.4", tested: true, path: "/state/harnesses/opencode/opencode-v1.19.4-linux-x64/opencode", installed_at: 1 }] };
const info = (opencode) => ({
  experimental: true, auto_approve: true, root: "/work/demo", home: "/home/demo",
  harnesses: [{ id: "builtin", label: "Built-in", available: true }, opencode],
});

// Routes the install API through a scripted job the test advances by setting window-side state.
async function installJobApi(page, { outcomes }) {
  const state = { posts: [], polls: 0, done: false, installed: false };
  await page.route("**/v1/agent/info", route => route.fulfill(json(info(state.installed ? installed : missing))));
  await page.route("**/admin/harnesses/opencode/install", route => {
    state.posts.push(route.request().postDataJSON());
    return route.fulfill(json({ id: `job${state.posts.length}`, harness: "opencode", status: "queued" }));
  });
  await page.route("**/admin/harnesses/jobs", route => {
    const id = `job${state.posts.length}`;
    const outcome = outcomes[state.posts.length - 1];
    state.polls++;
    if (state.polls < 2) return route.fulfill(json({ jobs: [{ id, harness: "opencode", status: "running", step: "Downloading OpenCode", bytes_done: 40 * 2 ** 20, bytes_total: 80 * 2 ** 20, log: [] }] }));
    state.polls = 0;
    if (outcome === "done") state.installed = true;
    return route.fulfill(json({ jobs: [outcome === "done"
      ? { id, harness: "opencode", status: "done", step: "Done", installed_version: "v1.19.4", tested: true, log: ["ok"] }
      : { id, harness: "opencode", status: "error", step: "Downloading OpenCode", error: "Download of opencode-linux-x64.tar.gz failed: synthetic", log: ["$ download", "synthetic failure"] }] }));
  });
  return state;
}

export async function testWorkAgentInstallFlow({ page, origin }) {
  const state = await installJobApi(page, { outcomes: ["error", "done"] });
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.waitForSelector('#work-harness option[value="opencode"]', { state: "attached" });
  const option = page.locator('#work-harness option[value="opencode"]');
  if (await option.evaluate(node => node.disabled)) throw new Error("an installable agent was offered as unusable");
  if (!(await option.textContent()).includes("install")) throw new Error("the menu did not say the agent can be installed");
  if ((await page.inputValue("#work-harness")) !== "builtin") throw new Error("a missing agent was preselected");

  // Choosing it explains what will be downloaded, and from where, before anything happens.
  await page.selectOption("#work-harness", "opencode");
  await page.waitForSelector(".harness-dialog:not([hidden])");
  const text = await page.locator(".harness-dialog-sheet").textContent();
  for (const want of ["v1.19.4", "MIT", "github.com/anomalyco/opencode", "/state/harnesses/opencode", "about 80 MB"]) {
    if (!text.includes(want)) throw new Error(`install dialog is missing ${want}: ${text}`);
  }
  if ((await page.inputValue("#work-harness")) !== "builtin") throw new Error("the menu switched before the agent was installed");
  if (state.posts.length) throw new Error("the download started without the user confirming");

  // A failure says what went wrong, keeps Work unchanged, and can be retried.
  await page.locator(".harness-dialog .harness-btn.primary", { hasText: "Install" }).click();
  await page.waitForFunction(() => document.querySelector(".harness-dialog")?.textContent.includes("didn’t install"));
  if (JSON.stringify(state.posts[0]) !== JSON.stringify({ version: "pinned" })) throw new Error("tested version not requested: " + JSON.stringify(state.posts));
  if (!(await page.locator(".harness-error").textContent()).includes("synthetic")) throw new Error("the failure reason was not shown");
  await page.locator(".harness-log summary").click();
  if (!(await page.locator(".harness-log pre").textContent()).includes("synthetic failure")) throw new Error("install log was not available");

  await page.locator(".harness-dialog .harness-btn.primary", { hasText: "Try again" }).click();
  await page.waitForFunction(() => document.querySelector(".harness-dialog")?.textContent.includes("is ready"));
  await page.waitForFunction(() => document.querySelector("#work-harness")?.value === "opencode");
  if (!(await page.locator('#work-harness option[value="opencode"]').textContent()).trim().endsWith("OpenCode")) throw new Error("the menu still says install after installing");
  await page.locator(".harness-dialog .harness-btn.primary", { hasText: "Done" }).click();
  if (await page.locator(".harness-dialog").isVisible()) throw new Error("dialog did not close");
}

export async function testWorkAgentInstallCancelKeepsSelection({ page, origin }) {
  const state = await installJobApi(page, { outcomes: ["done"] });
  await openChat(page, origin);
  await page.waitForSelector("#mode-switch:not([hidden])", { state: "attached" });
  await page.locator('.paper-rail a[aria-label="Work"]').click();
  await page.waitForSelector('#work-harness option[value="opencode"]', { state: "attached" });
  await page.selectOption("#work-harness", "opencode");
  await page.waitForSelector(".harness-dialog:not([hidden])");
  await page.keyboard.press("Escape");
  if (await page.locator(".harness-dialog").isVisible()) throw new Error("Escape did not close the install dialog");
  if ((await page.inputValue("#work-harness")) !== "builtin" || state.posts.length) throw new Error("cancel changed the agent or started a download");

  // "Newest" is an explicit, labelled choice.
  await page.selectOption("#work-harness", "opencode");
  await page.locator(".harness-dialog .harness-link", { hasText: "untested" }).click();
  await page.waitForFunction(() => document.querySelector(".harness-dialog")?.textContent.includes("is ready"));
  if (state.posts[0]?.version !== "latest") throw new Error("newest was not requested: " + JSON.stringify(state.posts));
}

export async function testSystemAgentsPanel({ page, origin }) {
  const hermes = { id: "hermes", label: "Hermes", available: true, source: "path", path: "/usr/bin/hermes", managed: [],
    install: { ...OFFER, version: "v0.22.5", source: "https://github.com/NousResearch/hermes-agent" } };
  const removed = [];
  await page.route("**/v1/agent/info", route => route.fulfill(json({ ...info(removed.length ? missing : installed), harnesses: [...info(removed.length ? missing : installed).harnesses, hermes] })));
  await page.route("**/admin/harnesses/opencode/*", route => {
    if (route.request().method() === "DELETE") { removed.push(route.request().url()); return route.fulfill(json({ removed: "v1.19.4" })); }
    return route.fallback();
  });
  page.on("dialog", dialog => dialog.accept());
  await page.goto(`${origin}/#system`);
  await page.waitForSelector("#harness-list .harness-row");
  const rows = await page.locator("#harness-list .harness-row").allTextContents();
  if (rows.length !== 2) throw new Error("expected OpenCode and Hermes rows: " + JSON.stringify(rows));
  if (!rows[0].includes("Installed by Arc Llama: v1.19.4") || !rows[1].includes("/usr/bin/hermes")) throw new Error("agent sources not described: " + JSON.stringify(rows));
  if (rows.join(" ").includes("Built-in")) throw new Error("the built-in agent was listed as installable");
  await page.locator("#harness-list .harness-btn", { hasText: "Remove v1.19.4" }).click();
  await page.waitForFunction(() => document.querySelector("#harness-list")?.textContent.includes("Not installed"));
  if (!removed[0]?.endsWith("/admin/harnesses/opencode/v1.19.4")) throw new Error("remove did not target the managed version: " + JSON.stringify(removed));
  if (!(await page.locator("#harness-list .harness-btn", { hasText: "Install" }).count())) throw new Error("install was not offered after removal");
}

export async function testSystemAgentsOffWithoutExperimental({ page, origin }) {
  await page.route("**/v1/agent/info", route => route.fulfill(json({ ...info(missing), experimental: false })));
  await page.goto(`${origin}/#system`);
  await page.waitForFunction(() => document.querySelector("#harness-list")?.textContent.includes("ARC_LLAMA_EXPERIMENTAL_AGENT"));
  if (await page.locator("#harness-list .harness-btn").count()) throw new Error("install offered while external agents are off");
}
