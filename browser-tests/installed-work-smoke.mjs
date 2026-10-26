// Real Work tab against an installed package and a native local model, without API mocks.
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { readFile } from "node:fs/promises";

const origin = process.argv[2], root = process.env.ARC_WORK_SMOKE_ROOT;
if (!origin?.startsWith("http://127.0.0.1:") || !root?.startsWith("/tmp/arc-work-")) throw new Error("Expected isolated smoke paths");
const options = { headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"] };
const browser = await chromium.launch({ ...options, channel: "chrome" }).catch(() => chromium.launch(options));
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", e => errors.push(e.message));
  await page.goto(origin + "/workbench");
  await page.waitForSelector("#workspace-model option", { state: "attached" });
  await page.evaluate(() => { document.querySelector("#project-editor").open = true; });
  await page.click("#new-workspace");
  await page.fill("#workspace-name", "Installed Work validation");
  await page.fill("#workspace-root", root);
  await page.selectOption("#workspace-model", "published-smoke");
  await page.fill("#workspace-instructions", "Use the file tools to make the requested changes. This project is an isolated validation fixture.");
  await page.click("#save-workspace");
  await page.waitForFunction(() => !document.querySelector("#start-job").disabled);
  await page.fill("#task", "Call write_file to create WORK_RESULT.txt with the exact text local worker ready. Use path WORK_RESULT.txt and content local worker ready. Do not modify other files. Finish after the tool succeeds.");
  await page.click("#start-job");
  await page.waitForFunction(() => /^(completed|failed|paused|interrupted):/.test(document.querySelector("#job-state").textContent), null, { timeout: 120000 });
  assert.match(await page.locator("#job-state").innerText(), /^completed:/);
  const copyText = await page.locator("#job-copy").innerText();
  const copy = copyText.replace("Project copy: ", "");
  assert.equal((await readFile(copy + "/WORK_RESULT.txt", "utf8")).trim(), "local worker ready");
  await assert.rejects(readFile(root + "/WORK_RESULT.txt"));
  await page.reload();
  await page.locator(".work-job").first().click();
  await page.click("#review-changes");
  await page.waitForFunction(() => document.querySelector("#diff").textContent.includes("+local worker ready"));
  await page.getByRole("button", { name: "Apply reviewed changes" }).click();
  await page.getByRole("button", { name: "Undo applied changes" }).waitFor();
  assert.equal((await readFile(root + "/WORK_RESULT.txt", "utf8")).trim(), "local worker ready");
  await page.getByRole("button", { name: "Undo applied changes" }).click();
  await page.getByRole("button", { name: "Apply reviewed changes" }).waitFor();
  await assert.rejects(readFile(root + "/WORK_RESULT.txt"));
  assert.equal((await readFile(root + "/README.txt", "utf8")).trim(), "Original project");
  assert.deepEqual(errors, []);
  await page.screenshot({ path: "/tmp/arc-work-installed.png", fullPage: true });
  console.log(JSON.stringify({ work_tab: "pass", native_agent_file_edit: "pass", browser_reload: "pass", reviewed_apply: "pass", undo: "pass", source_preserved: "pass" }));
} finally { await browser.close(); }
