// Records the README demo, one video per shot, against the isolated demo server.
//   node scripts/demo/record.mjs [all|terminal|discover|chat|work]
// Needs scripts/demo/env.sh's setup (see scripts/demo/README.md). Writes $DEMO/rec/<shot>.webm and
// <shot>.json (event times in seconds from the start of that video) for scripts/demo/edit.sh.
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const { chromium } = createRequire(path.join(here, "../../browser-tests/package.json"))("playwright");
const DEMO = process.env.DEMO || "/tmp/arc-demo";
const ORIGIN = "http://127.0.0.1:11437";
const OUT = path.join(DEMO, "rec");
const SIZE = { width: 1440, height: 900 };
const CHAT_PROMPT = "In three short bullet points, why does a mixture-of-experts model run fast on a 24 GB GPU?";
const WORK_PROMPT = 'Add a small script that prints "Hello from Arc". Before writing it, use the question tool to ask me which language to use: Python, Bash or PowerShell.';
const PROJECT = path.join(DEMO, "hello-arc");

const sh = (...args) => execFileSync("bash", ["-c", ...args], { stdio: "inherit" });
const serve = (cmd) => execFileSync(path.join(here, "serve.sh"), [cmd], { stdio: "inherit" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function cleanState() {
  for (const p of ["chats", "checkpoints", "work-sessions", "work-undo", "semantic_index", "perf-history.jsonl"])
    rmSync(path.join(DEMO, "state/arc-llama", p), { recursive: true, force: true });
  rmSync(path.join(DEMO, "data/opencode"), { recursive: true, force: true });
}

// First run: no config at all, started with the real `arc-llama start` (BROWSER is a stub).
function prepFresh() {
  serve("stop");
  rmSync(path.join(DEMO, "config/arc-llama"), { recursive: true, force: true });
  rmSync(path.join(DEMO, "opened-url"), { force: true });
  cleanState();
  serve("start-cmd");
  for (let i = 0; i < 60; i++) {
    try { return readFileSync(path.join(DEMO, "opened-url"), "utf8").trim(); } catch { execFileSync("sleep", ["0.5"]); }
  }
  throw new Error("arc-llama start did not open a browser");
}

// A model registered on the stock SYCL runtime (config saved by the setup steps), nothing loaded yet.
function prepModel() {
  serve("stop");
  mkdirSync(path.join(DEMO, "config/arc-llama"), { recursive: true });
  sh(`cp "${DEMO}/config-model.toml" "${DEMO}/config/arc-llama/config.toml"`);
  cleanState();
  rmSync(PROJECT, { recursive: true, force: true });
  mkdirSync(PROJECT);
  writeFileSync(path.join(PROJECT, "README.md"), "# hello-arc\n\nA tiny demo project.\n");
  serve("start");
}

// Headless video has no pointer, so draw one that follows the mouse and pulses on click.
const CURSOR = `(() => {
  const add = () => {
    if (document.getElementById("__demo_cursor")) return;
    const c = document.createElement("div"); c.id = "__demo_cursor";
    c.style.cssText = "position:fixed;left:0;top:0;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;" +
      "background:rgba(255,255,255,.85);box-shadow:0 0 0 2px rgba(0,0,0,.35);pointer-events:none;z-index:2147483647;" +
      "transition:transform .12s ease;transform:translate(-100px,-100px)";
    document.documentElement.appendChild(c);
    let x = -100, y = -100, s = 1;
    const draw = () => { c.style.transform = "translate(" + x + "px," + y + "px) scale(" + s + ")"; };
    addEventListener("mousemove", (e) => { x = e.clientX; y = e.clientY; draw(); }, true);
    addEventListener("mousedown", () => { s = .7; draw(); }, true);
    addEventListener("mouseup", () => { s = 1; draw(); }, true);
  };
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", add); else add();
})();`;

async function shot(name, body) {
  mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ channel: "chrome", args: ["--no-sandbox", "--disable-dev-shm-usage"] });
  const context = await browser.newContext({
    viewport: SIZE, deviceScaleFactor: 1, colorScheme: "dark", reducedMotion: "no-preference",
    recordVideo: { dir: path.join(OUT, "tmp-" + name), size: SIZE },
  });
  await context.addInitScript(CURSOR);
  const page = await context.newPage();
  const t0 = Date.now();
  const marks = {};
  globalThis.marks_box = (k, b) => { marks[k] = b && { x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.width), h: Math.round(b.height) }; };
  const mark = (k) => { marks[k] = +((Date.now() - t0) / 1000).toFixed(2); console.log(`  ${name}: ${k} @ ${marks[k]}s`); };
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  try {
    await body(page, mark);
    mark("end");
  } finally {
    const video = page.video();
    await context.close();
    await browser.close();
    renameSync(await video.path(), path.join(OUT, name + ".webm"));
    rmSync(path.join(OUT, "tmp-" + name), { recursive: true, force: true });
  }
  if (errors.length) throw new Error(`${name}: page errors: ${errors.join("; ")}`);
  writeFileSync(path.join(OUT, name + ".json"), JSON.stringify(marks, null, 2));
}

async function glide(page, locator, steps = 25) {
  const box = await locator.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps });
}

// Shot 1: the real `arc-llama start` output on a neutral `$` prompt, then the page it opened.
async function terminal(url) {
  const lines = readFileSync(path.join(DEMO, "start.log"), "utf8").split("\n")
    .filter((l) => l && !/^(INFO|\d{4}-\d\d-\d\d)/.test(l)).slice(0, 4);
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const fmt = (l) => l.startsWith("Open Arc Llama:")
    ? `<b>Open Arc Llama:</b> <span class="u">${esc(l.slice(16))}</span>`
    : l.startsWith("Keep this") ? `<span class="d">${esc(l)}</span>` : esc(l);
  const html = `<!doctype html><meta charset="utf-8"><style>
    html,body{margin:0;height:100%;background:#0b0d12}
    .win{position:absolute;inset:70px 120px;background:#14161c;border:1px solid #2a2e38;border-radius:12px;overflow:hidden;box-shadow:0 30px 80px rgba(0,0,0,.5)}
    .bar{height:40px;background:#1b1e26;border-bottom:1px solid #2a2e38;display:flex;align-items:center;gap:8px;padding:0 16px}
    .bar i{width:12px;height:12px;border-radius:50%;background:#3a3f4b}
    pre{margin:0;padding:28px 32px;font:24px/1.6 "DejaVu Sans Mono","Ubuntu Mono",monospace;color:#d7dae0;white-space:pre-wrap}
    .p{color:#7aa2f7}.u{color:#5fd7e8}.d{color:#7d8496}b{color:#fff}
    .c{display:inline-block;width:.6em;height:1.1em;vertical-align:-.15em;background:#d7dae0;animation:k 1s steps(1) infinite}
    @keyframes k{50%{opacity:0}}</style>
    <div class="win"><div class="bar"><i></i><i></i><i></i></div><pre id="t"><span class="p">$</span> <span id="cmd"></span><span class="c"></span></pre></div>`;
  await shot("terminal", async (page, mark) => {
    await page.setContent(html);
    await sleep(900);
    mark("type");
    for (const ch of "arc-llama start") { await page.evaluate((c) => { document.getElementById("cmd").textContent += c; }, ch); await sleep(70); }
    await sleep(350);
    await page.evaluate(() => document.querySelector(".c").remove());
    mark("enter");
    await sleep(700);
    for (const l of lines) { await page.evaluate((h) => { document.getElementById("t").insertAdjacentHTML("beforeend", "\n" + h); }, fmt(l)); await sleep(450); }
    await sleep(1600);
    mark("browser");
    await page.goto(url);
    // Wait for the catalog pick, not just the device line, so the page is settled.
    await page.getByRole("button", { name: "Download and start" }).waitFor({ timeout: 60000 });
    mark("page");
    await sleep(3200);
  });
}

// Shot 2: Discover's shelves picked for the detected card.
async function discover() {
  await shot("discover", async (page, mark) => {
    await page.goto(ORIGIN + "/#discover");
    const need = page.getByText(/^Needs about \d+ GB of 24 GB$/).first();
    await need.waitFor({ timeout: 60000 });
    await sleep(600);
    mark("ready");
    // Ease the shelf heading to the top of the view over 2.5 s.
    await page.evaluate(async () => {
      const target = [...document.querySelectorAll("*")].find((e) => /^Picked for your/.test(e.textContent || "") && e.children.length < 4);
      let s = target.parentElement;
      while (s && !(s.scrollHeight > s.clientHeight + 10 && /auto|scroll/.test(getComputedStyle(s).overflowY))) s = s.parentElement;
      s = s || document.scrollingElement;
      const from = s.scrollTop, to = from + target.getBoundingClientRect().top - 90;
      const t0 = performance.now(), D = 2500;
      await new Promise((done) => {
        const step = (t) => { const k = Math.min(1, (t - t0) / D), e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
          s.scrollTop = from + (to - from) * e; k < 1 ? requestAnimationFrame(step) : done(); };
        requestAnimationFrame(step);
      });
    });
    mark("scrolled");
    await sleep(500);
    // Hover the card's artwork, not its text, and note the card's box so the edit can push in on it.
    const card = need.locator("xpath=ancestor::article[1]");
    const box = await card.boundingBox();
    await glide(page, card.locator(".discover-cover"), 30);
    mark("hover");
    marks_box("zoom_box", box);
    await sleep(2800);
  });
}

// Shot 3: a real answer streamed from the B60; load time is sped up later, streaming is not.
async function chat() {
  await shot("chat", async (page, mark) => {
    await page.goto(ORIGIN + "/chat");
    await page.locator("#message-input").waitFor();
    await sleep(1500);
    mark("ready");
    await glide(page, page.locator("#message-input"));
    await page.click("#message-input");
    await page.keyboard.type(CHAT_PROMPT, { delay: 28 });
    await sleep(400);
    await glide(page, page.locator("#send-button"), 12);
    await page.click("#send-button");
    mark("send");
    await page.mouse.move(1300, 500, { steps: 15 });
    // The reply's own label switches to "Thinking" on the first streamed token.
    await page.waitForFunction(() => [...document.querySelectorAll("[data-phase]")].some((e) => e.dataset.phase === "thinking" || e.dataset.phase === "answering"),
      null, { timeout: 180000, polling: 30 });
    mark("first_token");
    await page.waitForFunction(() => /tok\/s/.test(document.getElementById("ctx-label-tps")?.textContent || ""), null, { timeout: 180000, polling: 30 });
    mark("done");
    const tps = await page.locator("#ctx-label-tps").innerText();
    console.log("  chat:", tps);
    await sleep(1200);
    const ring = page.locator("#paper-context-toggle");
    await glide(page, ring, 20);
    await ring.click();
    await page.locator("#paper-context-pop").getByText(tps).waitFor({ timeout: 5000 });
    mark("tps_shown");
    marks_box("zoom_box", await page.locator("#paper-context-pop").boundingBox());
    await sleep(3000);
  });
}

// Shot 4: Work asks a question with clickable choices and asks before writing or running anything.
async function work() {
  await shot("work", async (page, mark) => {
    await page.goto(ORIGIN + "/work");
    await page.locator('#work-harness option[value="opencode"]').waitFor({ state: "attached", timeout: 30000 });
    await page.fill("#work-root", PROJECT);
    await page.press("#work-root", "Tab");
    await sleep(1200);
    mark("ready");
    await glide(page, page.locator("#work-task"));
    await page.click("#work-task");
    await page.keyboard.type(WORK_PROMPT, { delay: 18 });
    await sleep(400);
    await glide(page, page.locator("#work-send"), 12);
    await page.click("#work-send");
    mark("send");
    await page.mouse.move(1300, 450, { steps: 15 });
    const python = page.locator(".work-choice", { hasText: "Python" }).first();
    await python.waitFor({ timeout: 240000 });
    mark("question");
    await sleep(900);
    await glide(page, python, 25);
    await sleep(500);
    await python.click();
    mark("answered");
    let approvals = 0;
    for (let i = 0; i < 600; i++) {
      if (await page.locator(".work-result").count()) break;
      const allow = page.locator(".work-sheet").getByRole("button", { name: "Allow", exact: true }).first();
      if (await allow.count() && await allow.isEnabled()) {
        mark("approval" + ++approvals);
        await sleep(900);
        await glide(page, allow, 20);
        await sleep(300);
        await allow.click();
        mark("allowed" + approvals);
        await page.mouse.move(1300, 450, { steps: 15 });
      }
      await sleep(200);
    }
    await page.locator(".work-result").first().waitFor({ timeout: 10000 });
    mark("result");
    await sleep(3500);
  });
}

const which = process.argv[2] || "all";
if (["all", "terminal", "discover"].includes(which)) {
  const url = prepFresh();
  if (which !== "discover") await terminal(url);
  if (which !== "terminal") await discover();
}
if (["all", "chat", "work"].includes(which)) {
  prepModel();
  // Load the model before recording: a cold load is just a wait, and the first reply after one
  // shows the loading note in front of the streamed text until it finishes.
  await fetch(ORIGIN + "/v1/chat/completions", { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: "ornith-35b", max_tokens: 1, messages: [{ role: "user", content: "hi" }] }) });
  if (which !== "work") await chat();
  if (which !== "chat") await work();
}
serve("stop");
console.log("Recorded to", OUT);
