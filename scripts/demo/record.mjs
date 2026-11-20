// Records the README demo, one video per shot, against the isolated demo server.
//   node scripts/demo/record.mjs [all|terminal|discover|chat|work]
// Needs scripts/demo/env.sh's setup (see scripts/demo/README.md). Writes $DEMO/rec/<shot>.mp4 (2880x1800)
// and <shot>.json (event times in seconds from the start of that video) for scripts/demo/edit.py.
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const { chromium } = createRequire(path.join(here, "../../browser-tests/package.json"))("playwright");
const DEMO = process.env.DEMO || "/tmp/arc-demo";
const ORIGIN = "http://127.0.0.1:11437";
const OUT = path.join(DEMO, "rec", ...(/^trailer/.test(process.argv[2] || "") ? ["trailer"] : []));
// Raw frames are large; keep them off the root disk.
const FRAMES = process.env.DEMO_FRAMES || "/mnt/storage/tmp/arc-demo-frames";
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
  rmSync(path.join(DEMO, "data/arc-llama/models"), { recursive: true, force: true });
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

// Frames are captured at 2x through Chrome's screencast as quality-100 JPEGs. Playwright's own
// recordVideo writes ~0.5 Mbps real-time VP8, which smears text and blocks dark gradients.
const DSF = 2;

// Chrome sends a frame only when the page repaints, so each frame is held until the next one.
function encodeFrames(dir, frames, end, out) {
  const list = frames.map((f, i) => `file '${f.file}'\nduration ${((frames[i + 1]?.t ?? end) - f.t).toFixed(4)}`);
  list.push(`file '${frames.at(-1).file}'`);
  writeFileSync(path.join(dir, "list.txt"), list.join("\n") + "\n");
  execFileSync("ffmpeg", ["-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", path.join(dir, "list.txt"),
    "-fps_mode", "cfr", "-r", "30", "-c:v", "libx264", "-crf", "4", "-preset", "veryfast", "-pix_fmt", "yuv444p", out], { stdio: "inherit" });
}

async function shot(name, body) {
  mkdirSync(OUT, { recursive: true });
  const frameDir = path.join(FRAMES, name);
  rmSync(frameDir, { recursive: true, force: true });
  mkdirSync(frameDir, { recursive: true });
  // Headless Chrome screencasts at CSS size unless the scale factor is forced at launch.
  const browser = await chromium.launch({ channel: "chrome", args: ["--no-sandbox", "--disable-dev-shm-usage", `--force-device-scale-factor=${DSF}`] });
  const context = await browser.newContext({ viewport: SIZE, deviceScaleFactor: DSF, colorScheme: "dark", reducedMotion: "no-preference" });
  await context.addInitScript(CURSOR);
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  const frames = [];
  let t0 = null;
  cdp.on("Page.screencastFrame", ({ data, metadata, sessionId }) => {
    cdp.send("Page.screencastFrameAck", { sessionId }).catch(() => {});
    if (t0 === null) return;
    const file = path.join(frameDir, String(frames.length).padStart(6, "0") + ".jpg");
    writeFileSync(file, Buffer.from(data, "base64"));
    frames.push({ file, t: Math.max(0, metadata.timestamp - t0) });
  });
  await cdp.send("Page.startScreencast", { format: "jpeg", quality: 100, maxWidth: SIZE.width * DSF, maxHeight: SIZE.height * DSF, everyNthFrame: 1 });
  t0 = Date.now() / 1000;
  const marks = { scale: DSF };
  globalThis.marks_box = (k, b) => { marks[k] = b && { x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.width), h: Math.round(b.height) }; };
  const mark = (k) => { marks[k] = +(Date.now() / 1000 - t0).toFixed(2); console.log(`  ${name}: ${k} @ ${marks[k]}s`); };
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  try {
    await body(page, mark, marks);
    mark("end");
  } finally {
    await cdp.send("Page.stopScreencast").catch(() => {});
    await context.close();
    await browser.close();
  }
  if (errors.length) throw new Error(`${name}: page errors: ${errors.join("; ")}`);
  if (!frames.length) throw new Error(`${name}: no frames captured`);
  frames[0].t = 0;
  encodeFrames(frameDir, frames, marks.end, path.join(OUT, name + ".mp4"));
  console.log(`  ${name}: ${frames.length} frames, ${(frames.length / marks.end).toFixed(1)} fps average`);
  rmSync(frameDir, { recursive: true, force: true });
  writeFileSync(path.join(OUT, name + ".json"), JSON.stringify(marks, null, 2));
}

async function glide(page, locator, steps = 25) {
  const box = await locator.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps });
}

// Shot 1: the real `arc-llama start` output on a neutral `$` prompt, then the page it opened.
async function terminal(url) {
  const lines = readFileSync(path.join(DEMO, "start.log"), "utf8").split("\n")
    .filter((l) => l && !/^(INFO|\d{4}-\d\d-\d\d)/.test(l)).slice(0, which === "trailer" ? 12 : 4);
  if (which === "trailer") {
    const token = await (await fetch(ORIGIN + "/admin/session-token")).json();
    const r = await fetch(ORIGIN + "/admin/onboarding", {
      headers: token.admin_token ? { Authorization: `Bearer ${token.admin_token}` } : {},
    });
    if (!r.ok) throw new Error("Cannot read detected device from running app");
    const { gpu } = await r.json();
    if (!gpu?.name || !gpu.vram_mb || !/B60/i.test(gpu.name)) throw new Error("Missing detected GPU");
    lines.push(`Detected device (live app): ${gpu.name} · ${gpu.vram_mb} MiB`);
  }
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
  await shot("terminal", async (page, mark, marks) => {
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
    if (which === "trailer") {
      const card = page.locator("#t");
      const line = lines.find(l => /Using |Detected .*GPU|Intel.*Arc/i.test(l));
      if (!line) throw new Error("No detected-card line in start.log");
      marks.card_line = line;
      const box = await card.boundingBox();
      marks_box("zoom_box", { ...box, y: box.y + 38 * (lines.indexOf(line) + 1), height: 50 });
    }
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
    await page.getByRole("button", { name: "Download and start" }).waitFor({ timeout: 60000 });
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
async function chat(name = "chat", model = null) {
  await shot(name, async (page, mark, marks) => {
    await page.goto(ORIGIN + "/chat");
    await page.locator("#message-input").waitFor();
    if (which === "trailer") await page.waitForFunction(() => document.querySelector("#model-select")?.value);
    // The 1.2 UI hides the native select behind its own picker; switch it the way the picker does.
    if (model) await page.locator("#model-select").evaluate((s, v) => {
      const available = Array.from(s.options, o => o.value);
      if (!available.includes(v)) throw new Error(`Requested model ${JSON.stringify(v)} is unavailable. Available model values: ${JSON.stringify(available)}`);
      s.value = v; s.dispatchEvent(new Event("change"));
    }, model);
    if (which === "trailer") {
      const status = await appData(page, "/admin/status");
      const selected = await page.locator("#model-select").inputValue();
      const m = status.models.find(m => m.id === selected || m.name === selected);
      const gpu = status.gpus.find(g => g.pci_slot === m?.gpu_pci_slot);
      if (!gpu?.name || !gpu.vram_mb || !/B60/i.test(gpu.name)) throw new Error("Missing measured GPU identity");
      marks.gpu = gpu;
      marks.model = await page.locator("#model-select option:checked").innerText();
      if (name === "bigger" && !/27B/i.test(marks.model)) throw new Error("Optional model must identify itself as 27B");
    }
    await sleep(1500);
    mark("ready");
    await glide(page, page.locator("#message-input"));
    await page.click("#message-input");
    await page.keyboard.type(CHAT_PROMPT, { delay: 28 });
    await sleep(400);
    await glide(page, page.locator("#send-button"), 12);
    const logFile = path.join(DEMO, "state/arc-llama", model + ".log");
    let logOffset = 0;
    if (name === "bigger") {
      try { logOffset = readFileSync(logFile).length; } catch { /* no log yet */ }
    }
    await page.click("#send-button");
    mark("send");
    await page.mouse.move(1300, 500, { steps: 15 });
    // The reply's own label switches to "Thinking" on the first streamed token.
    await page.waitForFunction(() => [...document.querySelectorAll("[data-phase]")].some((e) => e.dataset.phase === "thinking" || e.dataset.phase === "answering"),
      null, { timeout: 180000, polling: 30 });
    mark("first_token");
    await page.waitForFunction(() => /tok\/s/.test(document.getElementById("ctx-label-tps")?.textContent || ""), null, { timeout: 180000, polling: 30 });
    mark("done");
    marks_box("stream_box", await page.locator(".message.assistant").last().boundingBox());
    const tps = await page.locator("#ctx-label-tps").innerText();
    console.log("  chat:", tps);
    const measured = tps.match(/([\d.]+)\s*tok\/s/);
    if (!measured || !(Number(measured[1]) > 0)) throw new Error("No measured tok/s in this take");
    marks.tps = Number(measured[1]);
    marks.tps_label = tps;
    if (name === "bigger") {
      let log = "";
      try { log = readFileSync(logFile).subarray(logOffset).toString("utf8"); } catch { /* omit unverified claim */ }
      const evidence = log.split("\n").filter(l => /(?:drafted|n_drafted|draft accepted|#acc\s+(?:drafts|tokens))\s*[=:]\s*[1-9]\d*|\(\s*[1-9]\d* accepted/i.test(l));
      marks.speculative_active = evidence.length > 0;
      marks.speculative_evidence = evidence;
    }
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
        if (approvals === 1) marks_box("approval_box", await page.locator(".work-sheet").first().boundingBox());
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

// Trailer-only shots reuse the same isolated server and screencast encoder.
async function appData(page, route) {
  const session = await (await page.request.get(ORIGIN + "/admin/session-token")).json();
  const response = await page.request.get(ORIGIN + route, {
    headers: session.admin_token ? { Authorization: `Bearer ${session.admin_token}` } : {},
  });
  if (!response.ok()) throw new Error(`${route}: HTTP ${response.status()}`);
  return response.json();
}

async function firstRun(url) {
  await shot("first-run", async (page, mark, marks) => {
    await page.goto(url);
    const start = page.getByRole("button", { name: "Download and start", exact: true });
    await start.waitFor({ timeout: 60000 });
    marks.version = (await (await page.request.get(ORIGIN + "/openapi.json")).json()).info.version;
    marks.hardware = await page.locator('[data-setup="hardware"]').innerText();
    marks.starter = await page.locator('[data-setup="model"]').innerText();
    if (!/Ornith.*9B/i.test(marks.starter)) throw new Error("Expected the Ornith 9B starter; inspect catalog selection");
    mark("ready");
    marks_box("button_box", await start.boundingBox());
    await sleep(1200);
    await start.click();
    mark("download");
    await page.locator("#message-input").waitFor({ timeout: 3600000 });
    mark("loaded");
    const setup = await appData(page, "/admin/onboarding");
    marks.gpu = setup.gpu;
    marks.backend = setup.gpu?.backend;
    if (!["vulkan", "sycl"].includes(marks.backend)) throw new Error("Unknown active backend");
    await sleep(2500);
  });
}

async function thinking() {
  await shot("thinking", async (page, mark) => {
    await page.goto(ORIGIN + "/chat");
    await page.waitForFunction(() => document.querySelector("#model-select")?.value);
    await page.locator("#settings-toggle").click();
    await page.locator("#s-rbudget").waitFor();
    mark("ready");
    marks_box("budget_box", await page.locator("#s-rbudget").boundingBox());
    await page.fill("#s-rbudget", "");
    await page.locator("#s-rbudget").pressSequentially("256", { delay: 160 });
    await sleep(500);
    await page.click("#s-apply");
    await page.locator("#s-feedback").filter({ hasText: /^Saved/ }).waitFor({ timeout: 60000 });
    mark("saved");
    await sleep(3200);
  });
}

async function montage() {
  await shot("montage", async (page, mark) => {
    await page.goto(ORIGIN + "/#discover");
    await page.locator(".discover-cover").first().waitFor({ timeout: 60000 });
    mark("discover"); await sleep(1200);
    await page.goto(ORIGIN + "/work");
    await page.locator('#work-harness option[value="hermes"]').waitFor({ state: "attached", timeout: 60000 });
    await page.selectOption("#work-harness", "hermes");
    mark("hermes"); await sleep(1200);
    await page.goto(ORIGIN + "/chat");
    await page.click("#settings-toggle");
    await page.locator("#s-rbudget").waitFor();
    mark("settings"); await sleep(1200);
    await page.goto(ORIGIN + "/");
    await page.locator(".model-card").first().waitFor({ timeout: 60000 });
    mark("models"); await sleep(1200);
  });
}

// trailer-rest reruns only the 27B take and montage on top of an earlier trailer take.
const REST = process.argv[2] === "trailer-rest";
const which = REST ? "trailer" : process.argv[2] || "all";
if (which === "trailer") {
  // A failed recording must not leave old evidence available for a new edit.
  if (!REST) rmSync(OUT, { recursive: true, force: true });
  try {
    if (!REST) {
      const url = prepFresh();
      await terminal(url);
      await firstRun(url);
      await chat("speed");
      await thinking();
    }
    prepModel();
    if (!REST) await work();
    if (process.env.TRAILER_27B_MODEL) await chat("bigger", process.env.TRAILER_27B_MODEL);
    await montage();
  } finally { serve("stop"); }
}
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
