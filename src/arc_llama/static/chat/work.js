(function (root) {
  const ArcChat = root.ArcChat = root.ArcChat || {};

  // ---- Pure helpers: turn raw tool calls into plain language -------------------------------

  const REDIRECT = /^\s*(?:printf|echo|cat|tee)\b[^|;&]*?(?:>>?)\s*("[^"]+"|'[^']+'|[^\s;&|]+)/;
  const TEST_RUN = /\b(pytest|unittest|npm\s+(?:run\s+)?test|yarn\s+test|pnpm\s+test|cargo\s+test|go\s+test|make\s+test|vitest|jest)\b/;
  const INSTALL = /\b(pip3?\s+install|uv\s+(?:add|pip\s+install)|npm\s+(?:install|i)\b|yarn\s+add|pnpm\s+(?:add|install)|cargo\s+add)\b/;
  const RISKY = /\brm\s+-[a-z]*r|\bgit\s+(?:reset\s+--hard|clean\s+-[a-z]*f|push\s+(?:-f|--force))|\bsudo\b|\bmkfs\b|\bdd\s+if=|curl[^|]*\|\s*(?:ba)?sh|>\s*\/dev\/sd/;

  function baseName(path) {
    const parts = String(path || "").split(/[\\/]/).filter(Boolean);
    return parts.length ? parts[parts.length - 1] : String(path || "");
  }
  function unquote(text) {
    return String(text || "").replace(/^(["'])(.*)\1$/, "$2");
  }
  function shorten(text, limit) {
    const t = String(text || "");
    return t.length > limit ? t.slice(0, limit - 1) + "…" : t;
  }
  function firstString(...values) {
    for (const v of values) if (typeof v === "string" && v) return v;
    return "";
  }

  // Returns {kind, pending, done, detail, path, command, risky}. `detail` is what "Show command" reveals.
  function humanizeStep(name, args) {
    const tool = String(name || "").toLowerCase();
    const a = args && typeof args === "object" ? args : {};
    const path = firstString(a.filePath, a.file_path, a.path, Array.isArray(a.paths) ? a.paths[0] : "");
    const file = baseName(path);
    const out = (kind, pending, done, extra = {}) => ({ kind, pending, done, detail: "", path: "", command: "", risky: false, ...extra });

    if (["read", "read_file", "read_multiple_files"].includes(tool)) {
      return out("read", `Reading ${file || "a file"}`, `Read ${file || "a file"}`, { path, detail: path });
    }
    if (["write", "write_file"].includes(tool)) {
      return out("write", `Writing ${file || "a file"}`, `Wrote ${file || "a file"}`, { path, detail: path });
    }
    if (["edit", "apply_patch", "patch", "multiedit"].includes(tool)) {
      return out("edit", `Editing ${file || "a file"}`, `Edited ${file || "a file"}`, { path, detail: path });
    }
    if (["list", "list_directory", "ls"].includes(tool)) {
      return out("list", "Looking through the folder", "Looked through the folder", { detail: path });
    }
    if (["glob", "find"].includes(tool)) {
      return out("search", "Finding files", "Found files", { detail: firstString(a.pattern, a.glob) });
    }
    if (["grep", "search_files", "semantic_search", "repo_map", "codesearch"].includes(tool)) {
      return out("search", "Searching the project", "Searched the project", { detail: firstString(a.pattern, a.query) });
    }
    if (["webfetch", "websearch", "web_fetch", "web_search"].includes(tool)) {
      return out("web", "Using the internet", "Used the internet", { detail: firstString(a.url, a.query) });
    }
    if (tool === "task") return out("think", "Handing off a subtask", "Handed off a subtask", { detail: firstString(a.description, a.prompt) });
    if (["todowrite", "todoread"].includes(tool)) return out("think", "Updating the plan", "Updated the plan");

    if (["bash", "run_command", "shell", "sh", "terminal"].includes(tool)) {
      const command = firstString(a.command, a.cmd);
      const base = { command, detail: command, risky: RISKY.test(command) };
      const write = REDIRECT.exec(command);
      if (write) {
        const target = unquote(write[1]);
        return out("write", `Writing ${baseName(target)}`, `Wrote ${baseName(target)}`, { ...base, path: target });
      }
      const words = command.trim().split(/\s+/);
      const head = words[0] || "";
      if (head === "cat" && words.length === 2 && !/[|;&<>]/.test(command)) {
        return out("read", `Reading ${baseName(unquote(words[1]))}`, `Read ${baseName(unquote(words[1]))}`, { ...base, path: unquote(words[1]) });
      }
      if (["ls", "tree", "pwd", "find"].includes(head)) return out("list", "Looking through the folder", "Looked through the folder", base);
      if (head === "mkdir") return out("write", "Creating a folder", "Created a folder", base);
      if (["rm", "rmdir"].includes(head)) return out("delete", "Removing files", "Removed files", base);
      if (head === "mv") return out("write", "Moving files", "Moved files", base);
      if (head === "cp") return out("write", "Copying files", "Copied files", base);
      if (head === "touch" && words[1]) return out("write", `Creating ${baseName(unquote(words[1]))}`, `Created ${baseName(unquote(words[1]))}`, { ...base, path: unquote(words[1]) });
      if (head === "git") {
        const sub = words[1] || "";
        if (["status", "diff", "log", "show", "branch"].includes(sub)) return out("git", "Checking the git history", "Checked the git history", base);
        return out("git", `Running git ${sub}`.trim(), `Ran git ${sub}`.trim(), base);
      }
      if (TEST_RUN.test(command)) return out("test", "Running tests", "Ran tests", base);
      if (INSTALL.test(command)) return out("run", "Installing packages", "Installed packages", base);
      return out("run", "Running a command", "Ran a command", base);
    }
    // Anything else is a tool inherited from the selected agent's setup (usually "<server>_<tool>").
    const nice = String(name || "a tool").replace(/^([^_]+)_/, "$1: ").replace(/_/g, " ");
    return out("tool", `Using ${nice}`, `Used ${nice}`, { detail: shorten(JSON.stringify(a, null, 2), 600), toolName: String(name || "") });
  }

  // The approval sheet speaks about the permission being asked for, in the same plain words.
  function describeApproval(permission, step, folder) {
    const perm = String(permission || "").toLowerCase();
    const kind = step ? step.kind : "";
    const file = step && step.path ? baseName(step.path) : "";
    const outside = perm === "external_directory" || (step && step.path && isOutside(step.path, folder));
    let title = "Allow Arc Llama to continue?";
    let body = "It needs your permission before it goes on.";
    if (outside) {
      const where = step && step.path ? step.path : "somewhere outside it";
      title = "Allow Arc Llama to work outside your project folder?";
      body = `It wants to ${kind === "read" ? "read" : kind === "edit" ? "edit" : "write"} ${where}` + (folder ? `, which is not inside ${folder}.` : ".");
      return { title, body, file, showCommand: true, outside: true, detail: step && step.path ? step.path : "" };
    }
    if (kind === "delete" || (step && step.risky)) {
      title = kind === "delete" ? "Allow Arc Llama to delete files?" : "Allow Arc Llama to run this command?";
      body = "This can change or remove things in your project. Check the command before allowing it.";
    } else if (["write", "edit"].includes(kind) || perm === "edit") {
      title = kind === "write" && file ? "Allow Arc Llama to write a file?" : "Allow Arc Llama to change a file?";
      body = file ? "It will " + (kind === "edit" ? "edit " : "write to ") + file + " in your project folder." : "It will change files in your project folder.";
    } else if (kind === "tool") {
      const nice = step.pending.replace(/^Using /, "");
      title = `Allow Arc Llama to use ${nice}?`;
      body = "This tool comes from the selected agent and can reach beyond your project folder.";
      return { title, body, file, showCommand: true, outside: false, detail: "" };
    } else if (kind === "web") {
      title = "Allow Arc Llama to use the internet?";
      body = "It wants to reach out to a website or search the web.";
    } else if (kind === "list") {
      title = "Allow Arc Llama to look around?";
      body = "It wants to list the files in your project folder.";
    } else if (kind === "read") {
      title = "Allow Arc Llama to read a file?";
      body = file ? "It wants to read " + file + "." : "It wants to read a file in your project folder.";
    } else if (kind === "test") {
      title = "Allow Arc Llama to run your tests?";
      body = "Running tests executes your project's code.";
    } else if (kind === "git") {
      title = "Allow Arc Llama to check git?";
      body = "It wants to look at your repository's history or status.";
    } else if (perm === "bash" || kind === "run") {
      title = "Allow Arc Llama to run a command?";
      body = "It wants to run a command in your project folder.";
    }
    return { title, body, file, showCommand: !step || ["run", "other", "delete", "test", "web", "tool"].includes(kind) || !!step.risky, outside: false, detail: "" };
  }

  // What people type for "my home folder" is not always a path. Make the common forms work.
  function normalizeFolder(raw, home) {
    const text = String(raw || "").trim();
    if (!text) return "";
    const h = String(home || "").replace(/\/+$/, "");
    const lower = text.toLowerCase().replace(/\/+$/, "");
    if (["home", "~", "$home", "${home}", "home folder", "home directory", "my home", "my home folder"].includes(lower)) return h || text;
    if (text.startsWith("~/") && h) return h + text.slice(1);
    if (!/^([/\\]|[A-Za-z]:[/\\])/.test(text) && !text.startsWith("~") && h) return h + "/" + text.replace(/^\.\//, "");
    return text;
  }

  // Is `path` somewhere other than inside `folder`? Used so a prompt never says "in your project folder" when it is not.
  function isOutside(path, folder) {
    const p = String(path || "");
    const f = String(folder || "").replace(/\/+$/, "");
    if (!p) return false;
    if (/^\.\.([/\\]|$)/.test(p) || p.includes("/../")) return true;
    const absolute = /^([/\\]|[A-Za-z]:[/\\])/.test(p);
    if (!absolute) return false;
    if (!f) return false;
    return !(p === f || p.startsWith(f + "/"));
  }

  // A short title from the first sentence; whatever is left becomes the subtitle.
  function splitTask(task) {
    const text = String(task || "").trim();
    const first = text.split("\n")[0];
    const cut = /^(.{12,90}?[.!?])(\s+|$)/.exec(first);
    let title = cut ? cut[1] : shorten(first, 90);
    let rest = text.slice(cut ? cut[0].length : first.length).trim();
    if (!cut && first.length > 90) rest = text;           // the title was cut short, so keep the whole task visible
    title = title.replace(/[.!?]$/, "");
    return { title, rest: rest && rest !== title ? shorten(rest, 400) : "" };
  }

  function elapsedText(ms) {
    const s = Math.max(0, Math.round(ms / 1000));
    return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60} s`;
  }

  // ---- Component ---------------------------------------------------------------------------

  ArcChat.createWork = function (ctx) {
    const ROOT_KEY = "arc-llama-work-root";
    const HARNESS_KEY = "arc-llama-work-harness";
    const MODEL_KEY = "arc-llama-work-model";
    let info = null;
    let runs = [];             // the conversation: every task sent from this view, oldest first
    let run = null;            // the current (latest) turn
    let controller = null;     // AbortController for the active stream
    let tick = null;
    let loadedIds = new Set(); // models the server currently has in memory
    let loadedKnown = false;
    let loadedAt = 0;
    let modelPicked = false;   // the user chose a model in this view; never override it
    const AUTO_KEY = "arc-llama-work-auto";
    let autoOn = false;        // auto-approve; scoped to this browser tab on purpose
    try { autoOn = window.sessionStorage.getItem(AUTO_KEY) === "1"; } catch (_) { autoOn = false; }

    const $ = (sel) => document.querySelector(sel);
    function el(tag, props, ...kids) {
      const n = document.createElement(tag);
      for (const [k, v] of Object.entries(props || {})) {
        if (v == null || v === false) continue;
        if (k === "class") n.className = v;
        else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
        else n.setAttribute(k, v === true ? "" : v);
      }
      for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
      return n;
    }
    function store(key, value) { try { if (value == null) ctx.storage.removeItem(key); else ctx.storage.setItem(key, value); } catch (_) {} }
    function recall(key) { try { return ctx.storage.getItem(key); } catch (_) { return null; } }

    // ---- mode switch
    function setMode(mode) {
      const work = mode === "work" && info && info.experimental;
      document.body.classList.toggle("work-mode", !!work);
      $("#mode-chat").setAttribute("aria-selected", String(!work));
      $("#mode-work").setAttribute("aria-selected", String(!!work));
      if (work) { syncModels(); refreshWorkContext(); $("#work-task").focus(); }
      // The top navigation follows the mode, so Work reads as its own place.
      for (const link of document.querySelectorAll(".app-nav .nav-link")) {
        const target = link.getAttribute("href") === "/work" ? work : link.getAttribute("href") === "/chat" && !work;
        link.classList.toggle("active", !!target);
        if (target) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
      }
      window.ArcPaper?.mode();
      window.dispatchEvent(new Event("arc-paper-work-mode"));
      $("#page-tagline").textContent = work ? "Work" : "Chat";
      document.title = work ? "Work: arc-llama" : "Chat: arc-llama";
      try { history.replaceState(null, "", work ? "/work" : "/chat"); } catch (_) {}
    }

    // "code" unless the user picked Documents; Documents only works with vision models and the built-in agent.
    const workType = () => { const t = $("#work-type"); return t && !$("#work-type-label").hidden ? t.value : "code"; };

    function syncModels() {
      const source = $("#model-select");
      const target = $("#work-model");
      if (!source || !target) return;
      const keep = target.value || recall(MODEL_KEY) || source.value;
      const documents = workType() === "documents";
      const visionIds = new Set((info && info.vision_models) || []);
      const usable = Array.from(source.options).filter((o) => o.value && !o.disabled && (!documents || visionIds.has(o.value)));
      target.replaceChildren(...usable.map((o) => el("option", { value: o.value }, o.textContent)));
      target.value = usable.some((o) => o.value === keep) ? keep : (usable[0] ? usable[0].value : "");
      preferLoadedModel();
      refreshWorkContext();
    }

    // Starting on a model that is not in memory makes the server swap models, which is slow
    // and evicts whatever else is loaded. Default to a loaded model; say so when it isn't.
    async function preferLoadedModel() {
      if (Date.now() - loadedAt > 15000) {
        loadedAt = Date.now();
        try {
          const r = await ctx.request("/v1/models");
          if (r.ok) {
            const data = await r.json();
            loadedIds = new Set((data.data || []).filter((m) => m.metadata && m.metadata.loaded).map((m) => m.id));
            loadedKnown = true;
          }
        } catch (_) { /* keep the previous answer */ }
      }
      const target = $("#work-model");
      if (!modelPicked && !recall(MODEL_KEY) && loadedKnown && loadedIds.size && !loadedIds.has(target.value)) {
        const loaded = Array.from(target.options).find((o) => loadedIds.has(o.value));
        if (loaded) target.value = loaded.value;
      }
      modelNote();
    }

    function modelNote() {
      const hint = $("#work-hint");
      if (!hint || hint.classList.contains("error")) return;
      const value = $("#work-model").value;
      const parts = [];
      if (value && loadedKnown && !loadedIds.has(value)) parts.push(`${value} is not loaded yet, so starting will load it first. This can take a while.`);
      const selectedHarness = info?.harnesses?.find(h => h.id === $("#work-harness").value);
      if (selectedHarness?.approval_note) parts.push(selectedHarness.approval_note);
      if (autoOn) parts.push("Auto-approve is on: edits and routine commands run without asking. Risky commands still ask, and you can Undo.");
      if (folderIsWholeHome()) parts.push("This is your whole home folder, so Undo may not be available. A project folder works best.");
      hint.textContent = parts.length ? parts.join(" ") : "Enter to start, Shift+Enter for a new line";
    }

    function harnessAllowsAuto() {
      const selected = $("#work-harness").value;
      return selected !== "hermes" && info?.harnesses?.find(h => h.id === selected)?.auto_approve !== false;
    }

    function updateHarnessControls() {
      if (!harnessAllowsAuto() && autoOn) setAuto(false);
      renderAutoChip(); renderProfiles(); modelNote();
    }

    function renderAutoChip() {
      const chip = $("#work-auto");
      if (!chip) return;
      chip.hidden = !info?.auto_approve;
      chip.disabled = !harnessAllowsAuto();
      if (chip.disabled) {
        chip.setAttribute("aria-pressed", "false");
        chip.textContent = "Hermes approvals";
        chip.title = "Hermes asks before file edits and flagged commands. Auto-approve is unavailable.";
        return;
      }
      chip.setAttribute("aria-pressed", String(autoOn));
      chip.textContent = autoOn ? "Auto-approve" : "Ask before changes";
      chip.title = autoOn
        ? "Edits and routine commands run without asking. Destructive commands (rm -r, sudo, git reset --hard, force push...) still ask. Click to go back to asking."
        : "Click to let Arc Llama run edits and routine commands without asking. Destructive commands will still ask.";
    }

    function setAuto(on) {
      autoOn = !!on && harnessAllowsAuto();
      try { if (autoOn) window.sessionStorage.setItem(AUTO_KEY, "1"); else window.sessionStorage.removeItem(AUTO_KEY); } catch (_) {}
      renderAutoChip();
      notice("");
    }

    const measuredSpeed = (value) => Number.isFinite(value) && value > 0 ? `${value.toFixed(1)} tok/s` : "";

    // ---- rendering
    function stepRow(step) {
      const human = step.human;
      const state = step.state;
      const mark = el("span", { class: "work-mark " + state, "aria-hidden": "true" }, state === "done" ? "✓" : state === "failed" ? "!" : state === "waiting" ? "!" : "");
      const title = state === "done" || state === "failed" ? human.done : human.pending;
      const body = el("div", { class: "work-step-body" }, el("div", { class: "work-step-title" }, title));
      if (state === "waiting") body.append(el("div", { class: "work-step-sub" }, "Needs your OK before it changes anything"));
      if (state === "failed") body.append(el("div", { class: "work-step-sub error" }, shorten(step.output || "That step failed.", 160)));
      if (human.detail && human.kind !== "other" && state !== "waiting") {
        const open = !!step.open;
        const toggle = el("button", { class: "work-link", type: "button", "aria-expanded": String(open),
          onclick: () => { step.open = !step.open; render(); } }, open ? "Hide details" : (human.command ? "Show command" : "Show details"));
        body.append(toggle);
        if (open) {
          body.append(el("pre", { class: "work-code" }, human.detail));
          if (step.output) body.append(el("pre", { class: "work-code dim" }, shorten(step.output, 2000)));
        }
      }
      return el("div", { class: "work-step " + state }, mark, body, el("span", { class: "work-time" }, step.time || ""));
    }

    function approvalSheet(pending, step) {
      const d = describeApproval(pending.tool, step && step.human, run && run.folder);
      const answered = pending.answered;
      const code = (step && (step.human.command || (["tool", "web"].includes(step.human.kind) ? step.human.detail : "")))
        || (pending.arguments && pending.arguments.patterns && pending.arguments.patterns.join("\n")) || "";
      const sheet = el("div", { class: "work-sheet", role: "alertdialog", "aria-label": d.title },
        el("h2", {}, d.title),
        el("p", {}, d.body),
        run && run.auto ? el("p", { class: "work-auto-note" }, "Auto-approve is on, but this can do real damage, so it still asks.") : null);
      if (d.outside && d.detail) sheet.append(el("pre", { class: "work-code" }, d.detail));
      if (d.file && !d.outside) sheet.append(el("div", { class: "work-file" }, el("span", { class: "work-file-icon", "aria-hidden": "true" }, "▤"), el("div", {}, el("b", {}, d.file), el("span", {}, "In your project folder"))));
      if (code && (d.showCommand || pending.showCommand)) sheet.append(el("pre", { class: "work-code" }, code));
      const decide = (approved) => {
        pending.answered = true;
        render();
        ctx.request(`/v1/agent/${encodeURIComponent(pending.run_id || run.id)}/confirm`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ approved }),
        }).catch(() => {});
      };
      sheet.append(el("div", { class: "work-actions" },
        el("button", { class: "work-btn", type: "button", disabled: answered, onclick: () => decide(false) }, "Don’t Allow"),
        el("button", { class: "work-btn primary", type: "button", disabled: answered, onclick: () => decide(true) }, "Allow"),
        code && !d.showCommand ? el("button", { class: "work-link", type: "button", onclick: () => { pending.showCommand = !pending.showCommand; render(); } }, pending.showCommand ? "Hide command" : "Show command") : null));
      return sheet;
    }

    function resultCard(changes) {
      const files = changes.files || [];
      const card = el("section", { class: "work-result", "aria-label": "Files changed" });
      const words = { added: "Added", modified: "Edited", deleted: "Removed" };
      card.append(el("div", { class: "work-result-head" }, `${files.length} file${files.length === 1 ? "" : "s"} changed`,
        changes.truncated ? el("span", { class: "work-pill warn" }, "Large project: list may be incomplete") : null));
      for (const f of files) {
        const row = el("div", { class: "work-change" },
          el("span", { class: "work-pill " + f.status }, words[f.status] || f.status),
          el("span", { class: "work-change-path" }, f.path),
          el("span", { class: "work-change-stat" }, f.binary ? "binary" : `+${f.additions} -${f.deletions}`));
        card.append(row);
        if (f.preview && f.preview.length && f.status !== "deleted" && !(changes.reviews && changes.reviews[f.path])) {
          const asDiff = f.preview_kind === "diff";
          card.append(el("pre", { class: "work-preview" }, ...f.preview.map((line) => {
            const cls = asDiff ? (line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : "") : "add";
            return el("span", { class: "work-line " + cls }, (asDiff ? line : "+ " + line) + "\n");
          })));
        }
        if (changes.reviews && changes.reviews[f.path]) {
          card.append(el("pre", { class: "work-diff" }, ...changes.reviews[f.path].split("\n").filter((line) => !/^(---|\+\+\+) /.test(line)).map((line) =>
            el("span", { class: "work-line " + (line.startsWith("+") && !line.startsWith("+++") ? "add" : line.startsWith("-") && !line.startsWith("---") ? "del" : "") }, line + "\n"))));
        }
      }
      const undone = changes.undone;
      const actions = el("div", { class: "work-actions" });
      if (!undone) {
        actions.append(el("button", { class: "work-btn", type: "button", onclick: () => review(changes) }, changes.reviewing ? "Hide review" : "Review"));
        if (changes.undoable) actions.append(el("button", { class: "work-btn", type: "button", disabled: changes.undoing, onclick: () => undo(changes) }, changes.undoing ? "Undoing…" : "Undo"));
        else actions.append(el("span", { class: "work-note" }, changes.reopened ? "Undo is only offered right after a task runs." : "Undo is unavailable for this run (project too large or a file was too big to keep)."));
      } else {
        actions.append(el("span", { class: "work-note" }, undone.text));
      }
      card.append(actions);
      return card;
    }

    function render() {
      const scroller = document.querySelector(".work-scroll");
      const stick = !scroller || scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
      draw();
      refreshWorkContext();
      if (!scroller || !run) return;
      const sheet = document.querySelector(".work-sheet");
      if (sheet) sheet.scrollIntoView({ block: "nearest" });
      else if (stick) scroller.scrollTop = scroller.scrollHeight;
    }

    // Answers are markdown: headings, lists, tables and highlighted code with a copy button.
    function sayBlock(text, extra = "") {
      const block = el("div", { class: "work-say" + extra });
      if (ctx.renderMarkdown) ctx.renderMarkdown(block, text); else block.textContent = text;
      return block;
    }

    function draw() {
      const host = $("#work-run");
      const intro = $("#work-intro");
      if (!host) return;
      intro.hidden = runs.length > 0;
      host.hidden = runs.length === 0;
      host.replaceChildren();
      const busy = !!(run && run.status === "running");
      $("#work-send").classList.toggle("stop", busy);
      $("#work-send").setAttribute("aria-label", busy ? "Stop" : "Start");
      $("#work-send").title = busy ? "Stop" : "Start";
      const speed = $("#work-speed");
      if (speed) {
        const measured = measuredSpeed(run?.generationTokS);
        const label = measured || "N/A tok/s";
        if (speed.textContent !== label) speed.textContent = label;
        speed.title = measured
          ? `Latest measured model call${run?.model ? ` (${run.model})` : ""}; excludes loading, prompt processing, tools, and approvals. Updates when the backend reports generation timing.`
          : "Waiting for backend generation timing. No speed is estimated from response text or total task duration.";
      }
      const activity = $("#work-activity");
      if (activity) {
        const waiting = busy && run.pending && !run.pending.answered;
        const phases = { starting: "Starting agent", generating: "Generating", thinking: "Thinking", answering: "Writing answer", tools: "Running tools", waiting: "Waiting for approval" };
        const phase = waiting ? "waiting" : run?.activity || "starting";
        const modelState = ctx.getModelState?.(run?.model);
        const loading = ["starting", "generating"].includes(phase) && ["loading", "draining"].includes(modelState);
        const text = busy ? loading ? modelState === "loading" ? "Loading model" : "Waiting for model switch" : phases[phase] || "Working" : "";
        activity.hidden = !busy;
        activity.dataset.phase = phase;
        if ($("#work-activity-label").textContent !== text) $("#work-activity-label").textContent = text;
        $("#work-activity-time").textContent = busy ? elapsedText(Date.now() - run.startedAt) : "";
      }
      for (const turn of runs) host.append(drawTurn(turn, turn === run));
      window.ArcPaper?.workState({ pending: !!run?.pending && !run.pending.answered, result: host.querySelector(".work-turn:last-child .work-result"), recentTasks: $("#work-recent") });
    }

    // One message and everything that came of it. Earlier turns are compact; the latest is in full.
    function drawTurn(r, current) {
      const turn = el("section", { class: "work-turn" + (current ? "" : " prev"), "aria-label": "Task" });
      const mode = (r.auto ? " · auto-approve" : "") + (r.profileName ? ` · ${r.profileName}` : "");
      const eyebrow = r.status === "running" ? `${r.folderName}${mode} · working for ${elapsedText(Date.now() - r.startedAt)}`
        : r.status === "stopped" ? `${r.folderName}${mode} · stopped`
        : r.status === "error" ? `${r.folderName}${mode} · stopped by an error`
        : r.imported ? `${r.folderName} · imported conversation`
        : `${r.folderName}${mode} · finished in ${elapsedText((r.endedAt || Date.now()) - r.startedAt)}`;
      const heading = splitTask(r.task);
      const speed = measuredSpeed(r.generationTokS);
      turn.append(el("div", { class: "work-eyebrow" }, eyebrow + (speed ? ` · ${speed}` : "")), el("h1", { class: "work-title" }, heading.title));
      if (heading.rest) turn.append(el("p", { class: "work-sub" }, heading.rest));

      // Steps: always shown while a turn is live or waiting on you; folded away once it is history.
      const folded = !current && !r.showSteps && r.status !== "running" && !(r.pending && !r.pending.answered);
      if (folded && r.steps.length) {
        turn.append(el("button", { class: "work-link", type: "button", "aria-expanded": "false",
          onclick: () => { r.showSteps = true; render(); } }, `Show ${r.steps.length} step${r.steps.length === 1 ? "" : "s"}`));
      } else {
        const steps = el("div", { class: "work-steps", "aria-live": current ? "polite" : "off" });
        for (const step of r.steps) {
          steps.append(stepRow(step));
          if (r.pending && !r.pending.answered && r.pending.stepId === step.id) steps.append(approvalSheet(r.pending, step));
        }
        if (r.pending && !r.pending.answered && !r.steps.some((x) => x.id === r.pending.stepId)) steps.append(approvalSheet(r.pending, null));
        if (r.status === "running" && !r.steps.length && !r.pending && !r.answer.length) {
          const waited = (Date.now() - r.startedAt) / 1000;
          const livePhases = { thinking: "Thinking", answering: "Writing answer", generating: "Generating" };
          const phase = livePhases[r.activity] || (/starting/i.test(r.statusText || "") ? (r.harness === "hermes" ? "Starting Hermes" : r.harness === "opencode" ? "Starting OpenCode" : "Starting agent")
            : /working/i.test(r.statusText || "") ? (r.modelCold ? "Loading model / preparing prompt" : "Generating")
            : "Getting ready");
          const sub = r.modelCold && !["thinking", "answering"].includes(r.activity)
            ? `${r.model} was not in memory, so it is being loaded first. This usually takes a minute or two.`
            : waited > 8 ? "Still working. The first request also has to read the agent's instructions, which takes a moment."
            : "";
          steps.append(el("div", { class: "work-step running" }, el("span", { class: "work-mark running", "aria-hidden": "true" }),
            el("div", { class: "work-step-body" }, el("div", { class: "work-step-title" }, phase),
              sub ? el("div", { class: "work-step-sub" }, sub) : null)));
        }
        turn.append(steps);
        if (!current && r.showSteps && r.steps.length) {
          turn.append(el("button", { class: "work-link", type: "button", "aria-expanded": "true",
            onclick: () => { r.showSteps = false; render(); } }, "Hide steps"));
        }
      }
      const strays = r.steps.filter((x) => x.state === "done" && x.human.path && isOutside(x.human.path, r.folder) && ["write", "edit"].includes(x.human.kind));
      if (strays.length) {
        turn.append(el("div", { class: "work-error", role: "status" },
          "Written outside your project folder: " + strays.map((x) => x.human.path).join(", ") + ". Arc Llama can't undo changes there."));
      }
      if (r.changes && r.changes.files && r.changes.files.length) turn.append(resultCard(r.changes));
      if (r.answer.length) {
        turn.append(sayBlock(r.answer[r.answer.length - 1]));
        if (r.answer.length > 1) {
          turn.append(el("button", { class: "work-link", type: "button", "aria-expanded": String(!!r.showEarlier),
            onclick: () => { r.showEarlier = !r.showEarlier; render(); } },
            r.showEarlier ? "Hide earlier messages" : `Show earlier messages (${r.answer.length - 1})`));
          if (r.showEarlier) turn.append(sayBlock(r.answer.slice(0, -1).join("\n\n"), " dim"));
        }
      }
      if (r.error) {
        turn.append(el("div", { class: "work-error", role: "alert" }, r.error));
        if (r.suggest) {
          const fix = r.suggest, task = r.task;
          turn.append(el("div", { class: "work-actions" },
            el("button", { class: "work-btn primary", type: "button", onclick: () => {
              $("#work-root").value = fix;
              $("#work-task").value = task;
              autosize();
              runs = runs.filter((x) => x !== r);          // the failed attempt leaves the thread
              run = runs.length ? runs[runs.length - 1] : null;
              render();
              $("#work-task").focus();
            } }, "Use " + fix),
            el("span", { class: "work-note" }, "That is the closest folder that exists.")));
        }
      }
      if (r.status !== "running" && current) {
        const stat = [r.model, r.harness === "opencode" ? "OpenCode" : r.harness === "hermes" ? "Hermes" : "Built-in agent", "model on this computer"].join(" · ");
        turn.append(el("div", { class: "work-stat" }, stat));
      }
      return turn;
    }

    // What earlier turns contributed, in a form a model can use to understand a follow-up.
    // After /compact, earlier tasks are replaced by a summary; `upTo` is how many runs it covers.
    let compacted = null;
    function turnMessages(finished) {
      const out = [];
      for (const r of finished) {
        out.push({ role: "user", content: r.task });
        const parts = [];
        if (r.answer.length) parts.push(r.answer[r.answer.length - 1]);
        if (r.changes && r.changes.files && r.changes.files.length) {
          parts.push("Files changed: " + r.changes.files.map((f) => `${f.path} (${f.status})`).join(", ") + (r.changes.undone ? " (then undone by the user)" : ""));
        }
        if (r.error) parts.push("(This task failed: " + r.error + ")");
        if (r.status === "stopped") parts.push("(The user stopped this task before it finished.)");
        out.push({ role: "assistant", content: parts.join("\n") || "(no reply)" });
      }
      return out;
    }
    function summaryMessage() { return { role: "assistant", content: "Summary of the earlier work in this session (may omit details): " + compacted.summary }; }
    function historyForRequest() {
      // Called before the new turn is created, so every turn so far is "earlier"; skip any still running.
      const finished = runs.slice(compacted ? compacted.upTo : 0).filter((x) => x.status !== "running").slice(-6);
      return [...(compacted ? [summaryMessage()] : []), ...turnMessages(finished)];
    }
    // Context for the next follow-up is different from a running agent's full prompt.
    // Tokenizer counts below describe the carried conversation only; usage events include tools.
    let workBudget = null, budgetKey = "", budgetTimer = null, budgetAbort = null, budgetEpoch = 0;
    let compactingSession = false;
    const positive = n => Number.isFinite(n) && n > 0;
    const estimatedTokens = messages => Math.ceil(messages.reduce((n, m) => n + new TextEncoder().encode(m.content || "").length + 64, 0) / 4);
    function contextLimit() {
      const model = $("#work-model").value;
      const value = workBudget?.model === model ? workBudget.limit : ctx.getContextLimit?.(model);
      return positive(value) ? value : null;
    }
    function contextMessages() {
      const messages = historyForRequest();
      const draft = $("#work-task").value.trim();
      return [...messages, ...(draft ? [{ role: "user", content: draft }] : [])];
    }
    function budgetScope() {
      return JSON.stringify([$("#work-model").value, $("#work-harness").value, normalizeFolder($("#work-root").value, info && info.home), profileFor()]);
    }
    function workPayload() {
      return { task: $("#work-task").value.trim(), history: historyForRequest(), root: normalizeFolder($("#work-root").value, info && info.home), harness: $("#work-harness").value };
    }
    function drawWorkContext() {
      const label = $("#work-context-label"), track = $("#work-context-track"), note = $("#work-context-note");
      if (!label) return;
      const busy = run?.status === "running", measured = busy && run.context;
      const messages = contextMessages(), limit = positive(measured?.context_tokens) ? measured.context_tokens : contextLimit();
      const known = workBudget?.model === $("#work-model").value && workBudget.key === JSON.stringify(workPayload());
      const setup = [...runs].reverse().find(r => r.budgetScope === budgetScope() && Number.isFinite(r.setupTokens));
      const full = !!measured || (known && workBudget.scope === "full_prompt") || (known && !!setup);
      const used = measured ? measured.prompt_tokens + measured.completion_tokens : (known ? workBudget.tokens : estimatedTokens(messages)) + (known && workBudget.scope !== "full_prompt" && setup ? setup.setupTokens : 0);
      const percent = limit && full ? Math.round(used / limit * 100) : null;
      const heading = measured ? measured.source === "request" ? "Current prompt" : "Last model call" : full ? "Next prompt estimate" : "Request text · setup not measured";
      label.textContent = `${heading}: ${measured?.exact ? "" : "~"}${used.toLocaleString()}${limit ? ` / ${limit.toLocaleString()} tokens${percent != null ? ` (${percent}%)` : ""}` : " tokens · limit unavailable"}`;
      const fresh = runs.slice(compacted?.upTo || 0).filter(r => r.status !== "running");
      const carried = historyForRequest();
      const clipped = fresh.length > 6 || carried.some(m => m.content.length > 3000) || carried.reduce((n,m) => n + Math.min(3000,m.content.length) + 12, 0) > 10000;
      const caution = percent != null && percent >= 75;
      $("#work-context").classList.toggle("near-limit", caution);
      track.hidden = percent == null;
      if (percent != null) { track.setAttribute("aria-valuenow", String(Math.min(100, percent))); track.setAttribute("aria-valuetext", label.textContent); $("#work-context-fill").style.width = `${Math.min(100, percent)}%`; }
      note.textContent = measured
        ? `Includes agent instructions and tool definitions.${measured.source === "request" ? " Counted from the actual model request." : " Usage reported by the last model call; newer tool output may add more."}${caution ? " Finish or stop the task, then compact before continuing." : ""}`
        : `${caution ? "Context is getting full. Compact before continuing. " : ""}${clipped ? "Older or long tasks are shortened for follow-ups. Compact to carry their details forward. " : ""}${full ? workBudget?.scope === "full_prompt" ? "Includes the built-in agent’s instructions and tools. Reply needs additional space." : "Includes setup measured on the previous task. Tools and project instructions may change; reply needs additional space." : "This is a partial count. The agent’s instructions and tools are measured when its first model request starts."}${compacted ? " Earlier tasks are summarized; the full session stays saved." : ""}`;
      const button = $("#work-compact");
      button.disabled = busy || compactingSession || !fresh.length || !limit;
      button.textContent = compactingSession ? "Compacting…" : "Compact session";
    }
    function refreshWorkContext() {
      if (!$("#work-context-label")) return;
      drawWorkContext();
      const model = $("#work-model").value, messages = contextMessages();
      const work = workPayload();
      const key = JSON.stringify([model, work, budgetScope()]);
      if (key === budgetKey) return;
      budgetKey = key;
      const mine = ++budgetEpoch;
      clearTimeout(budgetTimer); budgetAbort?.abort();
      if (!model || run?.status === "running") return;
      budgetTimer = setTimeout(async () => {
        const abort = new AbortController(); budgetAbort = abort;
        const deadline = setTimeout(() => abort.abort(), 4000);
        try {
          const response = await ctx.request("/v1/chat/budget", { method: "POST", headers: { "Content-Type": "application/json" }, signal: abort.signal, body: JSON.stringify({ model, messages, max_tokens: 1, work }) });
          if (!response.ok) return;
          const value = await response.json();
          if (mine !== budgetEpoch || abort.signal.aborted || !Number.isFinite(value.prompt_tokens) || value.prompt_tokens < 0) return;
          workBudget = { model, scope: value.scope || "request_only", key: JSON.stringify(work), tokens: value.prompt_tokens, limit: positive(value.context_tokens) ? value.context_tokens : ctx.getContextLimit?.(model) };
          drawWorkContext();
        } catch (_) { /* the visible byte estimate remains explicitly approximate */ }
        finally { clearTimeout(deadline); }
      }, 250);
    }

    // ---- tool profiles: which of the harness's MCP servers, skills and web tools a task may use
    const PROFILE_KEY = "arc-llama-work-profile";
    let profiles = [], profileId = recall(PROFILE_KEY) || "", caps = null, capsKey = "", editingProfile = null;
    let capsLoading = false, capsError = "", managerAbort = null, managerEpoch = 0;
    const currentProfile = () => profiles.find((p) => p.id === profileId) || null;
    const profileFor = () => ($("#work-harness").value === "opencode" ? currentProfile() : null);
    function renderProfiles() {
      const select = $("#work-profile");
      if (profileId && !currentProfile()) { profileId = ""; store(PROFILE_KEY, null); }
      select.replaceChildren(new Option("All tools", ""), ...profiles.map((p) => new Option(p.name, p.id)), new Option("Manage profiles…", "__manage"));
      select.value = profileId;
      $("#work-profile-chip").hidden = $("#work-harness").value !== "opencode" || !info;
    }
    async function loadProfiles() {
      try {
        const r = await ctx.request("/v1/agent/profiles");
        if (r.ok) profiles = (await r.json()).profiles || [];
      } catch (_) { /* profiles are optional */ }
      renderProfiles();
    }
    function chooseProfile(id) {
      profileId = id; store(PROFILE_KEY, id || null); commandsKey = ""; updateHarnessControls(); refreshWorkContext();
    }
    async function refreshManagerCaps() {
      managerAbort?.abort();
      const mine = ++managerEpoch, abort = new AbortController();
      managerAbort = abort;
      const folder = normalizeFolder($("#work-root").value, info && info.home);
      capsError = "";
      capsLoading = !(caps && capsKey === folder);
      drawManager();
      if (!capsLoading) return;
      let timedOut = false;
      const deadline = setTimeout(() => { timedOut = true; abort.abort(); }, 120000);
      try {
        const r = await ctx.request(`/v1/agent/capabilities?root=${encodeURIComponent(folder)}`, { signal: abort.signal });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
        if (data.tools_complete === false) throw new Error((data.warnings || []).join("; ") || "Tool discovery was incomplete.");
        if (mine !== managerEpoch || abort.signal.aborted) return;
        caps = data; capsKey = folder;
      } catch (e) {
        if (mine !== managerEpoch) return;
        capsError = timedOut ? "Tool discovery timed out." : e.message;
      } finally {
        clearTimeout(deadline);
        if (mine === managerEpoch) { capsLoading = false; drawManager(); }
      }
    }
    function tickRow(label, checked, hint) {
      const box = el("input", { type: "checkbox" });
      box.checked = checked;
      return { box, row: el("label", { class: "tick" }, box, el("span", {}, label, hint ? el("small", {}, hint) : null)) };
    }
    function openManager(startNew) {
      $("#work-manager").hidden = false;
      editingProfile = startNew ? newProfile() : (currentProfile() ? { ...currentProfile() } : null);
      refreshManagerCaps();
    }
    function newProfile() { return { id: "p" + Date.now().toString(36) + Math.random().toString(36).slice(2, 8), name: "", mcp: [], skills: [], web: false, isNew: true }; }
    function drawManager() {
      const body = $("#work-manager-body");
      const list = el("div", { class: "work-profile-list" }, ...profiles.map((p) => {
        const row = el("button", { type: "button", class: "work-profile-row", "aria-current": String(!!editingProfile && editingProfile.id === p.id) }, el("b", {}, p.name), el("span", {}, `${p.mcp.length} server${p.mcp.length === 1 ? "" : "s"} · ${p.skills.length} skill${p.skills.length === 1 ? "" : "s"}${p.web ? " · web" : ""}`));
        row.addEventListener("click", () => { editingProfile = { ...p }; drawManager(); });
        return row;
      }), el("button", { type: "button", class: "work-link", onclick: () => { editingProfile = newProfile(); drawManager(); } }, "New profile"));
      body.replaceChildren(list);
      if (editingProfile && !editingProfile.isNew) {
        body.append(el("button", { type: "button", class: "work-btn", onclick: () => { chooseProfile(editingProfile.id); closeManager(); } }, "Use selected profile"));
      }
      if (capsLoading) { body.append(el("p", { class: "work-note work-discovery", role: "status" }, "Finding tools for this folder… Saved profiles are ready to use.")); return; }
      if (capsError) {
        body.append(el("div", { class: "work-discovery" },
          el("p", { class: "work-manager-error", role: "alert" }, "Tools unavailable. Your saved profiles are unchanged."),
          el("button", { type: "button", class: "work-btn", onclick: refreshManagerCaps }, "Retry"),
          el("details", {}, el("summary", {}, "Details"), el("p", { class: "work-note" }, capsError))));
        return;
      }
      if (!editingProfile) return;
      const draft = editingProfile, epoch = managerEpoch;
      const name = el("input", { type: "text", maxlength: "60", placeholder: "e.g. Research, Writing, Code review", "aria-label": "Profile name" });
      name.value = editingProfile.name;
      const off = new Set(editingProfile.tools_off || []);
      const allTools = caps.tools || [];
      const EDIT = ["edit", "write", "apply_patch", "multiedit"];
      const counter = el("p", { class: "work-note work-count", "aria-live": "polite" });
      // One box per tool. A server's tools are only offered while its own box is ticked.
      const toolBoxes = [];
      const tools = (ids, group) => ids.map((t) => {
        const row = tickRow(t.label || t.id, !t.ids.some((id) => off.has(id)));
        toolBoxes.push({ ...t, group, box: row.box });
        row.box.addEventListener("change", recount);
        return row.row;
      });
      const toolEntries = (group) => allTools.filter((t) => t.group === group).map((t) => ({ id: t.id, ids: [t.id], label: t.id.replace(new RegExp("^" + group + "_"), "") }));
      const servers = (caps.mcp || []).map((n) => {
        const row = tickRow(n, editingProfile.mcp.includes(n));
        const list = toolEntries(n);
        const inner = el("div", { class: "work-tool-list" }, ...tools(list, n));
        const details = el("details", { class: "work-tools" }, el("summary", {}, `${list.length} tool${list.length === 1 ? "" : "s"}`), inner);
        row.box.addEventListener("change", () => { details.hidden = !row.box.checked; recount(); });
        details.hidden = !row.box.checked;
        return { n, box: row.box, row: el("div", {}, row.row, details) };
      });
      const editPresent = allTools.filter((t) => EDIT.includes(t.id)).map((t) => t.id);
      const builtinEntries = allTools.filter((t) => t.group === "builtin" && !EDIT.includes(t.id) && !/^(list_mcp|read_mcp)/.test(t.id)).map((t) => ({ id: t.id, ids: [t.id], label: t.id }));
      if (editPresent.length) builtinEntries.unshift({ id: "edit", ids: editPresent, label: "edit and write files" });
      const builtinRows = tools(builtinEntries, "builtin");
      const pluginRows = tools(allTools.filter((t) => t.group === "plugin").map((t) => ({ id: t.id, ids: [t.id], label: t.id })), "plugin");
      function recount() {
        const serverOn = new Map(servers.map((x) => [x.n, x.box.checked]));
        const total = toolBoxes.length ? allTools.filter((t) => !["invalid", "question", "skill"].includes(t.id)).length : 0;
        let allowed = 0;
        for (const t of toolBoxes) if (t.box.checked && (t.group === "builtin" || t.group === "plugin" || serverOn.get(t.group))) allowed += t.ids.length;
        counter.textContent = total ? `${allowed} of ${total} tools enabled` : "";
      }
      const skills = (caps.skills || []).map((x) => ({ n: x.name, ...tickRow(x.name, editingProfile.skills.includes(x.name), (x.description || "").slice(0, 110)) }));
      const web = tickRow("Web search and fetch", !!editingProfile.web, "Lets the agent look things up on the internet.");
      const error = el("p", { class: "work-manager-error", role: "alert", hidden: true });
      const save = el("button", { type: "button", class: "work-btn primary" }, "Save and use");
      save.addEventListener("click", async () => {
        if (!name.value.trim()) { name.focus(); error.textContent = "Enter a profile name."; error.hidden = false; return; }
        const knownTools = new Set(toolBoxes.flatMap((t) => t.ids));
        const payload = {
          name: name.value.trim(), description: draft.description || "",
          mcp: [...draft.mcp.filter((n) => !servers.some((x) => x.n === n)), ...servers.filter((x) => x.box.checked).map((x) => x.n)],
          skills: [...draft.skills.filter((n) => !skills.some((x) => x.n === n)), ...skills.filter((x) => x.box.checked).map((x) => x.n)],
          web: web.box.checked,
          tools_off: [...off].filter((id) => !knownTools.has(id)).concat(toolBoxes.filter((t) => !t.box.checked).flatMap((t) => t.ids)),
        };
        save.disabled = true; error.hidden = true;
        try {
          const r = await ctx.request(`/v1/agent/profiles/${encodeURIComponent(draft.id)}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
          const data = await r.json().catch(() => ({}));
          if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
          profiles = profiles.filter((p) => p.id !== data.id).concat(data).sort((a, b) => a.name.localeCompare(b.name));
          renderProfiles();
          if (epoch === managerEpoch && editingProfile === draft) { chooseProfile(data.id); closeManager(); }
        } catch (e) { error.textContent = e.message; error.hidden = false; }
        finally { save.disabled = false; }
      });
      const actions = el("div", { class: "work-actions" }, save);
      if (!editingProfile.isNew) {
        const remove = el("button", { type: "button", class: "work-link" }, "Delete");
        remove.addEventListener("click", async () => {
          remove.disabled = true; save.disabled = true; error.hidden = true;
          try {
            const r = await ctx.request(`/v1/agent/profiles/${encodeURIComponent(draft.id)}`, { method: "DELETE" });
            if (!r.ok) { const data = await r.json().catch(() => ({})); throw new Error(data.detail || `HTTP ${r.status}`); }
            profiles = profiles.filter((p) => p.id !== draft.id);
            if (profileId === draft.id) chooseProfile("");
            renderProfiles();
            if (epoch === managerEpoch && editingProfile === draft) { editingProfile = null; drawManager(); }
          } catch (e) { error.textContent = e.message; error.hidden = false; }
          finally { remove.disabled = false; save.disabled = false; }
        });
        actions.append(remove);
      }
      body.append(
        el("fieldset", {}, el("legend", {}, "Name"), name),
        el("fieldset", {}, el("legend", {}, "MCP servers"), ...(servers.length ? servers.map((x) => x.row) : [el("p", { class: "work-note" }, "Your agent has no MCP servers set up.")])),
        el("fieldset", {}, el("legend", {}, "Skills"), ...(skills.length ? skills.map((x) => x.row) : [el("p", { class: "work-note" }, "No skills found.")])),
        ...(builtinRows.length ? [el("fieldset", {}, el("legend", {}, "Built-in tools"), ...builtinRows)] : []),
        ...(pluginRows.length ? [el("fieldset", {}, el("legend", {}, "Plugin tools"), ...pluginRows)] : []),
        el("fieldset", {}, el("legend", {}, "Web"), web.row),
        counter, error, actions);
      recount();
    }
    function closeManager() { managerEpoch++; managerAbort?.abort(); $("#work-manager").hidden = true; editingProfile = null; }

    // ---- saved sessions: every conversation is kept on the server so it can be reopened later
    let sessionId = null, sessionOrigin = "", saving = Promise.resolve();
    const newSessionId = () => "s" + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
    function sessionPayload() {
      const plain = JSON.parse(JSON.stringify(runs, (key, value) => (key === "pending" || key === "undoing" ? undefined : value)));
      for (const r of plain) if (r.status === "running") { r.status = "stopped"; r.error = r.error || "This task was interrupted when the page closed."; }
      const first = runs[0] ? splitTask(runs[0].task).title : "Untitled";
      const last = runs[runs.length - 1] || {};
      return { title: first, folder: last.folder || "", model: last.model || "", harness: last.harness || "", origin: sessionOrigin, compacted, profileId, runs: plain };
    }
    function persist() {
      if (!sessionId || !runs.length) return;
      const id = sessionId, body = JSON.stringify(sessionPayload());
      saving = saving.then(() => ctx.request(`/v1/agent/sessions/${encodeURIComponent(id)}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body })).catch(() => {});
    }
    const ago = (seconds) => {
      const d = Math.max(0, Date.now() / 1000 - seconds);
      return d < 90 ? "just now" : d < 5400 ? `${Math.round(d / 60)} min ago` : d < 129600 ? `${Math.round(d / 3600)} h ago` : `${Math.round(d / 86400)} days ago`;
    };
    let recentEpoch = 0;
    function sessionGroups(sessions) {
      const unique = new Map();
      for (const item of sessions) {
        if (!item || typeof item.id !== "string") continue;
        const old = unique.get(item.id);
        // A saved Work copy is the canonical continuation of the native ID.
        if (!old || (old.native && !item.native) || (!!old.native === !!item.native && item.updated > old.updated)) unique.set(item.id, item);
      }
      const groups = new Map();
      for (const item of [...unique.values()].sort((a, b) => b.updated - a.updated)) {
        const key = JSON.stringify([item.title || "Untitled", item.folder || "", item.harness || ""]);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(item);
      }
      return [...groups.values()];
    }
    async function loadRecent() {
      const box = $("#work-recent");
      if (!box) return;
      const mine = ++recentEpoch;
      let sessions = [], warnings = [];
      try {
        const r = await ctx.request("/v1/agent/sessions");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        sessions = data.sessions || [];
        warnings = data.warnings || [];
      } catch (_) { warnings = ["Could not load sessions. Try opening Sessions again."]; }
      if (mine !== recentEpoch) return;
      const groups = sessionGroups(sessions);
      let warning = $("#work-sessions-warning");
      if (!warning) { warning = el("p", { id: "work-sessions-warning", class: "work-note", role: "status" }); box.prepend(warning); }
      warning.textContent = warnings.join(" ");
      warning.hidden = !warnings.length;
      window.ArcPaper?.workSessions?.(sessions.length);
      box.hidden = !sessions.length && !warnings.length;
      function sessionRow(item, expanded = false, includeRelated = true) {
        const open = el("button", { type: "button", class: "work-recent-item" },
          el("b", {}, item.title),
          el("span", {}, `${item.harness === "opencode" ? "OpenCode · " : item.harness === "hermes" ? "Hermes · " : ""}${item.native ? "Harness history · " : item.origin ? "Work copy · " : ""}${baseName(item.folder) || "project"} · ${item.tasks} task${item.tasks === 1 ? "" : "s"} · ${expanded ? new Date(item.updated * 1000).toLocaleString() : ago(item.updated)}`));
        open.addEventListener("click", () => openSession(item.id));
        const remove = el("button", { type: "button", class: "work-recent-remove", "aria-label": `Delete ${item.title}`, title: "Delete" }, "×");
        remove.addEventListener("click", async () => {
          try { await ctx.request(`/v1/agent/sessions/${encodeURIComponent(item.id)}`, { method: "DELETE" }); } catch (_) {}
          loadRecent();
        });
        const row = el("div", { class: "work-recent-row", "data-session-id": item.id }, open, item.native ? null : remove);
        const related = includeRelated ? item.harness_history || [] : [];
        if (!related.length) return row;
        return el("div", { class: "work-linked-session" }, row,
          el("details", { class: "work-session-duplicates work-session-linked" },
            el("summary", {}, `${related.length} original harness session${related.length === 1 ? "" : "s"}`),
            ...related.map((entry) => sessionRow(entry, true, false))));
      }
      $("#work-recent-list").replaceChildren(...groups.map((group) => {
        const row = sessionRow(group[0]);
        if (group.length === 1) return row;
        const older = el("details", { class: "work-session-duplicates" },
          el("summary", {}, `${group.length - 1} more session${group.length === 2 ? "" : "s"} with this title`),
          ...group.slice(1).map((item) => sessionRow(item, true)));
        return el("div", { class: "work-session-group" }, row, older);
      }));
      window.ArcPaper?.workSessions?.(sessions.length);
    }
    async function openSession(id) {
      if (compactingSession) { notice("Wait for compaction to finish first."); return; }
      if (run && run.status === "running") { notice("Stop the current task first."); return; }
      try {
        const r = await ctx.request(`/v1/agent/sessions/${encodeURIComponent(id)}`);
        if (!r.ok) {
          const failure = await r.json().catch(() => ({}));
          throw new Error(failure.detail || `HTTP ${r.status}`);
        }
        const record = await r.json();
        runs = (record.runs || []).map((x) => {
          // Undo needs the files kept in memory right after a run, so it is only offered then.
          if (x.changes) x.changes = { ...x.changes, undoable: false, reopened: true };
          return { ...x, pending: null, showSteps: false };
        });
        run = runs[runs.length - 1] || null;
        compacted = record.compacted || null;
        if (record.model && [...$("#work-model").options].some(o => o.value === record.model && !o.disabled)) { $("#work-model").value = record.model; modelPicked = true; store(MODEL_KEY, record.model); }
        if (record.profileId && profiles.some((p) => p.id === record.profileId)) chooseProfile(record.profileId);
        sessionId = record.id;
        sessionOrigin = record.origin || "";
        if (record.folder || record.native) $("#work-root").value = record.folder || "";
        if (record.harness && [...$("#work-harness").options].some((o) => o.value === record.harness && !o.disabled)) $("#work-harness").value = record.harness;
        else if (record.native) $("#work-harness").value = "builtin";
        commandsKey = "";
        updateHarnessControls();
        problem("");
        render();
        window.ArcPaper?.closeWorkSessions?.();
        window.dispatchEvent(new Event("arc-paper-work-mode"));
        if (record.native) notice(`Opened ${record.harness === "opencode" ? "OpenCode" : "Hermes"} text history. Follow-ups save a Work copy using your selected local model; the original session stays in its harness.${record.folder ? "" : " Choose a project folder to continue."}`);
        $("#work-task").focus();
      } catch (e) {
        problem(`Could not reopen that session: ${e.message}`);
      }
    }

    // ---- slash commands: Arc Llama's own, plus whatever the chosen harness offers for this folder
    const LOCAL_COMMANDS = [
      { name: "compact", description: "Shorten earlier context. Add a focus, e.g. /compact keep file names", source: "arc-llama", takes_arguments: true },
      { name: "sessions", description: "Browse Work, OpenCode, and Hermes conversation history", source: "arc-llama", takes_arguments: false },
    ];
    let harnessCommands = [], commandsKey = "", menuItems = [], menuIndex = 0;
    const allCommands = () => [...LOCAL_COMMANDS, ...harnessCommands.filter((c) => !LOCAL_COMMANDS.some((l) => l.name === c.name))];
    async function loadCommands() {
      const harness = $("#work-harness").value, folder = normalizeFolder($("#work-root").value, info && info.home);
      const key = harness + "|" + folder + "|" + profileId;
      if (key === commandsKey) return;
      commandsKey = key; harnessCommands = [];
      if (harness !== "opencode") { renderMenu(); return; }
      try {
        const r = await ctx.request(`/v1/agent/commands?harness=${encodeURIComponent(harness)}&root=${encodeURIComponent(folder)}${profileFor() ? "&profile_id=" + encodeURIComponent(profileFor().id) : ""}`);
        if (key !== commandsKey) return;
        if (r.ok) harnessCommands = (await r.json()).commands || [];
        else commandsKey = "";           // try again next time, e.g. once the folder is right
      } catch (_) { commandsKey = ""; }
      renderMenu();
    }
    const sourceRank = (c) => (c.source === "skill" ? 2 : c.source === "mcp" ? 1 : 0);   // real commands first, skills last
    const typedCommand = () => { const m = /^\/([\w.:-]*)$/.exec($("#work-task").value); return m ? m[1].toLowerCase() : null; };
    function renderMenu() {
      const box = $("#work-commands"), typed = typedCommand();
      if (typed === null) { box.hidden = true; menuItems = []; return; }
      menuItems = allCommands().filter((c) => c.name.toLowerCase().includes(typed)).sort((a, b) => (b.name.toLowerCase().startsWith(typed) - a.name.toLowerCase().startsWith(typed)) || (sourceRank(a) - sourceRank(b)) || a.name.localeCompare(b.name));
      menuIndex = Math.min(menuIndex, Math.max(0, menuItems.length - 1));
      box.replaceChildren(...menuItems.map((c, i) => {
        const item = el("button", { type: "button", class: "work-command", role: "option", "aria-selected": String(i === menuIndex) }, el("b", {}, "/" + c.name), el("span", {}, c.description), el("em", {}, c.source === "command" ? "command" : c.source));
        item.addEventListener("mousedown", (e) => { e.preventDefault(); pickCommand(c); });
        return item;
      }));
      box.hidden = !menuItems.length;
    }
    function pickCommand(c) {
      const task = $("#work-task");
      task.value = "/" + c.name + (c.takes_arguments ? " " : "");
      autosize(); renderMenu(); task.focus();
    }
    // Returns {name, args} when the text names a command we know, else null (it is then an ordinary task).
    function parseCommand(text) {
      const m = /^\/([\w.:-]+)(?:\s+([\s\S]*))?$/.exec(text);
      if (!m) return null;
      const known = allCommands().find((c) => c.name.toLowerCase() === m[1].toLowerCase());
      return known ? { name: known.name, args: (m[2] || "").trim(), local: LOCAL_COMMANDS.includes(known) } : null;
    }

    async function compactSession(focus) {
      if (compactingSession) return;
      if (run && run.status === "running") { notice("Wait for the current task to finish before compacting."); return; }
      const finished = runs.filter((x) => x.status !== "running");
      const fresh = finished.slice(compacted ? compacted.upTo : 0);
      if (!fresh.length) { notice("Nothing to compact yet. Finish a task first."); return; }
      const model = $("#work-model").value;
      if (!model) { problem("Choose a model first. None is available yet."); return; }
      const limit = contextLimit();
      if (!limit) { problem("The model’s context size is unavailable. Choose a model with a known context size before compacting."); return; }
      problem("");
      compactingSession = true;
      drawWorkContext();
      const messages = [...(compacted ? [summaryMessage()] : []), ...turnMessages(fresh)];
      const button = $("#work-send");
      button.disabled = true;
      notice("Summarizing earlier tasks. A model that is not loaded yet may take a while to start.");
      try {
        const summary = await ArcChat.compaction.summarize({
          messages, ctx: limit, instruction: focus, signal: new AbortController().signal,
          request: async (body, signal) => {
            const response = await ctx.request("/v1/chat/completions", {
              method: "POST", headers: { "Content-Type": "application/json" }, signal,
              body: JSON.stringify({ ...body, model }),
            });
            if (!response.ok) throw new Error(`The summary request failed (HTTP ${response.status}).`);
            return response.json();
          },
        });
        compacted = { summary, upTo: finished.length };
        persist();
        notice(`Compacted ${fresh.length} earlier task${fresh.length === 1 ? "" : "s"}. New tasks start from a short summary; the steps above stay visible.`);
      } catch (e) {
        problem(`Could not compact: ${e.message}`);
      } finally { button.disabled = false; compactingSession = false; budgetKey = ""; refreshWorkContext(); }
    }

    // ---- actions
    async function review(changes) {
      changes.reviewing = !changes.reviewing;
      if (changes.reviewing && !changes.reviews) {
        changes.reviews = {};
        for (const f of changes.files) {
          if (f.binary) continue;
          try {
            const r = await ctx.request(`/v1/agent/${encodeURIComponent(changes.run_id)}/changes/${f.path.split("/").map(encodeURIComponent).join("/")}`);
            if (r.ok) changes.reviews[f.path] = (await r.json()).diff;
          } catch (_) { /* leave the preview */ }
        }
      }
      render();
      if (changes.reviewing) window.ArcPaper?.openReview($("#work-run .work-turn:last-child .work-result"));
    }

    async function undo(changes) {
      changes.undoing = true;
      render();
      try {
        const r = await ctx.request(`/v1/agent/${encodeURIComponent(changes.run_id)}/undo`, { method: "POST" });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || "Undo failed");
        const n = data.restored.length, k = data.skipped.length;
        changes.undone = { text: `Undone: ${n} file${n === 1 ? "" : "s"} restored.` + (k ? ` ${k} left alone because you changed ${k === 1 ? "it" : "them"} since.` : "") };
      } catch (e) {
        changes.undone = { text: "Couldn’t undo: " + (e.message || e) };
      }
      changes.undoing = false;
      persist();
      render();
    }

    function handle(event) {
      const step = (id) => run.steps.find((s) => s.id === id);
      switch (event.type) {
        case "activity": {
          if (["generating", "thinking", "answering"].includes(event.phase)) {
            run.activity = event.phase;
            if (event.phase === "generating") run.context = null;
          }
          break;
        }
        case "context": {
          if (Number.isFinite(event.prompt_tokens) && event.prompt_tokens >= 0) {
            run.context = { prompt_tokens: event.prompt_tokens, completion_tokens: 0, context_tokens: event.context_tokens, exact: event.exact === true, source: "request" };
            if (Number.isFinite(event.setup_tokens) && event.setup_tokens >= 0 && run.setupTokens == null) run.setupTokens = event.setup_tokens;
          }
          break;
        }
        case "performance": {
          if (Number.isFinite(event.generation_tok_s) && event.generation_tok_s > 0) run.generationTokS = event.generation_tok_s;
          break;
        }
        case "usage": {
          if (Number.isFinite(event.prompt_tokens) && event.prompt_tokens >= 0 && Number.isFinite(event.completion_tokens) && event.completion_tokens >= 0) run.context = { ...run.context, prompt_tokens: event.prompt_tokens, completion_tokens: event.completion_tokens, exact: true, source: "usage" };
          break;
        }
        case "status": run.statusText = event.message; if (/starting/i.test(event.message || "")) run.activity = "starting"; break;
        case "tool_call": {
          run.answerStreaming = false;
          run.activity = "tools";
          if (step(event.id)) break;
          const human = humanizeStep(event.name, event.arguments);
          run.steps.push({ id: event.id, human, state: "running", time: elapsedText(Date.now() - run.startedAt), open: human.kind === "other" ? false : false });
          break;
        }
        case "confirm_required": {
          const waiting = [...run.steps].reverse().find((s) => s.state === "running");
          if (waiting) waiting.state = "waiting";
          run.pending = { ...event, stepId: waiting ? waiting.id : null, run_id: event.run_id || run.id, answered: false };
          if (run.id !== run.pending.run_id) run.id = run.pending.run_id;
          break;
        }
        case "tool_result": {
          run.activity = "generating";
          const s = step(event.id);
          if (s) { s.state = event.error ? "failed" : "done"; s.output = event.content; s.time = elapsedText(Date.now() - run.startedAt); }
          if (run.pending && run.pending.stepId === event.id) run.pending = null;
          break;
        }
        case "assistant_delta":
          if (event.content) {
            if (run.answerStreaming) run.answer[run.answer.length - 1] += event.content;
            else { run.answer.push(event.content); run.answerStreaming = true; }
          }
          run.activity = "answering"; break;
        case "assistant": run.answerStreaming = false; if (event.content) run.answer.push(event.content); run.activity = "answering"; break;
        case "changes": run.changes = { ...event }; run.id = event.run_id || run.id; break;
        case "error": run.error = event.message || "Something went wrong."; run.status = "error"; run.sawEnd = true; break;
        case "done": run.sawEnd = true; if (run.status === "running") run.status = "done"; break;
        default: break;
      }
    }

    // Enter and the send button start a task; they never stop one. Stop is its own deliberate click.
    function submit() {
      if (run && run.status === "running") { notice("Still working on the last message. Wait for it to finish, or press Stop."); return; }
      begin();
    }
    function sendButton() {
      if (run && run.status === "running") { controller.abort(); return; }
      begin();
    }

    async function begin() {
      const task = $("#work-task").value.trim();
      if (compactingSession) { notice("Wait for compaction to finish before starting a task."); return; }
      if (!task) { $("#work-task").focus(); return; }
      // Local history browsing needs neither a harness process nor a selected model/folder.
      const localCommand = /^\/sessions\s*$/i.test(task);
      if (localCommand) {
        $("#work-task").value = ""; autosize();
        document.querySelector("#paper-sessions-toggle")?.click();
        await loadRecent();
        refreshWorkContext();
        return;
      }
      await loadCommands();
      const command = parseCommand(task);
      if (command && command.local) { await compactSession(command.args); if ($("#work-task").value.trim() === task) { $("#work-task").value = ""; autosize(); refreshWorkContext(); } return; }
      const folder = normalizeFolder($("#work-root").value, info && info.home);
      $("#work-root").value = folder;
      if (!folder) {
        problem("Choose a project folder first. Arc Llama uses it as the working folder for this task.", "folder");
        $("#work-root").focus();
        return;
      }
      const harness = $("#work-harness").value;
      if (!$("#work-model").value) syncModels();
      loadedAt = 0;                       // never trust a stale answer about what is in memory
      await preferLoadedModel();
      const model = $("#work-model").value;
      if (!model) { problem("Choose a model first. None is available yet."); return; }
      problem("");
      store(ROOT_KEY, folder);
      store(HARNESS_KEY, harness);
      const history = historyForRequest();
      run = { id: null, task, model, harness, folder, budgetScope: budgetScope(), activity: "starting", profileName: (profileFor() || {}).name || "", auto: autoOn, modelCold: loadedKnown && !loadedIds.has(model), folderName: baseName(folder) || "project", status: "running",
              startedAt: Date.now(), endedAt: 0, steps: [], answer: [], pending: null, changes: null, error: "", statusText: "Getting started" };
      runs.push(run);
      if (!sessionId) sessionId = newSessionId();
      persist();
      $("#work-task").value = "";
      autosize();
      clearInterval(tick);
      tick = setInterval(() => { if (run && run.status === "running") render(); }, 1000);
      render();
      controller = new AbortController();
      try {
        const res = await ctx.request("/v1/agent", {
          method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
          body: JSON.stringify({ model, task, harness, work_type: workType(), root: folder, work_session_id: sessionId, track_changes: true, local_only: true, auto_confirm: autoOn, history, ...(command ? { command: command.name, arguments: command.args } : {}), ...(profileFor() ? { tool_profile_id: profileFor().id } : {}) }),
        });
        if (!res.ok) {
          let detail = `The request failed (${res.status}).`;
          try {
            const body = await res.json();
            detail = body.detail || detail;
            if (body.closest) run.suggest = body.closest;
          } catch (_) {}
          throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let cut;
          while ((cut = buffer.indexOf("\n\n")) >= 0) {
            const block = buffer.slice(0, cut);
            buffer = buffer.slice(cut + 2);
            const line = block.split("\n").find((l) => l.startsWith("data:"));
            if (!line) continue;
            const payload = line.slice(5).trim();
            if (payload === "[DONE]") continue;
            try { handle(JSON.parse(payload)); } catch (_) { /* ignore a malformed frame */ }
            render();
          }
        }
        if (run.status === "running") {
          if (run.sawEnd) run.status = "done";
          else {
            run.status = "error";
            run.error = "The connection ended before the task finished. The arc-llama server log has the details.";
          }
        }
      } catch (e) {
        if (e && e.name === "AbortError") {
          run.status = "stopped";
          run.pending = null;
        } else {
          run.status = "error";
          run.error = e && e.message ? e.message : String(e);
        }
      } finally {
        loadedAt = 0;
        preferLoadedModel();              // the model is in memory now; update the note under the box
        run.endedAt = Date.now();
        clearInterval(tick);
        for (const s of run.steps) if (s.state === "running" || s.state === "waiting") s.state = run.status === "done" ? "done" : "failed";
        run.pending = null;
        budgetKey = "";
        render();
        persist();
      }
    }

    function notice(message) {
      const hint = $("#work-hint");
      hint.textContent = message || (autoOn ? "Auto-approve is on: edits and routine commands run without asking. Risky commands still ask, and you can Undo." : "Enter to start, Shift+Enter for a new line");
      hint.classList.toggle("error", !!message);
      if (!message) { modelNote(); folderNote(); }
    }

    // Problems the user must fix before anything can start are shown in the main area, not just the hint line.
    function problem(message, field) {
      const box = $("#work-problem");
      box.textContent = message || "";
      box.hidden = !message;
      const chip = $("#work-root").closest(".work-chip");
      if (chip) chip.classList.toggle("invalid", field === "folder" && !!message);
      if (message) notice(""); // keep the hint line from contradicting the banner
    }

    // A whole-home or filesystem-root folder is too large to track for Undo; say so (as a note, never an error).
    function folderIsWholeHome() {
      if (!info) return false;
      const raw = $("#work-root").value.trim().replace(/\/+$/, "");
      const home = (info.home || "").replace(/\/+$/, "");
      return raw === "/" || raw === "~" || (!!home && raw === home);
    }

    function folderNote() { modelNote(); }

    function autosize() {
      const t = $("#work-task");
      t.style.height = "auto";
      t.style.height = Math.min(160, t.scrollHeight) + "px";
    }

    function wire() {
      $("#mode-chat").addEventListener("click", () => setMode("chat"));
      $("#mode-work").addEventListener("click", () => setMode("work"));
      $("#work-send").addEventListener("click", sendButton);
      $("#work-compact").addEventListener("click", () => compactSession(""));
      $("#work-model").addEventListener("change", () => { budgetKey = ""; refreshWorkContext(); });
      window.addEventListener("arc-paper-models", refreshWorkContext);
      window.addEventListener("arc-paper-sessions-open", loadRecent);
      $("#work-auto").addEventListener("click", () => setAuto(!autoOn));
      renderAutoChip();
      $("#work-model").addEventListener("change", () => { modelPicked = true; store(MODEL_KEY, $("#work-model").value); modelNote(); });
      const task = $("#work-task");
      task.addEventListener("input", () => { refreshWorkContext(); autosize(); problem(""); notice(""); menuIndex = 0; if (typedCommand() !== null) loadCommands(); renderMenu(); });
      task.addEventListener("blur", () => { $("#work-commands").hidden = true; });
      $("#work-harness").addEventListener("change", () => { commandsKey = ""; updateHarnessControls(); refreshWorkContext(); });
      $("#work-profile").addEventListener("change", (e) => {
        if (e.target.value === "__manage") { e.target.value = profileId; openManager(!profiles.length); return; }
        chooseProfile(e.target.value);
      });
      $("#work-manager-done").addEventListener("click", closeManager);
      $("#work-manager").addEventListener("mousedown", (e) => { if (e.target === e.currentTarget) closeManager(); });
      document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#work-manager").hidden) closeManager(); });
      $("#work-root").addEventListener("change", () => { commandsKey = ""; });
      $("#work-root").addEventListener("input", () => { problem(""); notice(""); refreshWorkContext(); });
      $("#work-root").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); submit(); }
      });
      const source = $("#model-select");
      if (source) {
        new MutationObserver(syncModels).observe(source, { childList: true });
        source.addEventListener("change", () => { syncModels(); const t = $("#work-model"); if (source.value) t.value = source.value; });
      }
      task.addEventListener("keydown", (e) => {
        const open = !$("#work-commands").hidden && menuItems.length;
        if (open && (e.key === "ArrowDown" || e.key === "ArrowUp")) { e.preventDefault(); menuIndex = (menuIndex + (e.key === "ArrowDown" ? 1 : menuItems.length - 1)) % menuItems.length; renderMenu(); return; }
        if (open && e.key === "Escape") { e.preventDefault(); $("#work-commands").hidden = true; return; }
        if (open && (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey && !e.isComposing && typedCommand() !== menuItems[menuIndex].name.toLowerCase()))) { e.preventDefault(); pickCommand(menuItems[menuIndex]); return; }
        if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
      });
      $("#work-new").addEventListener("click", () => {
        if (compactingSession) { notice("Wait for compaction to finish first."); return; }
        if (run && run.status === "running") { notice("Stop the current task first."); return; }
        runs = [];
        run = null;
        compacted = null;
        sessionId = null;
        sessionOrigin = "";
        render();
        loadRecent();
        $("#work-task").focus();
      });
    }

    const onWorkPage = () => location.pathname === "/work" || location.hash === "#work";
    function workUnavailable(reason) {
      if (!onWorkPage()) return;
      const note = el("p", { class: "work-unavailable", role: "alert" }, reason);
      note.style.cssText = "margin:16px auto;max-width:640px;padding:12px 16px;border-radius:10px;background:#572d2d;color:#fff";
      document.body.prepend(note);
    }

    async function initInfo() {
      try {
        const r = await ctx.request("/v1/agent/info");
        if (!r.ok) return workUnavailable(r.status === 401 || r.status === 403 ? "Work needs the server admin token, which this browser could not get." : `Work could not start (HTTP ${r.status}).`);
        info = await r.json();
      } catch (_) { return workUnavailable("Could not reach the agent service."); }
      if (!info || !info.experimental) return workUnavailable("Work is not enabled on this server (set ARC_LLAMA_EXPERIMENTAL_AGENT=1).");
      $("#mode-switch").hidden = false;
      // Only offer auto-approve when the server says its adapter keeps destructive commands asking.
      $("#work-auto").hidden = !info.auto_approve;
      if (!info.auto_approve && autoOn) setAuto(false);
      const types = $("#work-type");
      if (types && Array.isArray(info.work_types) && info.work_types.length > 1) {
        types.replaceChildren(...info.work_types.map((t) => el("option", { value: t.id, disabled: !!t.needs_vision && !(info.vision_models || []).length }, t.label + (t.needs_vision && !(info.vision_models || []).length ? " (needs a vision model)" : ""))));
        $("#work-type-label").hidden = false;
        types.addEventListener("change", () => {
          // Documents runs on the built-in agent only.
          if (workType() === "documents") $("#work-harness").value = "builtin";
          $("#work-harness").disabled = workType() === "documents";
          commandsKey = ""; updateHarnessControls(); modelPicked = false; syncModels();
        });
      }
      const harness = $("#work-harness");
      harness.replaceChildren(...info.harnesses.map((h) => el("option", { value: h.id, disabled: !h.available }, h.label + (h.available ? "" : " (not installed)"))));
      const saved = recall(HARNESS_KEY);
      const pick = info.harnesses.find((h) => h.id === saved && h.available) || info.harnesses.find((h) => h.id === "opencode" && h.available) || info.harnesses[0];
      harness.value = pick.id;
      updateHarnessControls();
      // Default to the user's home folder (never the server's own working directory), then remember their choice.
      $("#work-root").value = recall(ROOT_KEY) || info.home || "";
      folderNote();
      syncModels();
      loadRecent();
      loadProfiles();
      if (onWorkPage()) setMode("work");
    }

    function begin_component() {
      wire();
      render();
      initInfo();
    }

    return { start: begin_component, humanizeStep, describeApproval, setMode, render, syncModels };
  };

  ArcChat.humanizeStep = humanizeStep;
  ArcChat.describeApproval = describeApproval;
  if (typeof module !== "undefined" && module.exports) module.exports = { humanizeStep, describeApproval, elapsedText, splitTask, normalizeFolder, isOutside };
})(typeof window !== "undefined" ? window : globalThis);
