// Hugging Face downloads and local model storage controller.
window.ArcDashboard = window.ArcDashboard || {};
window.ArcDashboard.createLibraryController = function createLibraryController({
  document: doc, request, authHeaders, makeButton, getDisplayName, formatMb,
  writeMessage, getSnapshot, fetchStatus, reviewModel, showModelsView, onJobsChange = () => {},
}) {
  // ---------------------------------------------------------------------------
  // Model library. Every string from Hugging Face is set with textContent.
  // ---------------------------------------------------------------------------

  // Badge text states the memory figure itself; option.need_mb is the file plus
  // the conversation that fit promises (model_library.memory_need_mb).
  function fitLabel(option, vramMb) {
    const need = option.need_mb > 0 ? `${Math.max(1, Math.round(option.need_mb / 1024))} GB` : null;
    const card = vramMb > 0 ? `${Math.round(vramMb / 1024)} GB` : null;
    switch (option.fit) {
      case "fits": return [need && card ? `Needs about ${need} of ${card}` : "Should fit in graphics memory", "ready", "Leaves room for a long conversation."];
      case "tight": return [need && card ? `Needs about ${need} of ${card} · short chats only` : "Short chats only", "warn", "Only a short conversation fits beside this file."];
      case "too_big": return [need && card ? `Needs about ${need} · more than this card's ${card}` : "More than this card's memory", "error", "Needs more graphics memory, or part of the model running on the CPU."];
      default: return ["Graphics memory unknown", "warn", "The file size or the card's memory isn't reported, so fit can't be worked out."];
    }
  }

  function compressionDescription(quant) {
    // Strip Unsloth's "UD-" (dynamic) prefix: UD-Q5_K_M is still a 5-bit copy.
    const code = String(quant || "unknown").toUpperCase().replace(/^UD-/, "");
    const bits = code.match(/^(?:I?Q)([1-8])(?:_|$)/)?.[1] ?? (/^MXFP4/.test(code) ? "4" : undefined);
    if (bits) {
      const description = Number(bits) <= 3
        ? "Very compact. Strong compression reduces memory use, but can affect answer quality."
        : Number(bits) === 4
          ? "Uses less memory than an 8-bit copy of the same model. Compression can affect answer quality."
          : Number(bits) < 8
            ? "A middle ground between smaller and less compressed copies of the same model."
            : "Keeps more numerical detail than a 4-bit copy of the same model, and uses more memory.";
      return [`${bits}-bit compressed`, description];
    }
    if (/^(BF16|F16|FP16|F32|FP32)$/.test(code)) {
      return [`${code.includes("32") ? "32" : "16"}-bit precision`, "Less compressed; usually a larger download with higher memory use than quantized copies of the same model."];
    }
    return ["Compression not reported", "This file name does not say how it was compressed. Check the model page before choosing."];
  }

  function modelSize(mb) {
    if (!(mb > 0)) return "Size not reported";
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GiB` : `${mb} MiB`;
  }

  const compatibilityIdentity = (target) => {
    const snapshot = getSnapshot();
    const model = target?.name && snapshot?.models?.find(item => item.name === target.name);
    return model && Object.hasOwn(model, "runtime_compatibility_identity") ? model.runtime_compatibility_identity : snapshot?.runtime_compatibility_identity;
  };
  let localCompatibility = null;
  const compatibilityHosts = new Map();
  function invalidateCompatibility() {
    for (const [host, state] of compatibilityHosts) {
      if (!host.isConnected) { state.cancel(); compatibilityHosts.delete(host); continue; }
      const identity = compatibilityIdentity(state.target);
      if (state.identity !== identity) { state.reset(); state.identity = identity; }
    }
  }
  function compatibilityControl(target) {
    const snapshot = getSnapshot();
    const identity = compatibilityIdentity(target);
    const model = target.name && snapshot?.models?.find(item => item.name === target.name);
    const localKey = target.name ? JSON.stringify([target.name, identity, model?.path, model?.file_readiness, model?.gpu_pci_slot]) : null;
    if (localKey && localCompatibility?.key === localKey) return localCompatibility.host;
    const host = doc.createElement("div");
    host.className = "library-compatibility";
    const message = doc.createElement("p");
    message.className = "plugin-meta";
    message.setAttribute("role", "status");
    message.textContent = "Runtime compatibility has not been checked.";
    const evidence = doc.createElement("details");
    evidence.className = "library-compatibility-evidence";
    evidence.hidden = true;
    const summary = doc.createElement("summary");
    summary.textContent = "What was checked";
    const explanation = doc.createElement("p");
    evidence.append(summary, explanation);
    let activeCheck = null;
    let revision = 0;
    const cancel = () => { revision++; activeCheck?.abort(); activeCheck = null; };
    const reset = () => {
      cancel();
      check.disabled = false;
      evidence.hidden = true;
      message.className = "plugin-meta";
      message.textContent = "Runtime changed. Check compatibility again.";
    };
    compatibilityHosts.set(host, { target, identity, reset, cancel });
    const check = makeButton("Check runtime compatibility", "secondary", async () => {
      const checkedIdentity = compatibilityIdentity(target);
      const attempt = ++revision;
      const controller = new AbortController();
      activeCheck = controller;
      const timeout = setTimeout(() => controller.abort(), 30000);
      check.disabled = true;
      evidence.hidden = true;
      message.textContent = "Checking model architecture against the installed runtime…";
      try {
        const response = await request("/admin/library/compatibility", {
          method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify(target), signal: controller.signal,
        });
        const result = await response.json().catch(() => ({}));
        if (attempt !== revision) return;
        if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
        if (checkedIdentity !== compatibilityIdentity(target)) throw new Error("Runtime changed. Check again.");
        message.textContent = `${result.label}: ${result.detail} ${result.action}`;
        message.className = `plugin-meta${result.status === "incompatible" ? " library-error" : ""}`;
        explanation.textContent = [result.scope, result.source, result.runtime && `Runtime: ${result.runtime} (${result.runtime_fingerprint || "unknown identity"})`, result.backend && `Configured backend: ${result.backend}`, result.architecture && `Architecture: ${result.architecture}`, result.revision && `Model revision: ${result.revision}`].filter(Boolean).join(" · ");
        evidence.hidden = false;
      } catch (e) {
        if (attempt !== revision) return;
        message.textContent = `Compatibility unknown: ${e.name === "AbortError" ? "The check timed out. Try again." : e.message}`;
        message.className = "plugin-meta";
      } finally {
        clearTimeout(timeout);
        if (attempt === revision) { activeCheck = null; check.disabled = false; }
      }
    });
    host.append(check, message, evidence);
    if (localKey) localCompatibility = { key: localKey, host };
    return host;
  }

  async function librarySearch(event) {
    event.preventDefault();
    const host = doc.querySelector("#library-results");
    const query = doc.querySelector("#library-query").value.trim();
    if (query.length < 2) return;
    writeMessage(host, "Searching…");
    try {
      const r = await request(`/admin/library/search?q=${encodeURIComponent(query)}`, { headers: authHeaders() });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      host.replaceChildren();
      if (!data.results.length) { writeMessage(host, "No GGUF repositories matched."); return; }
      for (const item of data.results) host.appendChild(repoCard(item).card);
    } catch (e) {
      writeMessage(host, `Search failed: ${e.message}`, "library-note error");
    }
  }

  // Derive a fixed HF URL from a namespace, never accept an arbitrary image URL.
  function avatarUrl(namespace) {
    return /^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(namespace) ? `https://huggingface.co/api/avatars/${encodeURIComponent(namespace)}` : null;
  }

  function repoCard(item) {
    const card = doc.createElement("article");
    card.className = "plugin-card library-repo";
    const main = doc.createElement("div");
    const title = doc.createElement("h3");
    title.textContent = item.repo;
    const meta = doc.createElement("p");
    meta.className = "plugin-meta";
    meta.textContent = `${(item.downloads ?? 0).toLocaleString()} downloads · ${(item.likes ?? 0).toLocaleString()} likes`;
    const options = doc.createElement("div");
    options.className = "library-options";
    const identity = doc.createElement("div");
    identity.className = "library-repo-identity";
    const publisher = String(item.repo || "").split("/")[0];
    const avatar = doc.createElement("span");
    avatar.className = "library-publisher-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = publisher.slice(0, 2).toUpperCase() || "HF";
    if (avatarUrl(publisher)) {
      const image = doc.createElement("img");
      image.alt = "";
      image.width = 36;
      image.height = 36;
      image.loading = "lazy";
      image.referrerPolicy = "no-referrer";
      image.addEventListener("error", () => image.remove(), { once: true });
      image.src = avatarUrl(publisher);
      avatar.appendChild(image);
    }
    const heading = doc.createElement("div");
    const byline = doc.createElement("p");
    byline.className = "plugin-meta library-publisher";
    byline.textContent = `Published by ${publisher || "an unknown account"} on Hugging Face`;
    byline.title = "The repository publisher may differ from the original model developer.";
    heading.append(title, byline);
    identity.append(avatar, heading);
    main.append(identity, meta, options);
    const side = doc.createElement("div");
    side.appendChild(makeButton("Compare versions", "secondary", () => libraryShowRepo(item.repo, options)));
    card.append(main, side);
    return { card, options };
  }

  // ---------------------------------------------------------------------------
  // Discover shelves: hand-picked models sized for this GPU (server: discover.py).
  // Curated text is ours; every Hub figure is still set with textContent.
  // ---------------------------------------------------------------------------

  const LICENSES = { "apache-2.0": "Apache 2.0", mit: "MIT", gemma: "Gemma license", other: "Custom license" };
  const gb = (mb) => mb >= 10240 ? `${Math.round(mb / 1024)} GB` : `${(mb / 1024).toFixed(1)} GB`;
  const compact = (n) => n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}K` : String(n);
  function node(tag, className, text) {
    const element = doc.createElement(tag);
    if (className) element.className = className;
    if (text != null) element.textContent = text;
    return element;
  }

  let featured = null;
  let featuredLoad = null;
  let publisherFilter = null;
  const featuredButtons = new Map();

  function avatarImage(namespace, className) {
    const url = avatarUrl(namespace);
    if (!url) return null;
    const image = node("img", className);
    image.alt = "";
    image.loading = "lazy";
    image.decoding = "async";
    image.referrerPolicy = "no-referrer";
    image.addEventListener("error", () => image.remove(), { once: true });
    image.src = url;
    return image;
  }

  function memoryLine(pick, vram) {
    const line = node("div", "discover-mem");
    const text = node("div", "discover-mem-text");
    if (!vram || !pick.need_mb) {
      text.append(node("span", null, `${gb(pick.size_mb)} download`), node("span", null, "Graphics memory unknown"));
      line.append(text);
      return line;
    }
    const need = node("span");
    need.append("Needs about ", node("b", null, `${Math.max(1, Math.round(pick.need_mb / 1024))} GB`), ` of ${Math.round(vram / 1024)} GB`);
    const spare = vram - pick.need_mb;
    const right = pick.fit === "tight"
      ? node("span", "discover-mem-warn", "Room for short chats only")
      : pick.fit === "too_big" ? node("span", "discover-mem-warn", "More than this card has")
      : node("span", null, `${Math.max(0, Math.floor(spare / 1024))} GB to spare`);
    text.append(need, right);
    const meter = node("div", "discover-meter");
    meter.setAttribute("aria-hidden", "true");
    const file = node("i", "discover-meter-file");
    file.style.width = `${Math.min(100, pick.size_mb / vram * 100)}%`;
    const chat = node("i", "discover-meter-chat");
    chat.style.left = file.style.width;
    chat.style.width = `${Math.max(0, Math.min(100 - pick.size_mb / vram * 100, (pick.need_mb - pick.size_mb) / vram * 100))}%`;
    meter.append(file, chat);
    line.append(text, meter);
    return line;
  }

  function showOtherVersions(repo) {
    const host = doc.querySelector("#library-results");
    if (!host) return;
    const { card, options } = repoCard({ repo });
    host.replaceChildren(card);
    libraryShowRepo(repo, options);
    card.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function featuredCard(entry, vram) {
    const pick = entry.pick;
    const card = node("article", "discover-card");
    card.setAttribute("aria-label", `${entry.name} by ${entry.author_name}`);
    const cover = node("div", "discover-cover");
    cover.setAttribute("aria-hidden", "true");
    const wash = avatarImage(entry.author, "discover-wash");
    const sleeve = avatarImage(entry.author, "discover-sleeve");
    const initial = node("span", "discover-sleeve discover-sleeve-initial", entry.author_name.slice(0, 1));
    if (sleeve) sleeve.addEventListener("error", () => cover.append(initial), { once: true });
    cover.append(...[wash, sleeve || initial].filter(Boolean));
    const params = node("span", "discover-params", entry.params);
    params.append(node("small", null, " parameters"));
    cover.append(params);

    const body = node("div", "discover-body");
    const by = node("p", "discover-by");
    by.append(node("b", null, entry.author_name));
    if (entry.converter) by.append(node("span", null, ` · file by ${entry.converter}`));
    const name = node("h4", "discover-name", entry.name);
    const description = node("p", "discover-desc", entry.description);
    const facts = node("div", "discover-facts");
    if (entry.license) facts.append(node("span", null, LICENSES[entry.license] || entry.license));
    if (Number.isFinite(entry.downloads)) facts.append(node("span", null, `${compact(entry.downloads)} downloads`));
    if (entry.vision) facts.append(node("span", null, "Reads images"));
    const foot = node("div", "discover-foot");
    const get = makeButton(`Get · ${gb(pick.size_mb)}`, "primary", async () => {
      get.disabled = true;
      get.textContent = "Starting…";
      try {
        const response = await request("/admin/library/download", {
          method: "POST",
          headers: authHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify({ repo: entry.repo, file: pick.file, size_mb: pick.size_mb }),
        });
        const job = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(job.detail || `HTTP ${response.status}`);
        poll({ force: true });
      } catch (e) {
        get.disabled = false;
        get.textContent = `Get · ${gb(pick.size_mb)}`;
        get.title = `Download did not start: ${e.message}`;
        status.textContent = `Download did not start: ${e.message}`;
      }
    });
    get.setAttribute("aria-label", `Download ${entry.name}, ${gb(pick.size_mb)}`);
    featuredButtons.set(JSON.stringify([entry.repo, pick.file]), { button: get, label: `Get · ${gb(pick.size_mb)}` });
    const other = makeButton("Other versions", "ghost", () => showOtherVersions(entry.repo));
    other.setAttribute("aria-label", `Other versions of ${entry.name}`);
    const status = node("p", "discover-status");
    status.setAttribute("role", "status");
    foot.append(other, get);
    body.append(by, name, description, memoryLine(pick, vram), facts, foot, status);
    card.append(cover, body);
    return card;
  }

  function updateFeaturedButtons() {
    for (const [key, { button, label }] of featuredButtons) {
      if (!button.isConnected) { featuredButtons.delete(key); continue; }
      const [repo, file] = JSON.parse(key);
      const job = [...knownJobs].reverse().find((item) => item.repo === repo && item.file === file && !supersededFailures.has(item.id));
      if (job && ["queued", "downloading", "registering"].includes(job.status)) {
        const pct = job.bytes_total ? Math.min(99, Math.round(job.bytes_done / job.bytes_total * 100)) : null;
        button.textContent = job.status === "downloading" && pct != null ? `Downloading ${pct}%` : job.status === "registering" ? "Adding…" : "Queued…";
        button.disabled = true;
      } else if (job?.status === "done") {
        button.textContent = "Downloaded";
        button.disabled = true;
      } else if (button.textContent !== "Starting…") {
        button.textContent = label;
        button.disabled = false;
      }
    }
  }

  function renderFeatured() {
    const host = doc.querySelector("#discover-featured");
    if (!host || !featured) return;
    featuredButtons.clear();
    const vram = featured.vram_mb;
    const head = node("div", "discover-head");
    const hardware = node("p", "discover-hardware");
    if (vram) {
      hardware.append("Picked for your ", node("b", null, featured.gpu_name || "graphics card"), " · ", node("b", null, `${Math.round(vram / 1024)} GB`), " graphics memory");
    } else {
      hardware.textContent = "Graphics memory is unknown, so sizes aren't checked against your card.";
    }
    head.append(hardware);

    const usable = featured.entries.filter((entry) => !entry.error && entry.pick && entry.pick.fit !== "too_big");
    const byId = new Map(usable.map((entry) => [entry.id, entry]));
    const publishers = node("div", "discover-publishers");
    publishers.setAttribute("role", "group");
    publishers.setAttribute("aria-label", "Show models from one publisher");
    for (const publisher of featured.publishers) {
      if (!usable.some((entry) => entry.author === publisher.id)) continue;
      const button = node("button", "discover-publisher");
      button.type = "button";
      button.setAttribute("aria-pressed", String(publisherFilter === publisher.id));
      const glow = avatarImage(publisher.id, "discover-publisher-wash");
      const face = avatarImage(publisher.id, "discover-publisher-avatar") || node("span", "discover-publisher-avatar", publisher.name.slice(0, 1));
      const label = node("span", "discover-publisher-label");
      label.append(node("b", null, publisher.name), node("span", null, publisher.families));
      button.append(...[glow, face, label].filter(Boolean));
      button.addEventListener("click", () => {
        publisherFilter = publisherFilter === publisher.id ? null : publisher.id;
        renderFeatured();
        doc.querySelector(`.discover-publisher[aria-pressed="true"]`)?.focus();
      });
      publishers.append(button);
    }

    const shelves = node("div", "discover-shelves");
    const shelf = (title, subtitle, entries) => {
      const section = node("section", "discover-shelf");
      const heading = node("div", "discover-shelf-head");
      heading.append(node("h3", null, title), node("p", null, subtitle));
      const cards = node("div", "discover-cards");
      for (const entry of entries) cards.append(featuredCard(entry, vram));
      section.append(heading, cards);
      return section;
    };
    if (publisherFilter) {
      const publisher = featured.publishers.find((item) => item.id === publisherFilter);
      const entries = usable.filter((entry) => entry.author === publisherFilter);
      const section = shelf(`From ${publisher.name}`, `${publisher.families} models that run on this card`, entries);
      const more = makeButton(`Search Hugging Face for more ${publisher.families.split(",")[0]} models`, "ghost", () => {
        const input = doc.querySelector("#library-query");
        input.value = publisher.families.split(",")[0];
        doc.querySelector("#library-search")?.requestSubmit();
      });
      section.append(more);
      shelves.append(section);
    } else {
      for (const item of featured.shelves) {
        const entries = item.entries.map((id) => byId.get(id)).filter(Boolean);
        if (entries.length) shelves.append(shelf(item.title, item.subtitle, entries));
      }
    }
    if (!shelves.childElementCount) {
      shelves.append(node("p", "library-note", featured.offline
        ? "Recommendations need a connection to Hugging Face. Check your connection, or search when you're back online."
        : "None of the recommended models fit this card. Search for smaller models above."));
      if (featured.offline) shelves.append(makeButton("Try again", "secondary", () => loadFeatured({ force: true })));
    }
    host.replaceChildren(head, publishers, shelves);
    if (!publishers.childElementCount) publishers.remove();
    updateFeaturedButtons();
  }

  function loadFeatured({ force = false } = {}) {
    const host = doc.querySelector("#discover-featured");
    if (!host) return Promise.resolve();
    if (featuredLoad) return featuredLoad;
    if (featured && !featured.offline && !force) return Promise.resolve();
    if (!featured) writeMessage(host, "Finding models for your graphics card…");
    featuredLoad = (async () => {
      try {
        const response = await request("/admin/library/featured", { headers: authHeaders() });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !Array.isArray(data.entries)) throw new Error(data.detail || `HTTP ${response.status}`);
        featured = data;
        renderFeatured();
      } catch (e) {
        featured = null;
        writeMessage(host, `Recommendations are unavailable: ${e.message}. Search still works.`, "library-note error");
        host.appendChild(makeButton("Try again", "secondary", () => loadFeatured({ force: true })));
      } finally {
        featuredLoad = null;
      }
    })();
    return featuredLoad;
  }

  // Folder the next download goes to. Remembered per browser; blank means the
  // server's configured models folder. Free space is checked as the user types.
  const FOLDER_KEY = "arc-llama.download-folder";
  function downloadFolderControl() {
    const node = doc.createElement("div");
    node.className = "library-folder";
    const label = doc.createElement("label");
    label.textContent = "Download folder on this server";
    const input = doc.createElement("input");
    input.type = "text";
    input.className = "library-folder-input";
    input.spellcheck = false;
    input.placeholder = "Default models folder";
    input.setAttribute("aria-label", "Download folder");
    try { input.value = localStorage.getItem(FOLDER_KEY) || ""; } catch { /* storage unavailable */ }
    const hint = doc.createElement("p");
    hint.className = "plugin-meta";
    hint.setAttribute("role", "status");
    node.append(label, input, hint);
    let timer = null;
    let seq = 0;
    async function refresh() {
      const mine = ++seq;
      const path = input.value.trim();
      try { localStorage.setItem(FOLDER_KEY, path); } catch { /* storage unavailable */ }
      try {
        let report;
        if (path) {
          const r = await request(`/admin/library/folder?path=${encodeURIComponent(path)}`, { headers: authHeaders() });
          report = await r.json().catch(() => ({}));
          if (!r.ok) throw new Error(report.detail || `HTTP ${r.status}`);
        } else {
          const r = await request("/admin/library/disk", { headers: authHeaders() });
          const d = await r.json().catch(() => ({}));
          if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
          report = { path: d.models_dir, free_mb: d.free_mb, exists: true };
        }
        if (mine !== seq) return;
        hint.className = "plugin-meta";
        hint.textContent = `${report.exists ? "" : "Will be created. "}${formatMb(report.free_mb)} free in ${report.path}`;
      } catch (e) {
        if (mine !== seq) return;
        hint.className = "library-note error";
        hint.textContent = e.message;
      }
    }
    input.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(refresh, 400); });
    refresh();
    return { node, value: () => input.value.trim() };
  }

  async function libraryShowRepo(repo, host) {
    writeMessage(host, "Reading files…");
    try {
      const r = await request(`/admin/library/repo?repo=${encodeURIComponent(repo)}`, { headers: authHeaders() });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      host.replaceChildren();
      const estimate = doc.createElement("p");
      estimate.className = "plugin-meta";
      estimate.textContent = data.vram_mb > 0
        ? `GPU capacity used for estimates: ${modelSize(data.vram_mb)}. Fit includes an allowance for conversation memory; actual use depends on the model and settings.`
        : "GPU memory capacity is unavailable. Fit cannot be estimated yet.";
      host.appendChild(estimate);
      if (data.vision) {
        const v = doc.createElement("p");
        v.className = "plugin-meta";
        v.textContent = "Extra vision file: this repository includes an image adapter (projector). A matching adapter downloads automatically; its extra size is not included below. Review the pairing after download.";
        host.appendChild(v);
      }
      if (!data.options.length) { writeMessage(host, "No downloadable GGUF files."); return; }
      const folder = downloadFolderControl();
      host.appendChild(folder.node);
      for (const option of data.options) {
        const row = doc.createElement("div");
        row.className = "library-option";
        const main = doc.createElement("div");
        main.className = "library-option-main";
        const [plainName, explanation] = compressionDescription(option.quant);
        const name = doc.createElement("strong");
        name.className = "library-file";
        name.textContent = plainName;
        const summary = doc.createElement("p");
        summary.className = "plugin-meta";
        summary.textContent = explanation;
        const size = doc.createElement("p");
        size.className = "library-option-size";
        size.textContent = `Download: ${modelSize(option.size_mb)}${option.shards > 1 ? ` total across ${option.shards} files` : ""}`;
        const [label, tone, fitExplanation] = fitLabel(option, data.vram_mb);
        const badge = doc.createElement("span");
        badge.className = `status-pill ${tone}`;
        badge.textContent = label;
        const fit = doc.createElement("p");
        fit.className = "plugin-meta";
        fit.textContent = fitExplanation;
        const details = doc.createElement("details");
        details.className = "library-technical";
        const detailLabel = doc.createElement("summary");
        detailLabel.textContent = "Technical details";
        const filename = doc.createElement("p");
        filename.textContent = `File: ${option.file}`;
        const encoding = doc.createElement("p");
        encoding.textContent = `Encoding: ${option.quant || "unknown"}`;
        details.append(detailLabel, filename, encoding);
        const compatibility = compatibilityControl({ repo, file: option.file });
        main.append(name, summary, size, badge, fit, compatibility, details);
        const download = makeButton("Download", "secondary", async () => {
          download.disabled = true;
          try {
            const resp = await request("/admin/library/download", {
              method: "POST",
              headers: authHeaders({ "Content-Type": "application/json" }),
              body: JSON.stringify({ repo, file: option.file, size_mb: option.size_mb, dest_dir: folder.value() || undefined }),
            });
            const job = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(job.detail || `HTTP ${resp.status}`);
            poll({ force: true });
          } catch (e) {
            download.disabled = false;
            badge.textContent = `Download failed: ${e.message}`;
            badge.className = "status-pill error";
          }
        });
        if (option.fit === "too_big") download.title = "Estimated to exceed GPU memory. Review memory and offloading settings before loading.";
        row.append(main, download);
        host.appendChild(row);
      }
    } catch (e) {
      writeMessage(host, `Could not read ${repo}: ${e.message}`, "library-note error");
    }
  }

  const retryingDownloads = new Set();
  const supersededFailures = new Set();
  const retriedJobs = new Set();
  let knownJobs = [];

  function recoveryMessage(text, error = false) {
    const notice = doc.querySelector("#library-recovery-status");
    if (!notice) return;
    notice.hidden = !text;
    notice.textContent = text;
    notice.className = error ? "library-note error" : "section-copy";
    if (error) { notice.scrollIntoView({ behavior: "smooth", block: "center" }); notice.focus({ preventScroll: true }); }
  }

  async function retryDownload(job, control) {
    const key = JSON.stringify([job.repo, job.file]);
    if (retryingDownloads.has(key) || supersededFailures.has(job.id)) return false;
    retryingDownloads.add(key);
    if (control) control.disabled = true;
    recoveryMessage("Starting another attempt for the same file…");
    let accepted = false;
    try {
      const body = { repo: job.repo, file: job.file };
      // target_dir is <chosen folder>/<repo name>; retry into the same folder.
      if (job.target_dir) {
        const target = job.target_dir.replace(/\\/g, "/");
        const parent = target.replace(/\/[^/]+\/?$/, "");
        // A drive root needs its separator: "D:" is drive-relative on Windows.
        body.dest_dir = /^[A-Za-z]:$/.test(parent) ? `${parent}/` : parent || "/";
      }
      if (Number.isFinite(job.bytes_total) && job.bytes_total >= 1048576) body.size_mb = Math.floor(job.bytes_total / 1048576);
      const response = await request("/admin/library/download", { method: "POST", headers: authHeaders({ "Content-Type": "application/json" }), body: JSON.stringify(body) });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
      accepted = true;
      supersededFailures.add(job.id);
      for (const previous of knownJobs) {
        if (previous.status === "error" && previous.repo === job.repo && previous.file === job.file) supersededFailures.add(previous.id);
      }
      if (result.id) retriedJobs.add(result.id);
      recoveryMessage("Retry queued. You can keep using the app while it downloads.");
      await poll({ force: true });
      return true;
    } catch (error) {
      recoveryMessage(accepted
        ? "The retry was queued, but its status could not be refreshed. Check downloads again."
        : `Could not start the retry: ${error.message}. Check your connection and try again.`, true);
      return false;
    } finally {
      retryingDownloads.delete(key);
      if (control && !accepted) control.disabled = false;
      // Polling may have replaced the clicked node while the submission waited.
      if (!accepted) await poll({ refreshStatus: false, force: true });
    }
  }

  let libraryJobTimer = null;
  let bound = false;
  let started = false;
  let pollRequest = null;
  let queuedPoll = null;
  let renderedJobsSignature = null;
  const refreshedCompletions = new Set();

  function schedulePoll(delay) {
    clearTimeout(libraryJobTimer);
    libraryJobTimer = setTimeout(poll, doc.hidden ? 30000 : delay);
  }

  function poll(options = {}) {
    if (pollRequest) {
      if (!options.force) return pollRequest;
      if (!queuedPoll) queuedPoll = pollRequest.then(() => poll({ ...options, force: false })).finally(() => { queuedPoll = null; });
      return queuedPoll;
    }
    clearTimeout(libraryJobTimer);
    pollRequest = pollJobs(options).finally(() => { pollRequest = null; });
    return pollRequest;
  }

  async function pollJobs({ refreshStatus = true } = {}) {
    const host = doc.querySelector("#library-jobs");
    if (!host) return;
    let jobs = [];
    try {
      const r = await request("/admin/library/jobs", { headers: authHeaders() });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      jobs = (await r.json()).jobs || [];
      if (!Array.isArray(jobs)) throw new Error("Invalid download status");
    } catch (_) {
      onJobsChange(null);
      writeMessage(host, "Download status unavailable. Check again to refresh it.", "library-note error");
      host.appendChild(makeButton("Check downloads again", "secondary", () => poll()));
      renderedJobsSignature = null;
      schedulePoll(5000);
      return;
    }
    knownJobs = jobs;
    updateFeaturedButtons();
    for (const job of jobs) {
      if (!retriedJobs.has(job.id) || !["done", "error"].includes(job.status)) continue;
      retriedJobs.delete(job.id);
      recoveryMessage(job.status === "done" ? "Download finished. Review the model below." : "The download did not finish. You can retry again or choose another model.");
    }
    onJobsChange(jobs);
    const completed = jobs.filter(job => job.status === "done" && !refreshedCompletions.has(job.id));
    let retryCompletion = false;
    if (completed.length && refreshStatus) {
      if (await fetchStatus(true)) for (const job of completed) refreshedCompletions.add(job.id);
      else retryCompletion = true;
    }
    const active = jobs.some(job => ["queued", "downloading", "registering"].includes(job.status));
    if (active || retryCompletion) schedulePoll(active ? 2000 : 5000);
    const visibleJobs = jobs.filter(job => !supersededFailures.has(job.id));
    const signature = JSON.stringify([visibleJobs, [...retryingDownloads], (getSnapshot()?.models || []).map(model => [model.name, getDisplayName(model)])]);
    if (signature === renderedJobsSignature) return;
    host.replaceChildren();
    for (const job of visibleJobs.slice(0, 6)) {
      const row = doc.createElement("div");
      row.className = "library-job";
      const name = doc.createElement("span");
      name.textContent = `${job.repo} · ${job.file.split("/").pop()}`;
      const state = doc.createElement("strong");
      if (job.status === "downloading") {
        const pct = job.bytes_total ? Math.min(99, Math.round(job.bytes_done / job.bytes_total * 100)) : null;
        state.textContent = pct == null ? `downloading ${formatMb(Math.round(job.bytes_done / 1048576))}` : `downloading ${pct}%`;
      } else if (job.status === "done") {
        const registered = Array.isArray(job.registered) ? job.registered : [];
        state.textContent = registered.length ? "downloaded and added" : "downloaded (registration details unavailable)";
        const actions = doc.createElement("div");
        actions.className = "library-job-actions";
        if (!registered.length) {
          actions.appendChild(makeButton("View your models", "secondary", () => showModelsView("#choose-title")));
        } else {
          const visible = registered.filter((modelName) => getSnapshot()?.models?.some((model) => model.name === modelName));
          for (const modelName of visible) {
            const model = getSnapshot().models.find((item) => item.name === modelName);
            const review = makeButton(`Review ${getDisplayName(model)}`, "secondary", async () => {
              if (!(await reviewModel(modelName))) await poll();
            });
            review.setAttribute("aria-label", `Review model ${modelName}`);
            review.title = `Review ${getDisplayName(model)} (${modelName})`;
            actions.appendChild(review);
          }
          if (visible.length < registered.length) {
            const pending = doc.createElement("span");
            pending.className = "library-note";
            pending.textContent = "Registration is not visible yet. Refresh to review the downloaded model.";
            actions.appendChild(pending);
            actions.appendChild(makeButton("Refresh to review", "secondary", async (event) => {
              event.currentTarget.disabled = true;
              await fetchStatus(true);
              await poll({ refreshStatus: false });
            }));
          }
        }
        row.append(name, state, actions);
        host.appendChild(row);
        continue;
      } else if (job.status === "error") {
        state.textContent = "Download did not finish";
        state.className = "library-error";
        const actions = doc.createElement("div");
        actions.className = "library-job-actions";
        const retry = makeButton("Retry download", "secondary", event => retryDownload(job, event.currentTarget));
        retry.disabled = retryingDownloads.has(JSON.stringify([job.repo, job.file]));
        retry.setAttribute("aria-label", `Retry download ${job.file}`);
        const find = makeButton("Find another model", "ghost", () => showModelsView("#library-title"));
        const details = doc.createElement("details");
        details.className = "library-technical";
        const summary = doc.createElement("summary"); summary.textContent = "Technical details";
        const error = doc.createElement("pre"); error.textContent = job.error || "No error details were reported.";
        details.append(summary, error);
        actions.append(retry, find, details);
        row.append(name, state, actions);
        host.appendChild(row);
        continue;
      } else {
        state.textContent = job.status;
      }
      row.append(name, state);
      host.appendChild(row);
    }
    renderedJobsSignature = signature;
  }

  async function loadLibraryDisk() {
    const host = doc.querySelector("#library-disk");
    if (!host) return;
    try {
      const r = await request("/admin/library/disk", { headers: authHeaders() });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      host.replaceChildren();
      const summary = doc.createElement("p");
      summary.className = "plugin-meta";
      const used = data.models.reduce((n, m) => n + m.size_mb, 0);
      summary.textContent = `${formatMb(used)} in ${data.models.length} model(s) · ${formatMb(data.free_mb)} free in ${data.models_dir}`;
      host.appendChild(summary);
      for (const m of data.models) {
        const row = doc.createElement("div");
        row.className = "library-option";
        const name = doc.createElement("span");
        const when = m.last_used ? new Date(m.last_used * 1000).toLocaleDateString() : "no recorded use";
        name.textContent = `${m.name} · ${formatMb(m.size_mb)} · ${m.missing ? "file missing" : when}`;
        const remove = makeButton(m.managed ? "Delete" : "Unregister", "ghost", async () => {
          const verb = m.managed ? `Delete ${m.name} and its files` : `Unregister ${m.name} (files stay where they are)`;
          if (!confirm(`${verb}?`)) return;
          remove.disabled = true;
          const resp = await request(`/admin/library/models/${encodeURIComponent(m.name)}?delete_files=${m.managed}`, {
            method: "DELETE", headers: authHeaders(),
          });
          if (resp.ok) { await loadLibraryDisk(); fetchStatus(true); }
          else remove.disabled = false;
        });
        remove.title = m.managed ? "Files inside the models folder are deleted" : "Outside the models folder: only the registration is removed";
        row.append(name, remove);
        host.appendChild(row);
      }
    } catch (e) {
      writeMessage(host, `Disk usage unavailable: ${e.message}`, "library-note error");
    }
  }

  function bind() {
    if (bound) return;
    bound = true;
    doc.querySelector("#library-search")?.addEventListener("submit", librarySearch);
    doc.querySelector("#library-disk-details")?.addEventListener("toggle", (event) => {
      if (event.target.open) loadLibraryDisk();
    });
  }

  return {
    bind,
    compatibilityControl,
    invalidateCompatibility,
    start() {
      if (started) return;
      started = true;
      doc.addEventListener("visibilitychange", () => {
        clearTimeout(libraryJobTimer);
        if (!doc.hidden) poll();
        else schedulePoll(30000);
      });
      poll();
    },
    poll,
    loadFeatured,
    retry: retryDownload,
    loadDisk: loadLibraryDisk,
  };
};
