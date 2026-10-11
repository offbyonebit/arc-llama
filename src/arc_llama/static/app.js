// arc-llama dashboard: a first-run friendly view over the local model registry.

/*
 * Isaiah 43:19 (KJV): UTF-8, in binary.
 * 01000010 01100101 01101000 01101111 01101100 01100100 00101100 00100000
 * 01001001 00100000 01110111 01101001 01101100 01101100 00100000 01100100
 * 01101111 00100000 01100001 00100000 01101110 01100101 01110111 00100000
 * 01110100 01101000 01101001 01101110 01100111 00111011 00100000 01101110
 * 01101111 01110111 00100000 01101001 01110100 00100000 01110011 01101000
 * 01100001 01101100 01101100 00100000 01110011 01110000 01110010 01101001
 * 01101110 01100111 00100000 01100110 01101111 01110010 01110100 01101000
 * 00111011 00100000 01110011 01101000 01100001 01101100 01101100 00100000
 * 01111001 01100101 00100000 01101110 01101111 01110100 00100000 01101011
 * 01101110 01101111 01110111 00100000 01101001 01110100 00111111 00100000
 * 01001001 00100000 01110111 01101001 01101100 01101100 00100000 01100101
 * 01110110 01100101 01101110 00100000 01101101 01100001 01101011 01100101
 * 00100000 01100001 00100000 01110111 01100001 01111001 00100000 01101001
 * 01101110 00100000 01110100 01101000 01100101 00100000 01110111 01101001
 * 01101100 01100100 01100101 01110010 01101110 01100101 01110011 01110011
 * 00101100 00100000 01100001 01101110 01100100 00100000 01110010 01101001
 * 01110110 01100101 01110010 01110011 00100000 01101001 01101110 00100000
 * 01110100 01101000 01100101 00100000 01100100 01100101 01110011 01100101
 * 01110010 01110100 00101110
 */

const $ = (selector) => document.querySelector(selector);
const THEME_KEY = "arc-llama-theme";
function applyTheme(theme) {
  const dark = theme !== "light";
  if (document.documentElement) document.documentElement.dataset.theme = dark ? "dark" : "light";
  const toggle = $("#theme-toggle");
  if (toggle) { toggle.textContent = dark ? "Light theme" : "Dark theme"; toggle.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme"); }
  if (document.querySelectorAll) document.querySelectorAll(".brand-logo").forEach((logo) => { logo.src = logo.dataset[dark ? "dark" : "light"] || logo.src; });
}
applyTheme(typeof localStorage === "undefined" ? "dark" : (localStorage.getItem(THEME_KEY) || "dark"));
const MIB = 1024;
const SELECTED_MODEL_KEY = "arc-llama-selected-model";

let snapshot = null;
let statusOnline = false;
let statusError = "";
let libraryJobs = null;
let libraryJobsUnavailable = false;
const reviewedDownloads = new Set();
let selectedModel = null;
let modelSelectionInitialized = false;
let adminToken = null;
let statusRequest = null;
let queuedStatusRequest = null;
let renderedModelsSignature = null;
let scanning = false;
const openDetailModels = new Set();

const fmtGiB = (mb) => mb == null ? "Unknown" : `${(mb / MIB).toFixed(mb >= MIB ? 1 : 0)} GiB`;
const fmtCtx = (ctx) => ctx ? `${Number(ctx).toLocaleString()} tokens` : "Default context";
const displayName = (model) => model.display_name || model.name;

function authHeaders(headers = {}) {
  return adminToken ? { ...headers, Authorization: `Bearer ${adminToken}` } : headers;
}

async function initAdminToken() {
  try {
    const response = await fetch("/admin/session-token");
    if (response.ok) adminToken = (await response.json()).admin_token || null;
  } catch (_) {
    // A remote deployment cannot expose its local session token. Read-only UI
    // state remains useful; protected actions explain their failure.
  }
}

function setServerState(kind, text) {
  const state = $("#server-state");
  const intro = $("#intro-status");
  state.className = `server-state ${kind}`;
  state.textContent = text;
  if (intro) {
    intro.className = `intro-status ${kind}`;
    intro.querySelector("span:last-child").textContent = text;
  }
}

function setFooter(text, isError = false) {
  $("#last-updated").textContent = text;
  $("#status-footer").classList.toggle("error", isError);
  $("#status-footer").classList.toggle("online", !isError);
}

function selected() {
  return snapshot?.models?.find((model) => model.name === selectedModel) || null;
}

function gpuFor(model) {
  return snapshot?.gpus?.find((gpu) => gpu.pci_slot === model.gpu_pci_slot) || null;
}

function readiness(model) {
  return model.file_readiness || { status: "unknown", available: null, detail: "File availability was not reported by this server." };
}

function modelStatus(model) {
  if (model.state === "loading") return { label: "Loading", tone: "warn" };
  if (model.state === "draining") return { label: "Draining", tone: "warn" };
  if (model.state === "ready" || model.loaded) return { label: "Loaded", tone: "ready" };
  const files = readiness(model);
  if (files.available === false) return { label: "File problem", tone: "error" };
  return { label: files.available === true ? "Files available" : "Configured", tone: "idle" };
}

function canStart(model) {
  return !!model.loaded || readiness(model).available !== false;
}

function preserveSelection(models) {
  const preferred = selectedModel || sessionStorage.getItem(SELECTED_MODEL_KEY);
  selectedModel = preferred && models.some((model) => model.name === preferred)
    ? preferred
    : modelSelectionInitialized ? null : models[0]?.name || null;
  modelSelectionInitialized = true;
  if (selectedModel) sessionStorage.setItem(SELECTED_MODEL_KEY, selectedModel);
  else sessionStorage.removeItem(SELECTED_MODEL_KEY);
}

function button(label, className, onClick) {
  const node = document.createElement("button");
  node.type = "button";
  node.className = className;
  node.textContent = label;
  node.addEventListener("click", onClick);
  return node;
}

function fmtMb(mb) {
  if (mb == null) return "?";
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
}

function libraryMessage(host, text, cls = "library-note") {
  const p = document.createElement("p");
  p.className = cls;
  p.textContent = text;
  host.replaceChildren(p);
}

function createDetails(model, gpu) {
  const details = document.createElement("details");
  details.className = "advanced-details";
  details.open = openDetailModels.has(model.name);
  details.addEventListener("toggle", () => {
    if (details.open) openDetailModels.add(model.name);
    else openDetailModels.delete(model.name);
  });
  // Do not let opening this disclosure also select and re-render its card.
  details.addEventListener("click", (event) => event.stopPropagation());
  details.addEventListener("keydown", (event) => event.stopPropagation());
  const summary = document.createElement("summary");
  summary.textContent = "Advanced details";
  details.appendChild(summary);
  const rows = [
    ["GPU", gpu?.name || "Not assigned"],
    ["PCI slot", model.gpu_pci_slot || "Not set"],
    ["SYCL device", gpu ? `level_zero:${gpu.sycl_index}` : "Not set"],
    ["GPU memory", fmtGiB(gpu?.vram_mb)],
    ["Context", fmtCtx(model.ctx)],
    ["KV cache", `${model.cache_type_k || "default"} / ${model.cache_type_v || "default"}`],
    ["Port", model.port || "Not set"],
    ["Model path", model.path || "Not set"],
  ];
  const list = document.createElement("dl");
  for (const [term, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    list.append(dt, dd);
  }
  details.appendChild(list);
  const configure = document.createElement("a");
  configure.className = "advanced-edit";
  configure.href = `/chat?model=${encodeURIComponent(model.name)}&settings=1`;
  configure.textContent = "Edit launch settings";
  details.appendChild(configure);
  return details;
}

function createModelCard(model) {
  const gpu = gpuFor(model);
  const status = modelStatus(model);
  const card = document.createElement("article");
  card.className = `model-card ${model.name === selectedModel ? "selected" : ""}`;
  card.dataset.modelName = model.name;
  card.tabIndex = 0;
  card.setAttribute("role", "option");
  card.setAttribute("aria-selected", String(model.name === selectedModel));

  const pick = () => {
    selectedModel = model.name;
    sessionStorage.setItem(SELECTED_MODEL_KEY, model.name);
    render();
  };
  card.addEventListener("click", pick);
  card.addEventListener("keydown", (event) => {
    if (event.target !== card) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      pick();
    }
  });

  const body = document.createElement("div");
  body.className = "model-card-main";
  const title = document.createElement("h3");
  title.textContent = displayName(model);
  const meta = document.createElement("p");
  meta.className = "model-meta";
  meta.textContent = [
    model.model_file_mb != null ? `${fmtGiB(model.model_file_mb)} on disk` : "Model size unavailable",
    gpu?.name || "GPU assignment unavailable",
  ].join(" · ");
  body.append(title, meta);
  const fileState = readiness(model);
  if (fileState.available === false) {
    const problem = document.createElement("p");
    problem.className = "model-card-file-warning";
    problem.textContent = `${fileState.detail || "Model files are unavailable."} ${model.loaded ? "This model remains loaded; restore the original file(s) before its next load." : "Inspect the path in Advanced details. Restore the original file(s) or scan a replacement."}`;
    body.appendChild(problem);
  }

  const side = document.createElement("div");
  side.className = "model-card-side";
  const pill = document.createElement("span");
  pill.className = `status-pill ${status.tone}`;
  pill.textContent = status.label;
  side.appendChild(pill);
  const review = button("Review", "choose-button", (event) => {
    event.stopPropagation();
    reviewModel(model.name);
  });
  review.setAttribute("aria-label", `Review model ${displayName(model)}`);
  review.title = `Review ${displayName(model)} and open its selected-model details`;
  side.appendChild(review);
  card.append(body, side, createDetails(model, gpu));
  return card;
}

function renderModels() {
  const list = $("#model-list");
  const models = snapshot?.models || [];
  const fields = ["name", "display_name", "path", "model_file_mb", "gpu_pci_slot", "port", "loaded", "state", "ctx", "cache_type_k", "cache_type_v", "kv_class", "file_readiness"];
  const signature = JSON.stringify([selectedModel, models.map(model => fields.map(key => model[key])), (snapshot?.gpus || []).map(gpu => [gpu.pci_slot, gpu.name, gpu.sycl_index, gpu.vram_mb])]);
  if (signature === renderedModelsSignature) return;
  list.replaceChildren();
  list.setAttribute("aria-busy", "false");
  if (!models.length) {
    const empty = document.createElement("div");
    empty.className = "empty-panel";
    empty.innerHTML = "<div class=\"empty-icon\" aria-hidden=\"true\"></div><h3>No models found</h3><p>Download a model above, or scan for files already on disk.</p>";
    empty.appendChild(button("Scan for models", "secondary", scanModels));
    empty.appendChild(button("Browse model library", "secondary", () => {
      if (location.hash !== "#library-title") history.pushState(null, "", "#library-title");
      updateDashboardView();
      $("#library-title").scrollIntoView({ behavior: "smooth", block: "start" });
    }));
    list.appendChild(empty);
    renderedModelsSignature = signature;
    return;
  }
  for (const model of models) list.appendChild(createModelCard(model));
  renderedModelsSignature = signature;
}

function renderReadiness() {
  const container = $("#readiness-card");
  const model = selected();
  container.replaceChildren();
  if (!model) {
    container.className = "readiness-card empty";
    container.innerHTML = "<div class=\"readiness-icon\" aria-hidden=\"true\"></div><div><h3>Select a model to see its launch settings.</h3><p>We will show its configured GPU, context, and memory capacity before you start.</p></div>";
    return;
  }
  container.className = "readiness-card";
  const gpu = gpuFor(model);
  const heading = document.createElement("div");
  heading.className = "readiness-heading";
  const eyebrow = document.createElement("p");
  eyebrow.className = "eyebrow";
  eyebrow.textContent = "SELECTED MODEL";
  const title = document.createElement("h3");
  title.textContent = displayName(model);
  heading.append(eyebrow, title);

  const metrics = document.createElement("div");
  metrics.className = "readiness-metrics";
  const fit = model.vram_estimate;
  let fitValue = "Unavailable";
  let fitTone = "";
  if (fit && fit.estimated_mb != null) {
    const mb = fit.estimated_mb.toLocaleString();
    if (fit.fit === true && fit.headroom_mb != null) {
      fitValue = `Fits, ${fit.headroom_mb.toLocaleString()} MiB headroom (~${mb} MiB)`;
      fitTone = "ok";
    } else if (fit.fit === false) {
      fitValue = `Does not fit (~${mb} MiB)`;
      fitTone = "warn";
    } else {
      fitValue = `~${mb} MiB · ${fit.detail || "GPU capacity unknown"}`;
    }
  }
  const fileState = readiness(model);
  const values = [
    ["Model size", model.model_file_mb != null ? `${fmtGiB(model.model_file_mb)} on disk` : "Unavailable"],
    ["GPU capacity", gpu ? `${gpu.name} · ${fmtGiB(gpu.vram_mb)}` : "GPU assignment unavailable"],
    ["Memory fit", fitValue],
  ["Estimate basis", fit?.confidence === "estimated_from_file_size" ? "File size + KV and overhead (approximate)" : fit ? "Model metadata and heuristic overhead (approximate)" : "Unavailable"],
    ["Context", fmtCtx(model.ctx)],
    ["Status", modelStatus(model).label],
  ];
  for (const [label, value] of values) {
    const item = document.createElement("div");
    item.className = "metric";
    const term = document.createElement("span");
    term.textContent = label;
    const detail = document.createElement("strong");
    if (label === "Memory fit" && fitTone) detail.classList.add(`tone-${fitTone}`);
    detail.textContent = value;
    item.append(term, detail);
    metrics.appendChild(item);
  }
  const note = document.createElement("p");
  note.className = "readiness-note";
  note.textContent = model.state === "loading" ? "This model is loading. Chat will wait until it is ready."
    : model.state === "draining" ? "This model is stopping. A new message will wait for it to finish."
    : model.loaded ? "This model is loaded now."
    : fileState.available === false ? "Fix the file problem before loading this model."
    : fileState.available === true ? "The first message loads this model. File contents and inference have not been checked."
    : "File availability has not been reported. The first message attempts to load this model.";
  container.append(heading, metrics, note);
  if (fileState.available === false) {
    const problem = document.createElement("p");
    problem.className = "model-card-file-warning";
    problem.textContent = `${fileState.detail || "Model files are unavailable."} ${model.loaded ? "This model remains loaded; restore the original file(s) before its next load." : "Inspect the path in Advanced details. Restore the original file(s) or scan a replacement."}`;
    container.appendChild(problem);
    const recovery = document.createElement("div");
    recovery.className = "toolbar";
    recovery.append(
      button("Find replacement", "secondary", () => showModelsView("#library-title")),
      button("Scan again", "ghost", scanModels),
    );
    container.appendChild(recovery);
  }
  if (fileState.available !== false) container.appendChild(libraryController.compatibilityControl({ name: model.name }));
  container.appendChild(createDetails(model, gpu));
}

function metricRow(label, value, tone) {
  const item = document.createElement("div");
  item.className = "metric";
  const term = document.createElement("span");
  term.textContent = label;
  const detail = document.createElement("strong");
  if (tone) detail.classList.add(`tone-${tone}`);
  detail.textContent = value;
  item.append(term, detail);
  return item;
}

function modelRegistrySummary(models, loadedCount) {
  const known = models.filter((model) => readiness(model).available != null).length;
  const available = models.filter((model) => readiness(model).available === true).length;
  const state = `${models.length} registered · ${loadedCount} loaded`;
  if (!known) return `${state} · file availability unknown`;
  return `${state} · ${available} files available`;
}

function setSystemReadinessAction(container, noteText, action) {
  const area = document.createElement("div");
  area.className = "system-readiness-action";
  const note = document.createElement("p");
  note.textContent = noteText;
  area.appendChild(note);
  if (action) area.appendChild(action);
  container.appendChild(area);
}

// One system-level readiness view over the /admin/status snapshot:
// server reachability, GPU detection, and model file availability,
// with exactly one next action for whatever is blocking first-run.
function renderSystemReadiness() {
  const container = $("#system-readiness");
  if (!container) return;
  container.replaceChildren();
  const models = snapshot?.models || [];
  const gpus = snapshot?.gpus || [];
  const loaded = models.filter((model) => model.loaded);
  const metrics = document.createElement("div");
  metrics.className = "readiness-metrics";

  if (!models.length && !gpus.length) {
    // Status answered but returned no runtime or registry data.
    container.className = "system-readiness pending";
    metrics.append(
      metricRow("Server", "Running", "ok"),
      metricRow("GPU", "No GPU configured", "warn"),
      metricRow("Models", "None found", "warn"),
    );
    container.appendChild(metrics);
    setSystemReadinessAction(
      container,
      "arc-llama is running, but no GPU is configured. Check GPU settings, then check again.",
      button("Check again", "secondary", () => fetchStatus(true)),
    );
    return;
  }

  if (!models.length) {
    container.className = "system-readiness degraded";
    metrics.append(
      metricRow("Server", "Running", "ok"),
      metricRow("GPU", gpus[0]?.name ? `${gpus[0].name} configured` : "GPU configuration unavailable", gpus[0]?.name ? "ok" : "warn"),
      metricRow("Models", "None found", "warn"),
    );
    container.appendChild(metrics);
    setSystemReadinessAction(
      container,
      "No GGUF models were found in your scan folders. Scan again after adding a model file.",
      button("Scan for models", "secondary", scanModels),
    );
    return;
  }

  if (!loaded.length) {
    const model = selected() || models[0];
    container.className = "system-readiness degraded";
    metrics.append(
      metricRow("Server", "Running", "ok"),
      metricRow("GPU", gpus[0]?.name ? `${gpus[0].name} configured` : "GPU configuration unavailable", gpus[0]?.name ? "ok" : "warn"),
      metricRow("Models", modelRegistrySummary(models, 0), null),
    );
    container.appendChild(metrics);
    const chat = document.createElement("a");
    chat.className = "chat-action";
    chat.href = `/chat?model=${encodeURIComponent(model.name)}`;
    chat.textContent = "Start chatting";
    setSystemReadinessAction(
      container,
      readiness(model).available === false
        ? `${displayName(model)} has a file problem. Inspect the path in Advanced details, restore the original file(s), or scan a replacement.`
        : "The first message loads this model.",
      readiness(model).available === false ? null : chat,
    );
    return;
  }

  container.className = "system-readiness ok";
  metrics.append(
    metricRow("Server", "Running", "ok"),
    metricRow("GPU", gpus[0]?.name ? `${gpus[0].name} configured` : "GPU configuration unavailable", gpus[0]?.name ? "ok" : "warn"),
    metricRow("Models", modelRegistrySummary(models, loaded.length), "ok"),
  );
  container.appendChild(metrics);
  const active = loaded.find((model) => model.name === selectedModel) || loaded[0];
  const chat = document.createElement("a");
  chat.className = "chat-action";
  chat.href = `/chat?model=${encodeURIComponent(active.name)}`;
  chat.textContent = "Open chat";
  setSystemReadinessAction(
    container,
    `${displayName(active)} is loaded.`,
    chat,
  );
}

function renderSystemReadinessOffline() {
  const container = $("#system-readiness");
  if (!container) return;
  container.replaceChildren();
  container.className = "system-readiness blocked";
  const metrics = document.createElement("div");
  metrics.className = "readiness-metrics";
  metrics.append(
    metricRow("Server", "Offline", "error"),
    metricRow("GPU", "Unknown", null),
    metricRow("Models", "Unknown", null),
  );
  container.appendChild(metrics);
  setSystemReadinessAction(
    container,
    "Could not reach arc-llama. Make sure the server is running, then check again.",
    button("Check again", "secondary", () => fetchStatus(true)),
  );
}

// The coordinator combines status and library jobs without starting any work.
function renderNextStep() {
  const host = $("#next-step");
  if (!host?.dataset) return;
  let title, detail, label, action, href;
  const models = snapshot?.models || [];
  const model = selected();
  const loaded = models.some(item => item.loaded);
  const jobs = [...(libraryJobs || [])].sort((a, b) => (b.started_at || 0) - (a.started_at || 0));
  const active = jobs.find(job => ["queued", "downloading", "registering"].includes(job.status));
  const latest = jobs[0];
  const findModels = () => showModelsView("#library-title");
  if (!statusOnline) {
    title = statusError ? "Reconnect to Arc Llama" : "Checking your setup";
    detail = statusError || "Waiting for server status.";
    if (statusError) { label = "Check again"; action = () => fetchStatus(true); }
  } else if (!snapshot?.gpus?.length && !loaded) {
    title = "Check your GPU setup";
    detail = "No GPU is configured. Run arc-llama doctor to check drivers and permissions, then review System details.";
    label = "Open System details";
    action = () => { location.hash = "system"; updateDashboardView(); };
  } else if (active) {
    title = active.status === "registering" ? "Adding your downloaded model" : "Your model download is in progress";
    const pct = active.bytes_total ? `${Math.min(99, Math.round((active.bytes_done || 0) / active.bytes_total * 100))}%` : "size unknown";
    detail = `${active.repo} · ${active.file?.split("/").pop() || "GGUF"} · ${active.status === "downloading" ? pct : active.status}. You can keep using your existing models.`;
    label = "View download"; action = findModels;
  } else if (latest?.status === "done" && !reviewedDownloads.has(latest.id)) {
    const names = Array.isArray(latest.registered) ? latest.registered : [];
    const exact = names.find(name => models.some(item => item.name === name));
    title = "Review your downloaded model";
    detail = "The download is complete. Review its files and launch settings; memory fit is an estimate, and inference has not been verified.";
    if (exact) { label = `Review ${displayName(models.find(item => item.name === exact))}`; action = () => reviewModel(exact); }
    else if (!names.length) { detail += " Registration details are unavailable; choose the model from your library."; label = "View your models"; action = () => showModelsView("#choose-title"); }
    else { detail += " Registration is not visible yet."; label = "Refresh to review"; action = async () => { await fetchStatus(true); await libraryController.poll({ refreshStatus: false }); }; }
  } else if (!models.length && latest?.status === "error") {
    title = "Your download needs attention";
    detail = "The download did not finish. Retry the same file; technical details and other choices are in the download list.";
    label = "Retry download"; action = event => libraryController.retry(latest, event.currentTarget);
  } else if (!models.length) {
    title = "Get your first model";
    detail = libraryJobsUnavailable ? "Download status is unavailable. Check the library again, or scan for a GGUF already on disk." : "Download a model from Hugging Face, or scan for one already on disk. Downloads are added to your library automatically.";
    label = "Find a model"; action = findModels;
  } else if (model && !canStart(model)) {
    title = "Restore or replace this model's files";
    detail = `${readiness(model).detail || "The model files are unavailable."} Review the path in Advanced details, restore the files, or find a replacement.`;
    label = "Review model files"; action = () => showModelsView("#readiness-title");
  } else if (model) {
    title = `Open chat with ${displayName(model)}`;
    detail = model.loaded ? "This model is loaded." : "Review its files and launch settings. The first message loads the model; estimated memory fit does not verify inference.";
    label = "Open chat"; href = `/chat?model=${encodeURIComponent(model.name)}`;
  } else {
    title = "Choose a model to review";
    detail = "Select a model from your library to review its files and launch settings.";
    label = "Choose a model"; action = () => showModelsView("#choose-title");
  }
  // Preserve the action node (and keyboard focus) when polling changes nothing.
  const signature = JSON.stringify([title, detail, label, href, latest?.id, active?.id, selectedModel]);
  if (host.dataset.signature === signature) return;
  const focused = document.activeElement?.id === "next-step-action";
  host.dataset.signature = signature;
  host.replaceChildren();
  const heading = document.createElement("h2"); heading.id = "next-step-title"; heading.textContent = title;
  const copy = document.createElement("p"); copy.textContent = detail;
  host.append(heading, copy);
  if (label) {
    const node = href ? document.createElement("a") : button(label, "secondary", action);
    node.id = "next-step-action";
    if (href) { node.href = href; node.textContent = label; node.className = "primary-cta"; }
    host.appendChild(node);
    if (focused) node.focus({ preventScroll: true });
  }
}

function renderStart() {
  const link = $("#chat-primary");
  const copy = $("#start-copy");
  const model = selected();
  const reviewSection = $(".readiness-section");
  if (reviewSection) reviewSection.hidden = !model && !$("#model-review-status")?.textContent;
  if (!statusOnline || !model) {
    link.removeAttribute("href");
    link.classList.add("is-disabled");
    link.setAttribute("aria-disabled", "true");
    link.textContent = "Start chatting";
    copy.textContent = statusOnline ? "Select a model to continue." : "Reconnect to the server to continue.";
    return;
  }
  if (canStart(model)) link.href = `/chat?model=${encodeURIComponent(model.name)}`;
  else link.removeAttribute("href");
  link.classList.toggle("is-disabled", !canStart(model));
  link.setAttribute("aria-disabled", String(!canStart(model)));
  link.textContent = `Start chatting with ${displayName(model)}`;
  if (model.loaded) copy.textContent = "This model is loaded.";
  else if (!canStart(model)) copy.textContent = `${readiness(model).detail || "Model files are unavailable."} Inspect the path in Advanced details, restore the original file(s), or scan a replacement.`;
  else copy.textContent = "The first message loads the model.";
}

function updateDashboardView() {
  const dashboard = $(".dashboard");
  if (!dashboard) return;
  const system = location.hash === "#system";
  dashboard.dataset.dashboardView = system ? "system" : "models";
  for (const link of document.querySelectorAll("[data-view-link]")) {
    const active = link.dataset.viewLink === (system ? "system" : "models");
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  }
  document.title = system ? "System: arc-llama" : "Models: arc-llama";
  const title = $("#page-title");
  const description = title?.parentElement?.querySelector("p:not(.eyebrow)");
  if (title) title.textContent = system ? "System" : "Models";
  if (description) description.textContent = system
    ? "Server, GPU, plugins, measurements, and frontend integration."
    : "Find a model, choose it from your library, and open chat.";
}

function showModelsView(targetSelector = "#choose-title") {
  if (location.hash) history.pushState(null, "", `${location.pathname}${location.search}`);
  updateDashboardView();
  const target = $(targetSelector);
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "start" });
    target.focus({ preventScroll: true });
  }
}

function setReviewMessage(message = "", { focus = false } = {}) {
  const notice = $("#model-review-status");
  if (!notice) return;
  notice.textContent = message;
  notice.hidden = !message;
  const reviewSection = $(".readiness-section");
  if (reviewSection) reviewSection.hidden = !selected() && !message;
  if (message && focus) {
    notice.scrollIntoView({ behavior: "smooth", block: "center" });
    notice.focus({ preventScroll: true });
  }
}

async function reviewModel(modelName) {
  const refreshed = await fetchStatus(true);
  if (!refreshed) {
    setReviewMessage("Could not refresh the model list. Use Refresh and try reviewing again.", { focus: true });
    return false;
  }
  const model = snapshot?.models?.find((item) => item.name === modelName);
  if (!model) {
    setReviewMessage("This model is no longer registered. Refresh the model list and choose a currently registered model.", { focus: true });
    return false;
  }
  setReviewMessage();
  selectedModel = modelName;
  for (const job of libraryJobs || []) {
    if (job.status === "done" && job.registered?.includes(modelName)) reviewedDownloads.add(job.id);
  }
  sessionStorage.setItem(SELECTED_MODEL_KEY, modelName);
  render();
  showModelsView("#readiness-title");
  return true;
}

const measurementsController = window.ArcDashboard.createMeasurementsController({
  document,
  request: (...args) => fetch(...args),
  authHeaders,
});
function fetchMeasurements() { return measurementsController.poll(); }

const libraryController = window.ArcDashboard.createLibraryController({
  document,
  request: (...args) => fetch(...args),
  authHeaders,
  makeButton: button,
  getDisplayName: displayName,
  formatMb: fmtMb,
  writeMessage: libraryMessage,
  getSnapshot: () => snapshot,
  fetchStatus,
  reviewModel,
  showModelsView,
  onJobsChange: (jobs) => {
    libraryJobs = jobs;
    libraryJobsUnavailable = jobs === null;
    renderNextStep();
  },
});
function pollLibraryJobs(options) { return libraryController.poll(options); }
function loadLibraryDisk() { return libraryController.loadDisk(); }

const pluginsController = window.ArcDashboard.createPluginsController({
  document,
  request: (...args) => fetch(...args),
  authHeaders,
  makeButton: button,
  navigate: (route) => { window.location.href = route; },
});
function fetchPlugins() { return pluginsController.poll(); }
function renderPlugins() { return pluginsController.render(); }

const integrationController = window.ArcDashboard.createIntegrationController({
  document,
  request: (...args) => fetch(...args),
  authHeaders,
  browserNavigator: navigator,
});
function openFrontendDialog() { return integrationController.open(); }

function render() {
  libraryController.invalidateCompatibility();
  const loadedCount = snapshot?.models?.filter((model) => model.loaded).length || 0;
  $("#stop-all").disabled = loadedCount === 0;
  setServerState("online", loadedCount ? `${loadedCount} model${loadedCount === 1 ? "" : "s"} loaded` : "Server ready");
  renderModels();
  renderReadiness();
  renderSystemReadiness();
  renderStart();
  renderNextStep();
  renderPlugins();
}

function modelListFocusTarget() {
  const active = document.activeElement;
  const card = active?.closest?.(".model-card");
  if (!card) return null;
  let control = "card";
  if (active.matches(".choose-button")) control = "review";
  else if (active.matches("details summary")) control = "details";
  else if (active.matches(".advanced-edit")) control = "advanced";
  return { modelName: card.dataset.modelName, control };
}

function restoreModelListFocus(target) {
  if (!target) return;
  const card = [...document.querySelectorAll(".model-card")]
    .find((item) => item.dataset.modelName === target.modelName);
  const node = !card ? null : target.control === "review" ? card.querySelector(".choose-button")
    : target.control === "details" ? card.querySelector("details summary")
      : target.control === "advanced" ? card.querySelector(".advanced-edit") : card;
  node?.focus({ preventScroll: true });
}

function fetchStatus(force = false) {
  if (statusRequest) {
    if (!force) return statusRequest;
    // A mutation/manual refresh gets one fresh read after the older read ends.
    if (!queuedStatusRequest) queuedStatusRequest = statusRequest.then(() => fetchStatus()).finally(() => { queuedStatusRequest = null; });
    return queuedStatusRequest;
  }
  statusRequest = fetchStatusRequest().finally(() => { statusRequest = null; });
  return statusRequest;
}

async function fetchStatusRequest() {
  try {
    const response = await fetch("/admin/status", { headers: authHeaders() });
    if (!response.ok) {
      throw new Error(response.status === 401 ? "Local admin access is unavailable" : `Server returned ${response.status}`);
    }
    snapshot = await response.json();
    statusOnline = true;
    statusError = "";
    preserveSelection(snapshot.models || []);
    const focusedModelControl = modelListFocusTarget();
    render();
    restoreModelListFocus(focusedModelControl);
    setFooter(`Updated ${new Date().toLocaleTimeString()}`);
    return true;
  } catch (error) {
    statusOnline = false;
    statusError = error.message === "Local admin access is unavailable"
      ? "Local admin access is unavailable. Open the dashboard on the server machine and check again."
      : "Could not reach Arc Llama. Check that the server is running, then try again.";
    renderStart();
    renderNextStep();
    setServerState("error", "Could not reach arc-llama");
    $("#model-list").setAttribute("aria-busy", "false");
    if (!snapshot) {
      renderedModelsSignature = null;
      $("#model-list").innerHTML = "<div class=\"empty-panel error-panel\"><div class=\"empty-icon\" aria-hidden=\"true\">!</div><h3>Could not reach arc-llama</h3><p>Make sure the server is running, then refresh this page.</p></div>";
    }
    renderSystemReadinessOffline();
    setFooter(error.message, true);
    return false;
  }
}

const SCAN_LABEL_IDLE = "Scan for models";
const SCAN_LABEL_BUSY = "Scanning…";

function setScanStatus(text, { busy = false, showProgress = false } = {}) {
  const container = $("#scan-status");
  const progress = $("#scan-progress");
  if (!container) return;
  container.classList.toggle("busy", busy);
  if (text == null) {
    container.hidden = true;
    $("#scan-status-text").textContent = "";
    if (progress) progress.hidden = true;
    return;
  }
  container.hidden = false;
  $("#scan-status-text").textContent = text;
  if (progress) progress.hidden = !showProgress;
}

async function scanModels() {
  if (scanning) return;
  scanning = true;
  // The toolbar button anchors the visible loading state; secondary "scan"
  // buttons in empty/readiness panels re-render on fetchStatus and need no
  // per-instance state beyond the shared status region below.
  const buttonNode = $("#scan");
  buttonNode.disabled = true;
  buttonNode.setAttribute("aria-busy", "true");
  buttonNode.textContent = SCAN_LABEL_BUSY;
  setScanStatus("Scanning your folders for models. This may take a moment.", { busy: true, showProgress: true });
  try {
    const response = await fetch("/admin/scan", { method: "POST", headers: authHeaders() });
    if (!response.ok) throw new Error(`Scan failed (${response.status})`);
    const result = await response.json();
    const added = result.added?.length || 0;
    setScanStatus(added ? `Scan finished: found ${result.found}, added ${added} model${added === 1 ? "" : "s"}.` : "Scan finished: no new models found.");
    setFooter(added ? `Found ${result.found}; added ${added} model${added === 1 ? "" : "s"}` : `Found ${result.found}; no new models`);
    await fetchStatus(true);
  } catch (error) {
    setScanStatus(`Scan failed: ${error.message}. Check your scan folders and try again.`, { busy: true });
    setFooter(`${error.message}. Check your scan folders and try again.`, true);
  } finally {
    scanning = false;
    buttonNode.disabled = false;
    buttonNode.removeAttribute("aria-busy");
    buttonNode.textContent = SCAN_LABEL_IDLE;
  }
}

async function stopAll() {
  const loadedCount = snapshot?.models?.filter((model) => model.loaded).length || 0;
  if (!loadedCount || !confirm(`Stop ${loadedCount} loaded model${loadedCount === 1 ? "" : "s"}?`)) return;
  const buttonNode = $("#stop-all");
  buttonNode.disabled = true;
  try {
    const response = await fetch("/admin/stop-all", { method: "POST", headers: authHeaders() });
    if (!response.ok) throw new Error(`Stop failed (${response.status})`);
    setFooter("All models stopped.");
    await fetchStatus(true);
  } catch (error) {
    setFooter(error.message, true);
  } finally {
    buttonNode.disabled = false;
  }
}

// Plugin catalog and frontend integration guidance use focused controllers.

// Frontend connection guidance is owned by its focused controller.

// Timers wait for their request to finish; hidden tabs use a quieter cadence.
let statusTimer = null;
let statusPolling = false;
function scheduleStatusPoll() {
  clearTimeout(statusTimer);
  if (!statusPolling) return;
  statusTimer = setTimeout(async () => { await fetchStatus(); scheduleStatusPoll(); }, document.hidden ? 30000 : 5000);
}
function startStatusPolling() {
  if (statusPolling) return;
  statusPolling = true;
  document.addEventListener("visibilitychange", async () => {
    clearTimeout(statusTimer);
    if (!document.hidden) await fetchStatus();
    scheduleStatusPoll();
  });
  scheduleStatusPoll();
}

// Dashboard event binding and bootstrap.
integrationController.bind();

// In-page System view; ordinary modified/middle clicks retain native link behavior.
for (const link of document.querySelectorAll("[data-view-link]")) {
  link.addEventListener("click", (event) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const hash = link.dataset.viewLink === "system" ? "#system" : "";
    if (location.hash !== hash) history.pushState(null, "", hash || "/");
    updateDashboardView();
    window.scrollTo(0, 0);
  });
}
window.addEventListener("popstate", updateDashboardView);
window.addEventListener("hashchange", updateDashboardView);
updateDashboardView();

$("#refresh").addEventListener("click", () => fetchStatus(true));
$("#scan").addEventListener("click", scanModels);
$("#stop-all").addEventListener("click", stopAll);
$("#theme-toggle").addEventListener("click", () => { const next = document.documentElement.dataset.theme === "light" ? "dark" : "light"; localStorage.setItem(THEME_KEY, next); applyTheme(next); });

(async () => {
  libraryController.bind();
  await initAdminToken();
  await fetchStatus(true);
  await pluginsController.start();
  pluginsController.bind();
  await fetchMeasurements();
  libraryController.start();
  measurementsController.start();
  startStatusPolling();
})();
