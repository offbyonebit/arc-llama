// arc-llama dashboard: a first-run friendly view over the local model registry.

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
let selectedModel = null;
let adminToken = null;
let fetching = false;
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

function modelStatus(model) {
  return model.loaded ? { label: "Loaded", tone: "ready" } : { label: "Ready on first message", tone: "idle" };
}

function preserveSelection(models) {
  const preferred = selectedModel || sessionStorage.getItem(SELECTED_MODEL_KEY);
  selectedModel = preferred && models.some((model) => model.name === preferred)
    ? preferred
    : models[0]?.name || null;
  if (selectedModel) sessionStorage.setItem(SELECTED_MODEL_KEY, selectedModel);
}

function button(label, className, onClick) {
  const node = document.createElement("button");
  node.type = "button";
  node.className = className;
  node.textContent = label;
  node.addEventListener("click", onClick);
  return node;
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

  const side = document.createElement("div");
  side.className = "model-card-side";
  const pill = document.createElement("span");
  pill.className = `status-pill ${status.tone}`;
  pill.textContent = status.label;
  side.appendChild(pill);
  const choose = button(model.name === selectedModel ? "Selected" : "Choose", "choose-button", (event) => {
    event.stopPropagation();
    pick();
  });
  choose.disabled = model.name === selectedModel;
  side.appendChild(choose);
  card.append(body, side, createDetails(model, gpu));
  return card;
}

function renderModels() {
  const list = $("#model-list");
  list.replaceChildren();
  list.setAttribute("aria-busy", "false");
  const models = snapshot?.models || [];
  if (!models.length) {
    const empty = document.createElement("div");
    empty.className = "empty-panel";
    empty.innerHTML = "<div class=\"empty-icon\" aria-hidden=\"true\"></div><h3>No models found</h3><p>Add a GGUF file to a scan folder, then scan again.</p>";
    empty.appendChild(button("Scan for models", "secondary", scanModels));
    list.appendChild(empty);
    return;
  }
  for (const model of models) list.appendChild(createModelCard(model));
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
  const values = [
    ["Model size", model.model_file_mb != null ? `${fmtGiB(model.model_file_mb)} on disk` : "Unavailable"],
    ["GPU capacity", gpu ? `${gpu.name} · ${fmtGiB(gpu.vram_mb)}` : "GPU assignment unavailable"],
    ["Context", fmtCtx(model.ctx)],
    ["Status", model.loaded ? "Loaded and ready" : "Loads when you send a message"],
  ];
  for (const [label, value] of values) {
    const item = document.createElement("div");
    item.className = "metric";
    const term = document.createElement("span");
    term.textContent = label;
    const detail = document.createElement("strong");
    detail.textContent = value;
    item.append(term, detail);
    metrics.appendChild(item);
  }
  const note = document.createElement("p");
  note.className = "readiness-note";
  note.textContent = model.loaded
    ? "This model is loaded now. You can start a conversation immediately."
    : "The first message starts the model. arc-llama will stop another local model first when your memory policy requires it.";
  container.append(heading, metrics, note, createDetails(model, gpu));
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
// server reachability, runtime/GPU detection, and model availability,
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
      metricRow("Runtime", "No GPU detected", "warn"),
      metricRow("Models", "None found", "warn"),
    );
    container.appendChild(metrics);
    setSystemReadinessAction(
      container,
      "arc-llama is running but did not detect an Arc GPU. Check the llama.cpp runtime, then check again.",
      button("Check again", "secondary", () => fetchStatus(true)),
    );
    return;
  }

  if (!models.length) {
    container.className = "system-readiness degraded";
    metrics.append(
      metricRow("Server", "Running", "ok"),
      metricRow("Runtime", gpus[0]?.name ? `${gpus[0].name} detected` : "GPU detection unavailable", gpus[0]?.name ? "ok" : "warn"),
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
      metricRow("Runtime", gpus[0]?.name ? `${gpus[0].name} detected` : "GPU detection unavailable", gpus[0]?.name ? "ok" : "warn"),
      metricRow("Models", `${models.length} available · none loaded`, null),
    );
    container.appendChild(metrics);
    const chat = document.createElement("a");
    chat.className = "chat-action";
    chat.href = `/chat?model=${encodeURIComponent(model.name)}`;
    chat.textContent = "Start chatting";
    setSystemReadinessAction(
      container,
      "Everything is set up. Models start on their first message.",
      chat,
    );
    return;
  }

  container.className = "system-readiness ok";
  metrics.append(
    metricRow("Server", "Running", "ok"),
    metricRow("Runtime", gpus[0]?.name ? `${gpus[0].name} detected` : "GPU detection unavailable", gpus[0]?.name ? "ok" : "warn"),
    metricRow("Models", `${models.length} available · ${loaded.length} loaded`, "ok"),
  );
  container.appendChild(metrics);
  const active = loaded.find((model) => model.name === selectedModel) || loaded[0];
  const chat = document.createElement("a");
  chat.className = "chat-action";
  chat.href = `/chat?model=${encodeURIComponent(active.name)}`;
  chat.textContent = "Open chat";
  setSystemReadinessAction(
    container,
    `${displayName(active)} is loaded and ready.`,
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
    metricRow("Runtime", "Unknown", null),
    metricRow("Models", "Unknown", null),
  );
  container.appendChild(metrics);
  setSystemReadinessAction(
    container,
    "Could not reach arc-llama. Make sure the server is running, then check again.",
    button("Check again", "secondary", () => fetchStatus(true)),
  );
}

function renderStart() {
  const link = $("#chat-primary");
  const copy = $("#start-copy");
  const model = selected();
  if (!model) {
    link.removeAttribute("href");
    link.classList.add("is-disabled");
    link.setAttribute("aria-disabled", "true");
    link.textContent = "Start chatting";
    copy.textContent = "Select a model to continue.";
    return;
  }
  link.href = `/chat?model=${encodeURIComponent(model.name)}`;
  link.classList.remove("is-disabled");
  link.setAttribute("aria-disabled", "false");
  link.textContent = `Start chatting with ${displayName(model)}`;
  copy.textContent = model.loaded ? "This model is ready now." : "The model will load when you send your first message.";
}

function render() {
  const loadedCount = snapshot?.models?.filter((model) => model.loaded).length || 0;
  $("#stop-all").disabled = loadedCount === 0;
  setServerState("online", loadedCount ? `${loadedCount} model${loadedCount === 1 ? "" : "s"} loaded` : "Server ready");
  renderModels();
  renderReadiness();
  renderSystemReadiness();
  renderStart();
}

async function fetchStatus(force = false) {
  if (fetching && !force) return;
  fetching = true;
  try {
    const response = await fetch("/admin/status", { headers: authHeaders() });
    if (!response.ok) {
      throw new Error(response.status === 401 ? "Local admin access is unavailable" : `Server returned ${response.status}`);
    }
    snapshot = await response.json();
    preserveSelection(snapshot.models || []);
    render();
    setFooter(`Updated ${new Date().toLocaleTimeString()}`);
  } catch (error) {
    setServerState("error", "Could not reach arc-llama");
    $("#model-list").setAttribute("aria-busy", "false");
    if (!snapshot) {
      $("#model-list").innerHTML = "<div class=\"empty-panel error-panel\"><div class=\"empty-icon\" aria-hidden=\"true\">!</div><h3>Could not reach arc-llama</h3><p>Make sure the server is running, then refresh this page.</p></div>";
    }
    renderSystemReadinessOffline();
    setFooter(error.message, true);
  } finally {
    fetching = false;
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

// Connect-a-frontend guided panel. The endpoint returns only locally computed
// discovery data (base URL from the configured host/port, loopback Ollama
// reachability, registered upstreams); the panel is copy-only and never
// mutates anything on the user's side or ours.
let integrationLoaded = false;
const frontendTabs = [
  ["#tab-openwebui", "#panel-openwebui"],
  ["#tab-ollama", "#panel-ollama"],
  ["#tab-generic", "#panel-generic"],
];
let activeFrontendTab = "openwebui";

function isWindows() {
  return navigator.platform && /win/i.test(navigator.platform);
}

async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (_) {
    // Older or non-secure contexts may lack the async clipboard API.
    try {
      const helper = document.createElement("textarea");
      helper.value = text;
      helper.setAttribute("readonly", "true");
      helper.style.position = "fixed";
      helper.style.opacity = "0";
      document.body.appendChild(helper);
      helper.select();
      const ok = document.execCommand("copy");
      helper.remove();
      return ok;
    } catch (_) {
      return false;
    }
  }
}

function markCopied(buttonNode) {
  const original = buttonNode.textContent;
  buttonNode.textContent = "Copied";
  buttonNode.classList.add("copied");
  setTimeout(() => {
    buttonNode.textContent = original;
    buttonNode.classList.remove("copied");
  }, 1500);
}

function bindCopyButton(id, getText) {
  const node = $(id);
  if (!node) return;
  node.addEventListener("click", async () => {
    if (await copyToClipboard(getText())) markCopied(node);
  });
}

function selectFrontendTab(kind) {
  activeFrontendTab = kind;
  for (const [tabId, panelId] of frontendTabs) {
    const tab = $(tabId);
    const panel = $(panelId);
    const active = tabId.includes(kind);
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    panel.toggleAttribute("hidden", !active);
  }
}

function renderOllamaStatus(ollama) {
  const status = $("#ollama-status");
  const registered = $("#ollama-registered-note");
  if (!ollama?.reachable) {
    status.textContent =
      "No Ollama found on this machine (checked its default address). " +
      "The command below still works later; run it once Ollama is installed and running.";
    status.classList.add("unreachable");
  } else {
    status.textContent = `Ollama detected${ollama.version ? ` (version ${ollama.version})` : ""}.`;
    status.classList.remove("unreachable");
  }
  registered.hidden = !ollama?.already_registered;
}

function renderIntegration(data) {
  const openwebuiUrl = $("#openwebui-url");
  const genericUrl = $("#generic-url");
  const genericCurl = $("#generic-curl");
  const ollamaCommand = $("#ollama-command");
  const portHint = $("#frontend-port-hint");
  if (!openwebuiUrl || !data) return;
  openwebuiUrl.textContent = data.base_url;
  genericUrl.textContent = data.base_url;
  genericCurl.textContent = isWindows() ? data.curl_example_windows : data.curl_example;
  ollamaCommand.textContent = data.ollama?.upstream_add_command || "";
  portHint.textContent = `port ${data.server?.port}, at /v1/chat/completions and /v1/models`;
  const lanNote = $("#openwebui-lan-note");
  if (data.lan_note) {
    lanNote.textContent = data.lan_note;
    lanNote.hidden = false;
  } else {
    lanNote.hidden = true;
  }
  const apiKeyNote = $("#openwebui-key-note");
  if (data.api_key_guidance) {
    apiKeyNote.textContent = `API Key: ${data.api_key_guidance}`;
  }
  renderOllamaStatus(data.ollama);
  integrationLoaded = true;
}

async function loadIntegration(force = false) {
  if (integrationLoaded && !force) return;
  try {
    const response = await fetch("/admin/integration", { headers: authHeaders() });
    if (!response.ok) throw new Error(`status ${response.status}`);
    renderIntegration(await response.json());
  } catch (_) {
    renderIntegration(null);
  }
}

function openFrontendDialog() {
  $("#frontend-dialog").showModal();
  loadIntegration();
}

$("#connect-frontend").addEventListener("click", openFrontendDialog);
$("#frontend-close").addEventListener("click", () => $("#frontend-dialog").close());
for (const [tabId] of frontendTabs) {
  $(tabId).addEventListener("click", () => selectFrontendTab(tabId.split("-")[1]));
}
bindCopyButton("#copy-openwebui-url", () => $("#openwebui-url")?.textContent);
bindCopyButton("#copy-ollama-command", () => $("#ollama-command")?.textContent);
bindCopyButton("#copy-generic-url", () => $("#generic-url")?.textContent);
bindCopyButton("#copy-generic-curl", () => $("#generic-curl")?.textContent);

$("#refresh").addEventListener("click", () => fetchStatus(true));
$("#scan").addEventListener("click", scanModels);
$("#stop-all").addEventListener("click", stopAll);
$("#theme-toggle").addEventListener("click", () => { const next = document.documentElement.dataset.theme === "light" ? "dark" : "light"; localStorage.setItem(THEME_KEY, next); applyTheme(next); });

(async () => {
  await initAdminToken();
  await fetchStatus(true);
  setInterval(fetchStatus, 5000);
})();
