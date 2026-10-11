const $ = (sel) => document.querySelector(sel);
const THEME_KEY = "arc-llama-theme";
function applyTheme(theme) {
  const dark = theme !== "light";
  if (document.documentElement) document.documentElement.dataset.theme = dark ? "dark" : "light";
  const toggle = $("#theme-toggle");
  if (toggle) { toggle.textContent = dark ? "Light theme" : "Dark theme"; toggle.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme"); }
  if (document.querySelectorAll) document.querySelectorAll(".brand-logo").forEach((logo) => { logo.src = logo.dataset[dark ? "dark" : "light"] || logo.src; });
}
applyTheme(typeof localStorage === "undefined" ? "dark" : (localStorage.getItem(THEME_KEY) || "dark"));
// Side panels open beneath the header so its history/settings buttons stay reachable.
(() => {
  const header = $("header");
  if (!header) return;
  const publish = () => document.documentElement?.style?.setProperty?.("--header-h", `${header.offsetHeight}px`);
  publish();
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(publish).observe(header);
})();
const chatLog = $("#chat-log");
const emptyState = $("#empty-state");
const modelSelect = $("#model-select");
const modelStatus = $("#model-status");
const statusText = $("#status-text");
const input = $("#message-input");
const sendButton = $("#send-button");
const inputWrap = $("#input-wrap");
const commandPalette = $("#command-palette");
const attachButton = $("#attach-button");
const pdfInput = $("#pdf-input");
const attachmentStrip = $("#attachment-strip");
const pluginTools = $("#plugin-tools");
const pluginToolsToggle = $("#plugin-tools-toggle");
const pluginToolsMenu = $("#plugin-tools-menu");
const pluginToolsCount = $("#plugin-tools-count");
const visionModeChip = $("#vision-mode-chip");

function setPluginToolsOpen(open) {
  if (!pluginToolsMenu || !pluginToolsToggle) return;
  pluginToolsMenu.hidden = !open;
  pluginToolsToggle.setAttribute("aria-expanded", String(open));
  pluginTools.classList.toggle("open", open);
}

pluginToolsToggle?.addEventListener("click", () => {
  setPluginToolsOpen(pluginToolsMenu.hidden);
});
document.addEventListener("click", (event) => {
  if (pluginTools && !pluginTools.contains(event.target)) setPluginToolsOpen(false);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    setPluginToolsOpen(false);
    // Escape also closes the settings panel, so keyboard users can leave
    // every overlay without reaching for the mouse.
    if (settingsPanel && settingsPanel.classList.contains("open")) {
      settingsPanel.classList.remove("open");
      settingsToggle?.focus();
    }
    if (historyPanel && historyPanel.classList.contains("open")) {
      historyPanel.classList.remove("open");
    }
    if (activeComposerAction && !generatingImage && document.activeElement === input) setComposerAction(null);
  }
});

let models = [];
let selectedModel = null;
let loadingModel = null;
let generating = false;
let sendingMessage = false;
let statusPoller = null;
let adminToken = null;
const MIN_VISION_LOADER_MS = 850;

// The user-selected image-generation tool. When set, the main composer is in
// "image mode": typed text is the image prompt and pressing Enter submits it
// to the plugin's generation endpoint instead of the chat model.
let activeComposerAction = null;
let generatingImage = false;

function setComposerAction(action) {
  activeComposerAction = action;
  if (!action) {
    inputWrap.classList.remove("vision-mode");
    input.dataset.imageMode = "false";
    input.placeholder = "Message arc-llama…";
    input.setAttribute("aria-label", "Message arc-llama");
    if (visionModeChip) visionModeChip.hidden = true;
    attachButton.disabled = false;
    return;
  }
  const mode = action.composer?.mode || "text";
  inputWrap.classList.toggle("vision-mode", mode === "text");
  input.dataset.composerMode = mode;
  input.dataset.imageMode = String(mode === "text" && action.composer?.result === "image");
  input.placeholder = action.composer?.placeholder || (mode === "attachments"
    ? `Add files for ${action.label || "this tool"}…`
    : `Enter a prompt for ${action.label || "this tool"}…`);
  input.setAttribute("aria-label", mode === "attachments" ? "Tool input and attachments" : "Tool prompt");
  if (visionModeChip) {
    visionModeChip.hidden = false;
    visionModeChip.setAttribute("aria-live", "polite");
    const label = visionModeChip.querySelector("#vision-mode-label");
    if (label) label.textContent = action.label ? `${action.label} mode` : "Tool mode";
  }
  // Attachments are chat-context extras; they have no meaning for prompts
  // sent to the diffusion companion, so park them while the mode is on.
  clearAttachments();
  attachButton.disabled = mode !== "attachments";
  hideCommandPalette();
  setPluginToolsOpen(false);
  input.focus();
}

function isComposerActionActive() {
  return activeComposerAction !== null && !generating;
}

const visionModeChipCancel = $("#vision-mode-chip-cancel");
if (visionModeChipCancel) {
  visionModeChipCancel.addEventListener("click", () => setComposerAction(null));
}

function selectComposerAction(action) {
  if (generating || generatingImage) return;
  // Toggle behavior: picking the same tool twice turns image mode off.
  if (activeComposerAction && activeComposerAction.id === action.id) {
    setComposerAction(null);
    return;
  }
  setComposerAction(action);
}

async function sendComposerAction() {
  const action = activeComposerAction;
  const prompt = input.value.trim();
  if (!action || generating || generatingImage) return;
  if (!prompt && !hasReadyAttachments()) return;
  if (!action.route) {
    showError("Selected tool has no route.");
    setComposerAction(null);
    return;
  }

  const attachmentText = buildAttachmentText();
  const payload = { prompt };
  if (attachmentText) payload.attachments = attachmentText;
  clearAttachments();
  input.value = "";
  input.style.height = "auto";
  createMessage("user", prompt || "Attached input");
  hideCommandPalette();

  generatingImage = true;
  sendButton.disabled = true;
  inputWrap.classList.add("generating");

  const wrapper = document.createElement("div");
  wrapper.className = "message assistant vision-generation";
  wrapper.setAttribute("role", "status");
  wrapper.setAttribute("aria-live", "polite");
  wrapper.innerHTML = `<div class="vision-loader" aria-label="Running ${escapeHtml(action.label || "tool")}"><div class="vision-loader-glow"></div><div class="vision-loader-core"></div><span>${escapeHtml(action.label || "Tool")} is working…</span></div>`;
  chatLog.appendChild(wrapper); chatLog.scrollTop = chatLog.scrollHeight;
  const loaderStartedAt = performance.now();
  try {
    const responsePromise = fetch(action.route, {method: action.method || "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    const response = await responsePromise;
    const data = await response.json();
    const remaining = MIN_VISION_LOADER_MS - (performance.now() - loaderStartedAt);
    if (remaining > 0) await new Promise((resolve) => setTimeout(resolve, remaining));
    if (!response.ok) throw new Error(data.detail || "Tool action failed");
    const resultType = action.composer?.result || "image";
    if (resultType === "image") {
      const image = data.data?.[0]?.b64_json;
      if (!image) throw new Error("Tool returned no image");
      wrapper.className = "message assistant vision-generation complete";
      wrapper.replaceChildren();
      const img = document.createElement("img"); img.src = `data:image/png;base64,${image}`; img.alt = prompt; img.style.maxWidth = "100%"; wrapper.appendChild(img);
    } else {
      wrapper.className = "message assistant tool-generation complete";
      wrapper.textContent = data.output || data.text || data.message || JSON.stringify(data);
    }
    chatLog.appendChild(wrapper); chatLog.scrollTop = chatLog.scrollHeight;
    setComposerAction(null);
  } catch (error) {
    wrapper.className = "message assistant vision-generation failed";
    wrapper.innerHTML = `<div class="vision-generation-error"><strong>Tool action failed</strong><span>${escapeHtml(error.message)}</span></div>`;
    setComposerAction(null);
  }
  finally {
    generatingImage = false;
    sendButton.disabled = false;
    inputWrap.classList.remove("generating");
    input.focus();
  }
}

async function loadPluginActions() {
  const host = $("#plugin-actions");
  if (!host) return;
  try {
    const r = await fetch("/admin/ui/layout", {headers: authHeaders()});
    if (!r.ok) return;
    const data = await r.json();
    const ids = [...(data.layout?.chat || []), ...(data.layout?.plugins || []), ...(data.layout?.toolbar || [])];
    const hidden = new Set(data.hidden || []);
    host.replaceChildren();
    let visibleCount = 0;
    for (const action of data.actions || []) {
      if (!ids.includes(action.id) || hidden.has(action.id)) continue;
      visibleCount += 1;
      const button = document.createElement("button");
      button.type = "button"; button.className = "plugin-action-button"; button.setAttribute("role", "menuitem");
      const icon = document.createElement("img");
      icon.className = "plugin-action-icon";
      icon.src = action.icon === "image" ? "/assets/arc-llama-vision.png?v=ui-0.21" : "/assets/arc-llama-tools.png?v=ui-0.21";
      icon.alt = ""; icon.setAttribute("aria-hidden", "true");
      const copy = document.createElement("span"); copy.className = "plugin-action-copy";
      const label = document.createElement("strong"); label.textContent = action.label || action.id;
      const detail = document.createElement("small"); detail.textContent = action.description || "Plugin action";
      copy.append(label, detail); button.append(icon, copy);
      button.addEventListener("click", () => {
        setPluginToolsOpen(false);
        if (action.composer?.mode) selectComposerAction(action);
        else if (action.route) window.location.href = action.route;
      });
      host.appendChild(button);
    }
    if (pluginToolsCount) pluginToolsCount.textContent = String(visibleCount);
    if (pluginToolsToggle) pluginToolsToggle.disabled = visibleCount === 0;
    if (!visibleCount) setPluginToolsOpen(false);
  } catch (_) { /* plugin actions are optional */ }
}

async function initAdminToken() {
  try {
    const r = await fetch("/admin/session-token");
    if (r.ok) {
      const data = await r.json();
      adminToken = data.admin_token || null;
    }
  } catch (e) {
    // Non-loopback deployment or offline -- admin calls will 401/403 until
    // the user supplies a token some other way.
  }
}

function authHeaders(extra = {}) {
  return adminToken ? { ...extra, Authorization: `Bearer ${adminToken}` } : extra;
}

// Remote (LAN) browsers cannot fetch the admin token, so the inference API
// needs an API key there. Same-origin /v1 and /api calls get the admin token
// or the stored key automatically; a 401 asks for a key once and retries.
const API_KEY_STORAGE = "arc-llama-api-key";
function storedApiKey() {
  try { return localStorage.getItem(API_KEY_STORAGE) || null; } catch (_) { return null; }
}
function rememberApiKey(key) {
  try { if (key) localStorage.setItem(API_KEY_STORAGE, key); else localStorage.removeItem(API_KEY_STORAGE); } catch (_) {}
}
function isClientApiPath(input) {
  const url = typeof input === "string" ? input : (input && input.url) || "";
  return url.startsWith("/v1/") || url.startsWith("/api/");
}
function withClientAuth(init = {}) {
  const credential = adminToken || storedApiKey();
  const headers = new Headers(init.headers || {});
  if (credential && !headers.has("Authorization")) headers.set("Authorization", `Bearer ${credential}`);
  return { ...init, headers };
}
// A small modal; the UI never uses blocking browser dialogs.
function askForApiKey() {
  return new Promise((resolve) => {
    const dialog = document.createElement("dialog");
    dialog.className = "api-key-dialog";
    const form = document.createElement("form");
    form.method = "dialog";
    const label = document.createElement("label");
    label.textContent = "This server needs an API key for remote access.";
    const field = document.createElement("input");
    field.type = "password";
    field.autocomplete = "off";
    field.placeholder = "arc_...";
    label.appendChild(field);
    const save = document.createElement("button");
    save.type = "submit";
    save.value = "save";
    save.textContent = "Use key";
    const cancel = document.createElement("button");
    cancel.type = "submit";
    cancel.value = "cancel";
    cancel.className = "ghost";
    cancel.textContent = "Cancel";
    form.append(label, save, cancel);
    dialog.appendChild(form);
    document.body.appendChild(dialog);
    dialog.addEventListener("close", () => {
      const value = dialog.returnValue === "save" ? field.value.trim() : "";
      dialog.remove();
      resolve(value || null);
    });
    dialog.showModal();
    field.focus();
  });
}

// Called once from init(): wraps fetch so every existing /v1 call site gets
// credentials without being rewritten.
function installClientAuth() {
  const nativeFetch = window.fetch.bind(window);
  let apiKeyPrompted = false;
  window.fetch = async (input, init = {}) => {
    if (!isClientApiPath(input)) return nativeFetch(input, init);
    const response = await nativeFetch(input, withClientAuth(init));
    if (response.status !== 401 || adminToken || apiKeyPrompted) return response;
    apiKeyPrompted = true;
    const key = await askForApiKey();
    if (!key) return response;
    rememberApiKey(key.trim());
    const retry = await nativeFetch(input, withClientAuth(init));
    if (retry.status === 401) {
      rememberApiKey(null);
      apiKeyPrompted = false;
    }
    return retry;
  };
}
let lastUsage = null;
let streamStartTime = null;
let streamTokenCount = 0;
const conversation = [];
// Abort handle for the generation in flight; the send button stops it.
let activeAbort = null;
// DOM of the latest exchange, for Regenerate / Edit.
let lastUserDiv = null;
let lastAssistantDiv = null;
// Measured prompt size from the last reply: {tokens, length}. Context
// estimates start from it and only approximate what was added since.
let budgetBase = null;
const PRESETS_KEY = "arc-llama-presets";

const ctxMeter   = $("#ctx-meter");
const ctxBarFill = $("#ctx-bar-fill");
const ctxLabelL  = $("#ctx-label-left");
const ctxLabelTps = $("#ctx-label-tps");
const ctxLabelR  = $("#ctx-label-right");
const settingsToggle = $("#settings-toggle");
const settingsPanel  = $("#settings-panel");
const sModelName     = $("#s-model-name");
const sFields        = $("#s-fields");
const sFeedback      = $("#s-feedback");

const historyToggle  = $("#history-toggle");
const historyPanel   = $("#history-panel");
const hNew           = $("#h-new");
const hList          = $("#h-list");
const hExport        = $("#h-export");
const hImport        = $("#h-import");
const hImportInput   = $("#h-import-input");
const hFolder        = $("#h-folder");
const hNewFolder     = $("#h-new-folder");

let currentChatId = null;

const renderingComponent = window.ArcChat.createRendering({
  chatLog,
  emptyState,
  autoScroll,
  shouldAutoScroll,
});
const {
  attachCopyButtons,
  createMessage,
  escapeHtml,
  parseThinking,
  renderMarkdown,
  renderThinking,
  showError,
  appendChunk,
} = renderingComponent;

const attachmentsComponent = window.ArcChat.createAttachments({
  attachmentStrip,
  attachButton,
  pdfInput,
  inputWrap,
  authHeaders,
  request: (...args) => fetch(...args),
  escapeHtml,
  modelCanSeeImages,
  refreshBudget,
  get selectedModel() { return selectedModel; },
});
const {
  addAttachment,
  buildAttachmentText,
  clearAttachments,
  hasProcessingAttachments,
  hasReadyAttachments,
  isImageFile,
  isPdfFile,
  isTextFile,
  processAttachment,
  removeAttachment,
  renderAttachments,
  getItems: getAttachments,
} = attachmentsComponent;

const settingsComponent = window.ArcChat.createSettings({
  $,
  authHeaders,
  fetchStatus,
  request: (...args) => fetch(...args),
  storage: { getItem: (...args) => localStorage.getItem(...args), setItem: (...args) => localStorage.setItem(...args) },
  refreshBudget,
  sFeedback,
  sFields,
  sModelName,
  settingsToggle,
  settingsPanel,
  get models() { return models; },
  get budgetBase() { return budgetBase; },
  set budgetBase(value) { budgetBase = value; },
  get selectedModel() { return selectedModel; },
});
const {
  applySettings,
  loadPresets,
  presetFor,
  renderPresetPanel,
  renderSettingsPanel,
  savePreset,
  vramFitText,
} = settingsComponent;

const historyComponent = window.ArcChat.createHistory({
  request: (...args) => fetch(...args),
  storage: { getItem: (...args) => localStorage.getItem(...args), setItem: (...args) => localStorage.setItem(...args) },
  autoScroll,
  chatLog,
  conversation,
  createMessage,
  emptyState,
  estimateTokens,
  escapeHtml,
  hFolder,
  hImportInput,
  hList,
  hNew,
  hNewFolder,
  hExport,
  hImport,
  historyPanel,
  historyToggle,
  input,
  parseStructuredFailure,
  renderMarkdown,
  renderThinking,
  shouldAutoScroll,
  showError,
  updateCtxMeter,
  updatePickerStatus,
  get budgetBase() { return budgetBase; },
  set budgetBase(value) { budgetBase = value; },
  get currentChatId() { return currentChatId; },
  set currentChatId(value) { currentChatId = value; },
  get lastAssistantDiv() { return lastAssistantDiv; },
  set lastAssistantDiv(value) { lastAssistantDiv = value; },
  get lastUserDiv() { return lastUserDiv; },
  set lastUserDiv(value) { lastUserDiv = value; },
  get loadingModel() { return loadingModel; },
  set loadingModel(value) { loadingModel = value; },
  modelSelect,
  get models() { return models; },
  get selectedModel() { return selectedModel; },
  set selectedModel(value) { selectedModel = value; },
  newChat,
});
const {
  apiRequest,
  buildMoveSelect,
  createFolder,
  deleteChat,
  ensureServerChat,
  exportChats,
  formatRelativeTime,
  generateId,
  importChatsFromFile,
  loadChat,
  loadFolders,
  loadChats,
  moveChat,
  populateFolderSelects,
  renderHistoryPanel,
  saveChats,
  serverAppendMessages,
  serverChatToLocal,
  syncChatsFromServer,
  truncateTitle,
  getChats,
  getCurrentFolder,
} = historyComponent;


function openSettingsFromLink() {
  if (new URLSearchParams(window.location.search).get("settings") !== "1") return;
  settingsPanel.classList.add("open");
  settingsToggle.classList.add("open");
  renderSettingsPanel();
}



async function saveCurrentChat() {
  if (conversation.length === 0) return;
  const firstUser = conversation.find(m => m.role === "user");
  const title = truncateTitle(firstUser ? firstUser.content : "New chat");
  const now = Date.now();
  if (!currentChatId) {
    currentChatId = generateId();
  }

  const chatDoc = {
    id: currentChatId,
    title,
    model: selectedModel,
    messages: conversation.map(m => ({ role: m.role, content: m.content })),
    createdAt: now,
    updatedAt: now,
  };

  // Server is the source of truth; persist the full chat there first.
  try {
    await apiRequest(`/v1/chats/${encodeURIComponent(currentChatId)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title,
        messages: chatDoc.messages,
      }),
    });
  } catch (e) {
    console.warn("Could not save chat to server:", e.message);
  }

  // Update the local cache to match.
  const chats = loadChats();
  const idx = chats.findIndex(c => c.id === currentChatId);
  if (idx >= 0) {
    chats[idx] = { ...chats[idx], ...chatDoc };
  } else {
    chats.unshift(chatDoc);
  }
  chats.sort((a, b) => b.updatedAt - a.updatedAt);
  while (chats.length > historyComponent.maxHistory()) chats.pop();
  saveChats(chats);
}

async function newChat() {
  conversation.length = 0;
  budgetBase = null;
  lastUserDiv = null;
  lastAssistantDiv = null;
  currentChatId = null;
  chatLog.innerHTML = "";
  chatLog.appendChild(emptyState);
  emptyState.style.display = "";
  input.value = "";
  input.style.height = "auto";
  historyPanel.classList.remove("open");
  historyToggle.classList.remove("open");
  updateCtxMeter(0, models.find(m => m.id === selectedModel)?.ctx || 131072);
  ctxMeter.classList.remove("visible");
  ctxLabelTps.textContent = "";
  await ensureServerChat("New chat", getCurrentFolder());
  input.focus();
}

function estimateTokens() {
  const chars = conversation.reduce((n, m) => n + (m.content || "").length, 0);
  return Math.round(chars / 4);
}

function updateCtxMeter(tokens, ctx) {
  if (!ctx) return;
  const pct = Math.min(100, tokens / ctx * 100);
  ctxBarFill.style.width = pct.toFixed(1) + "%";
  ctxBarFill.className = "ctx-bar-fill" + (pct >= 90 ? " critical" : pct >= 70 ? " warn" : "");
  ctxLabelL.textContent = `${tokens.toLocaleString()} / ${ctx.toLocaleString()} tokens`;
  ctxLabelR.textContent = pct.toFixed(1) + "%";
  ctxMeter.classList.add("visible");
}

async function fetchModels() {
  try {
    const r = await fetch("/v1/models");
    if (!r.ok) throw new Error(`status ${r.status}`);
    const data = await r.json();
    const local = (data.data || []).filter(m => m.object === "model" && m.owned_by !== "arc-llama-alias");
    for (const m of local) {
      m.ctx = m.ctx ?? m.metadata?.ctx;
      m.capabilities = m.metadata?.capabilities || [];
    }
    models = local;
    renderModelPicker();
  } catch (e) {
    showError("Could not fetch models: " + e.message);
  }
}

function renderModelPicker() {
  const requested = new URLSearchParams(window.location.search).get("model");
  const current = requested || selectedModel || sessionStorage.getItem("arc-llama-selected-model") || modelSelect.value;
  modelSelect.innerHTML = "";
  if (models.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "No models available";
    opt.disabled = true;
    opt.selected = true;
    modelSelect.appendChild(opt);
    selectedModel = null;
    updateStatus("unavailable");
    return;
  }
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.id;
    modelSelect.appendChild(opt);
  }
  if (current && models.some(m => m.id === current)) {
    modelSelect.value = current;
    selectedModel = current;
  } else {
    selectedModel = models[0].id;
    modelSelect.value = selectedModel;
  }
  if (selectedModel) sessionStorage.setItem("arc-llama-selected-model", selectedModel);
  openSettingsFromLink();
}

async function fetchStatus() {
  try {
    const r = await fetch("/admin/status", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    const modelMap = new Map((data.models || []).map(m => [m.name, m]));
    const waiting = document.querySelector(".load-wait-card .content");
    if (waiting && loadingModel) {
      const target = modelMap.get(loadingModel);
      const draining = (data.models || []).some(m => m.state === "draining");
      waiting.textContent = draining
        ? "Waiting for active responses to finish before switching models…"
        : target?.state === "loading"
          ? "Loading the model and waiting for the runtime to become ready…"
          : "Waiting for model readiness…";
    }
    models = models.map(m => {
      const s = modelMap.get(m.id);
      if (s) {
        m.loaded        = s.loaded;
        m.ctx           = s.ctx           ?? m.ctx;
        m.cache_type_k  = s.cache_type_k  ?? m.cache_type_k;
        m.cache_type_v  = s.cache_type_v  ?? m.cache_type_v;
        m.kv_class      = s.kv_class      ?? m.kv_class;
        m.vram_estimate = s.vram_estimate || null;
      }
      return m;
    });
    updatePickerStatus();
    if (settingsPanel.classList.contains("open")) renderSettingsPanel();
  } catch (e) {
    // silent: the chat endpoint will surface real errors
  }
}

function updatePickerStatus() {
  refreshBudget();
  const m = models.find(m => m.id === selectedModel);
  if (!m) {
    updateStatus("unavailable");
    return;
  }
  if (loadingModel === selectedModel) {
    updateStatus("loading");
  } else if (m.loaded) {
    updateStatus("ready");
  } else {
    updateStatus("idle");
  }
}

function updateStatus(state) {
  modelStatus.className = "model-status " + state;
  statusText.textContent = state;
}

modelSelect.addEventListener("change", () => {
  selectedModel = modelSelect.value;
  sessionStorage.setItem("arc-llama-selected-model", selectedModel);
  loadingModel = null;
  updatePickerStatus();
  if (settingsPanel.classList.contains("open")) renderSettingsPanel();
  restoreDraft();
});


function showStartupFailure(failure, { onRetry = null, retryLabel = "Retry" } = {}) {
  const category = failure?.category;
  const message = failure?.message || "";
  const action = failure?.action || "";
  const diagId = failure?.diagnostics_id || "";
  const details = failure?.details || null;
  if (!category || !message) return showError(message || "Model load failed.");

  const { div, content } = createMessage("error", "");
  div.classList.add("error-card", "load-failure-card");
  div.dataset.failureCategory = category;
  if (diagId) div.dataset.diagnosticsId = diagId;
  div.querySelector(".role").textContent = "Load failed";

  content.textContent = message;

  if (action) {
    const actionEl = document.createElement("p");
    actionEl.className = "load-failure-action";
    const key = document.createElement("span");
    key.className = "fail-key";
    key.textContent = "What to do: ";
    actionEl.append(key, action);
    content.appendChild(actionEl);
  }

  const meta = document.createElement("div");
  meta.className = "load-failure-meta";
  if (diagId) {
    const idEl = document.createElement("span");
    idEl.className = "load-failure-id";
    idEl.textContent = `diagnostics ${diagId}`;
    meta.appendChild(idEl);
  }
  const retry = document.createElement("button");
  retry.type = "button";
  retry.className = "load-failure-retry";
  retry.textContent = retryLabel;
  retry.disabled = !onRetry;
  retry.addEventListener("click", async () => {
    if (!onRetry || generating) return;
    retry.disabled = true;
    try {
      const result = await onRetry();
      if (result !== false) div.remove();
    } catch (error) {
      showError(error.message || "Retry failed.");
    } finally {
      retry.disabled = !onRetry;
    }
  });
  meta.appendChild(retry);
  if (details && Object.keys(details).length) {
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "load-failure-details-toggle";
    toggle.setAttribute("aria-expanded", "false");
    const detailsRegion = document.createElement("div");
    detailsRegion.className = "load-failure-details";
    detailsRegion.hidden = true;
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(details, null, 2);
    detailsRegion.appendChild(pre);
    toggle.textContent = "Show diagnostics";
    toggle.addEventListener("click", () => {
      const open = detailsRegion.toggleAttribute("hidden");
      toggle.setAttribute("aria-expanded", String(!open));
      toggle.textContent = open ? "Show diagnostics" : "Hide diagnostics";
    });
    meta.appendChild(toggle);
    content.appendChild(detailsRegion);
  }
  content.appendChild(meta);
}

// Parse the HTTP error body of /admin/load or /v1/chat/completions into the
// structured failure shape ({category, message, action, diagnostics_id,
// details}) when the server sent one; otherwise return null.
async function parseStructuredFailure(response) {
  try {
    const body = await response.clone().json();
    const err = body && body.error;
    if (err && typeof err === "object" && typeof err.message === "string") return err;
  } catch (_) { /* not JSON */ }
  try {
    const body = await response.clone().json();
    const text = typeof body?.detail === "string" ? body.detail : null;
    if (text) return { category: "http_error", message: text, action: "", diagnostics_id: "" };
  } catch (_) { /* not JSON */ }
  return null;
}

async function ensureModelLoaded() {
  const m = models.find(m => m.id === selectedModel);
  if (!m) throw new Error("No model selected");
  if (m.loaded || (m.owned_by && m.owned_by.startsWith("upstream:"))) return;
  loadingModel = selectedModel;
  updateStatus("loading");
  try {
    const r = await fetch(`/admin/load/${encodeURIComponent(selectedModel)}`, {
      method: "POST",
      headers: authHeaders(),
    });
    if (!r.ok) {
      const structured = await parseStructuredFailure(r);
      if (structured) {
        const err = new Error(structured.message);
        err.structured = structured;
        throw err;
      }
      const t = await r.text();
      throw new Error(`Load failed: ${r.status} ${t}`);
    }
    m.loaded = true;
  } finally {
    loadingModel = null;
    updatePickerStatus();
  }
}

// ------------------------------------------------------------------
// Presets, context budget, and request shaping
// ------------------------------------------------------------------


function modelCanSeeImages(modelId) {
  const m = models.find(x => x.id === modelId);
  return !!(m && (m.capabilities || []).includes("vision"));
}

// OpenAI messages for a transcript: images become image_url parts, local
// bookkeeping fields are dropped, and the model's preset system prompt leads.
function toApiMessages(entries, modelId = selectedModel) {
  const out = [];
  const preset = presetFor(modelId);
  if (preset.system) out.push({ role: "system", content: preset.system });
  for (const m of entries) {
    if (m.images && m.images.length) {
      out.push({
        role: m.role,
        content: [
          { type: "text", text: m.content || "" },
          ...m.images.map(url => ({ type: "image_url", image_url: { url } })),
        ],
      });
    } else {
      out.push({ role: m.role, content: m.content || "" });
    }
  }
  return out;
}

// Rough cost of an image in prompt tokens; projectors vary (256..1500).
const IMAGE_TOKEN_ESTIMATE = 768;

function estimateBudget(draft = "") {
  const pending = getAttachments().filter(a => !a.error);
  const chars = (entries) => entries.reduce((n, m) => n + (m.content || "").length, 0);
  const images = (entries) => entries.reduce((n, m) => n + (m.images ? m.images.length : 0), 0);
  let tokens;
  let exact = false;
  if (budgetBase && budgetBase.length <= conversation.length) {
    const added = conversation.slice(budgetBase.length);
    tokens = budgetBase.tokens + Math.round(chars(added) / 4) + images(added) * IMAGE_TOKEN_ESTIMATE;
    exact = added.length === 0;
  } else {
    tokens = Math.round(chars(conversation) / 4) + images(conversation) * IMAGE_TOKEN_ESTIMATE;
  }
  const preset = presetFor(selectedModel);
  if (!budgetBase && preset.system) tokens += Math.round(preset.system.length / 4);
  const draftChars = draft.length + pending.reduce((n, a) => n + (a.text || "").length, 0);
  tokens += Math.round(draftChars / 4) + pending.filter(a => a.image).length * IMAGE_TOKEN_ESTIMATE;
  return { tokens, exact: exact && !draftChars && !pending.length };
}

function refreshBudget() {
  const m = models.find(x => x.id === selectedModel);
  if (!m || !m.ctx) return;
  const { tokens, exact } = estimateBudget(input.value);
  if (!tokens) { ctxMeter.classList.remove("visible"); return; }
  updateCtxMeter(tokens, m.ctx);
  if (!exact) ctxLabelL.textContent = `~${tokens.toLocaleString()} / ${m.ctx.toLocaleString()} tokens`;
  const over = tokens > m.ctx;
  ctxMeter.classList.toggle("over-budget", over);
  ctxMeter.title = over
    ? "This conversation is larger than the model's context. Older turns will be cut off; start a new chat or run /compact."
    : "";
}

function setStopMode(on) {
  sendButton.classList.toggle("stop", on);
  sendButton.setAttribute("aria-label", on ? "Stop generating" : "Send");
  sendButton.title = on ? "Stop generating" : "";
  if (on) sendButton.disabled = false;
}

function clearTurnActions() {
  for (const node of document.querySelectorAll(".turn-actions")) node.remove();
}

function turnButton(label, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "turn-action";
  b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}

// Regenerate and Edit act on the latest exchange only, which keeps the
// transcript, the DOM, and the stored chat trivially in step.
function renderTurnActions() {
  clearTurnActions();
  if (lastAssistantDiv && conversation.length && conversation[conversation.length - 1].role === "assistant") {
    const bar = document.createElement("div");
    bar.className = "turn-actions";
    bar.appendChild(turnButton("Regenerate", regenerateLast));
    lastAssistantDiv.appendChild(bar);
  }
  if (lastUserDiv) {
    const bar = document.createElement("div");
    bar.className = "turn-actions";
    bar.appendChild(turnButton("Edit", editLastUser));
    lastUserDiv.appendChild(bar);
  }
}

async function regenerateLast() {
  if (generating || sendingMessage) return;
  if (!conversation.length || conversation[conversation.length - 1].role !== "assistant") return;
  conversation.pop();
  if (lastAssistantDiv) lastAssistantDiv.remove();
  lastAssistantDiv = null;
  budgetBase = null;
  clearTurnActions();
  await saveCurrentChat();
  await runSend();
}

async function editLastUser() {
  if (generating || sendingMessage) return;
  let i = conversation.length - 1;
  while (i >= 0 && conversation[i].role !== "user") i--;
  if (i < 0) return;
  const entry = conversation[i];
  conversation.length = i;
  if (lastAssistantDiv) lastAssistantDiv.remove();
  if (lastUserDiv) lastUserDiv.remove();
  lastAssistantDiv = null;
  lastUserDiv = null;
  budgetBase = null;
  clearTurnActions();
  input.value = entry.content || "";
  input.dispatchEvent(new Event("input"));
  input.focus();
  if (entry.images && entry.images.length) {
    showError("Images from the edited message were not kept; attach them again if needed.");
  }
  await saveCurrentChat();
}

function renderUserImages(div, images) {
  if (!images || !images.length) return;
  const strip = document.createElement("div");
  strip.className = "message-images";
  for (const url of images) {
    const img = document.createElement("img");
    img.src = url;
    img.alt = "Attached image";
    img.loading = "lazy";
    strip.appendChild(img);
  }
  div.appendChild(strip);
}

async function sendMessage() {
  if (isComposerActionActive()) { sendComposerAction(); return; }
  if (generating || sendingMessage || !selectedModel) return;
  const text = input.value.trim();
  if (!text && !hasReadyAttachments()) return;
  if (hasProcessingAttachments()) {
    showError("Please wait for attachments to finish processing.");
    return;
  }

  const images = getAttachments().filter(a => a.image && !a.error && !a.processing);
  if (images.length && !modelCanSeeImages(selectedModel)) {
    showError(`${selectedModel} cannot read images. Pick a vision model or remove the image.`);
    return;
  }
  const attachmentText = buildAttachmentText();
  const fullText = text
    ? attachmentText ? `${text}\n\n${attachmentText}` : text
    : attachmentText;

  input.value = "";
  input.style.height = "auto";
  clearAttachments();
  clearDraft();
  const userEntry = { role: "user", content: fullText };
  if (images.length) userEntry.images = images.map(a => a.image);
  conversation.push(userEntry);
  clearTurnActions();
  const userMsg = createMessage("user", fullText);
  renderUserImages(userMsg.div, userEntry.images);
  lastUserDiv = userMsg.div;
  lastAssistantDiv = null;

  // Reserve this send while persistence awaits, before generation begins.
  sendingMessage = true;
  try {
    await ensureServerChat(fullText, getCurrentFolder());
    serverAppendMessages(currentChatId, [{ role: "user", content: fullText }]);
    await runSend();
  } finally {
    sendingMessage = false;
  }
}

// The generation half of a send, after the user turn has been appended to
// the transcript. A failed startup renders a retry card that re-enters
// here directly, so a retried send never duplicates the user turn.
async function runSend() {
  if (generating || !selectedModel || !conversation.length) return false;
  const retryChatId = currentChatId;
  const retryModel = selectedModel;
  const retryTranscript = conversation;
  const retryLength = conversation.length;
  const retryTurn = () => {
    if (generating || currentChatId !== retryChatId || selectedModel !== retryModel ||
        conversation !== retryTranscript || conversation.length !== retryLength) {
      showError("This retry belongs to an earlier conversation or turn. Send a new message to continue.");
      return false;
    }
    return runSend();
  };
  generating = true;
  sendButton.disabled = true;
  inputWrap.classList.add("generating");

  // Honest waiting stage: if the model is not loaded yet, say so in the log
  // instead of leaving an empty assistant bubble for tens of seconds of
  // cold start. The card is removed as soon as the load resolves one way
  // or another.
  const needsLoad = (() => {
    const m = models.find(m => m.id === selectedModel);
    return !(m && (m.loaded || (m.owned_by && m.owned_by.startsWith("upstream:"))));
  })();
  let loadingCard = null;
  if (needsLoad) {
    loadingCard = createMessage("system", "Starting model, this can take a while on first load…");
    loadingCard.div.classList.add("load-wait-card");
  }
  const streamingDot = $("#streaming-indicator");
  if (streamingDot) streamingDot.style.opacity = "1";

  try {
    await ensureModelLoaded();
  } catch (e) {
    if (loadingCard) loadingCard.div.remove();
    if (e.structured) {
      showStartupFailure(e.structured, { onRetry: retryTurn, retryLabel: "Retry load" });
    } else {
      showError(e.message);
    }
    finishGeneration();
    return;
  }
  if (loadingCard) loadingCard.div.remove();

  const assistantMsg = createMessage("assistant");
  lastAssistantDiv = assistantMsg.div;
  conversation.push({ role: "assistant", content: "", thinking: "" });
  const convoIndex = conversation.length - 1;
  const controller = new AbortController();
  activeAbort = controller;
  setStopMode(true);
  const preset = presetFor(selectedModel);

  let streamRaw = "";
  let lastDisplayedContent = "";
  let currentThinking = "";

  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        model: selectedModel,
        messages: toApiMessages(conversation.slice(0, -1)),
        stream: true,
        stream_options: { include_usage: true },
        ...(preset.temperature != null ? { temperature: preset.temperature } : {}),
      }),
    });
    if (!r.ok) {
      const structured = await parseStructuredFailure(r);
      if (structured) {
        const err = new Error(structured.message);
        err.structured = structured;
        throw err;
      }
      const t = await r.text();
      throw new Error(`${r.status} ${t}`);
    }
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        const chunkText = processSseLine(line);
        if (chunkText != null) {
          if (streamStartTime === null) streamStartTime = Date.now();
          streamTokenCount += Math.max(1, Math.round(chunkText.length / 4));
          streamRaw += chunkText;
          const parsed = parseThinking(streamRaw);
          if (!parsed.hasPartialTag) {
            const newContent = parsed.content.slice(lastDisplayedContent.length);
            lastDisplayedContent = parsed.content;
            currentThinking = parsed.thinking;
            if (newContent) {
              conversation[convoIndex].content = parsed.content;
              appendChunk(assistantMsg.content, newContent);
              if (shouldAutoScroll(chatLog)) autoScroll(chatLog);
            }
            const thinkingBlock = assistantMsg.div.querySelector(".thinking-block");
            const thinkingContent = assistantMsg.div.querySelector(".thinking-content");
            if (thinkingBlock && thinkingContent) {
              const trimmedThinking = currentThinking.trim();
              if (trimmedThinking) {
                thinkingBlock.style.display = "";
                thinkingContent.textContent = trimmedThinking;
              } else {
                thinkingBlock.style.display = "none";
                thinkingContent.textContent = "";
              }
            }
          }
        }
      }
    }
    const finalParsed = parseThinking(streamRaw);
    conversation[convoIndex].content = finalParsed.content;
    conversation[convoIndex].thinking = finalParsed.thinking;
    renderMarkdown(assistantMsg.content, finalParsed.content);
    const m = models.find(m => m.id === selectedModel);
    const completionToks = lastUsage ? lastUsage.completion_tokens : streamTokenCount;
    const totalToks      = lastUsage ? lastUsage.total_tokens      : estimateTokens();
    const elapsed        = streamStartTime ? (Date.now() - streamStartTime) / 1000 : null;
    const tps            = (elapsed && elapsed > 0 && completionToks > 0)
                           ? (completionToks / elapsed).toFixed(1)
                           : null;
    if (tps) ctxLabelTps.textContent = tps + " tok/s";
    updateCtxMeter(totalToks, m?.ctx || 131072);
    budgetBase = lastUsage ? { tokens: lastUsage.total_tokens, length: conversation.length } : null;
    await serverAppendMessages(currentChatId, [{ role: "assistant", content: conversation[convoIndex].content }]);
    await saveCurrentChat();
    renderTurnActions();
  } catch (e) {
    if (e.name === "AbortError") {
      // Stopped by the user: keep whatever arrived, marked as interrupted.
      const partial = parseThinking(streamRaw);
      if (partial.content.trim()) {
        conversation[convoIndex].content = partial.content;
        conversation[convoIndex].thinking = partial.thinking;
        renderMarkdown(assistantMsg.content, partial.content);
        assistantMsg.div.classList.add("stopped");
        const note = document.createElement("div");
        note.className = "stopped-note";
        note.textContent = "Stopped";
        assistantMsg.div.appendChild(note);
        budgetBase = null;
        await serverAppendMessages(currentChatId, [{ role: "assistant", content: partial.content }]);
        await saveCurrentChat();
        renderTurnActions();
      } else {
        assistantMsg.div.remove();
        lastAssistantDiv = null;
        conversation.pop();
        renderTurnActions();
      }
      return;
    }
    assistantMsg.div.remove();
    lastAssistantDiv = null;
    conversation.pop();
    if (e.structured) {
      // The user turn is the last transcript entry again; retry re-enters
      // runSend directly so the turn is not duplicated.
      showStartupFailure(e.structured, { onRetry: retryTurn, retryLabel: "Retry" });
    } else {
      showError("Generation failed: " + e.message);
    }
  } finally {
    if (activeAbort === controller) activeAbort = null;
    lastUsage = null;
    streamStartTime = null;
    streamTokenCount = 0;
    finishGeneration();
  }
}

function processSseLine(line) {
  const trimmed = line.trim();
  if (!trimmed || !trimmed.startsWith("data:")) return null;
  const payload = trimmed.slice(5).trim();
  if (payload === "[DONE]") return null;
  try {
    const obj = JSON.parse(payload);
    if (obj.usage) lastUsage = obj.usage;
    const delta = obj.choices?.[0]?.delta;
    if (!delta) return null;
    let text = "";
    if (delta.reasoning_content) {
      text += "<think>" + delta.reasoning_content + "</think>";
    }
    if (delta.content != null) {
      text += delta.content;
    }
    return text || null;
  } catch (e) {
    return null;
  }
}


function finishGeneration() {
  generating = false;
  setStopMode(false);
  sendButton.disabled = false;
  inputWrap.classList.remove("generating");
  const indicator = $("#streaming-indicator");
  if (indicator) indicator.style.opacity = "0";
  // Collapse any thinking blocks that are open and hide empty ones
  const openThinking = document.querySelectorAll(".thinking-toggle.open");
  for (const t of openThinking) {
    t.classList.remove("open");
    t.nextElementSibling?.classList.remove("open");
  }
  for (const block of document.querySelectorAll(".thinking-block")) {
    const content = block.querySelector(".thinking-content");
    if (content && !content.textContent.trim()) {
      block.style.display = "none";
    }
  }
  input.focus();
}


function shouldAutoScroll(container) {
  if (!container) return true;
  const threshold = 60;
  return container.scrollHeight - container.scrollTop - container.clientHeight <= threshold;
}

function autoScroll(container) {
  if (!container) return;
  container.scrollTop = container.scrollHeight;
}

// ------------------------------------------------------------------
// Slash commands
// ------------------------------------------------------------------

const SLASH_COMMANDS = [
  { name: "help", desc: "Show available slash commands", needsArgs: false },
  { name: "clear", desc: "Clear the current conversation", needsArgs: false },
  { name: "new", desc: "Start a new chat", needsArgs: false },
  { name: "model", desc: "Switch model, e.g. /model <id>", needsArgs: true },
  { name: "compact", desc: "Summarize context, optional: /compact <focus>", needsArgs: false },
];

let paletteSelectedIndex = -1;

function parseSlashCommand(text) {
  const trimmed = text.trim();
  if (!trimmed.startsWith("/")) return null;
  const withoutSlash = trimmed.slice(1);
  const firstSpace = withoutSlash.search(/\s/);
  const command = firstSpace === -1 ? withoutSlash : withoutSlash.slice(0, firstSpace);
  const rest = firstSpace === -1 ? "" : withoutSlash.slice(firstSpace + 1).trim();
  return { command: command.toLowerCase(), rest, raw: trimmed };
}

function getFilteredCommands(prefix) {
  const p = prefix.toLowerCase();
  return SLASH_COMMANDS.filter((c) => c.name.startsWith(p));
}

function hideCommandPalette() {
  commandPalette.classList.remove("open");
  commandPalette.innerHTML = "";
  paletteSelectedIndex = -1;
}

function renderCommandPalette(filter = "") {
  const items = filter === "" ? SLASH_COMMANDS.slice() : getFilteredCommands(filter);
  commandPalette.innerHTML = "";
  if (items.length === 0) {
    hideCommandPalette();
    return;
  }
  paletteSelectedIndex = Math.min(Math.max(paletteSelectedIndex, 0), items.length - 1);
  for (let i = 0; i < items.length; i++) {
    const cmd = items[i];
    const div = document.createElement("div");
    div.className = "command-item" + (i === paletteSelectedIndex ? " selected" : "");
    div.setAttribute("role", "option");
    div.setAttribute("aria-selected", String(i === paletteSelectedIndex));
    div.innerHTML = `
      <span class="cmd-name">/${escapeHtml(cmd.name)}</span>
      <span class="cmd-desc">${escapeHtml(cmd.desc)}</span>
      <span class="cmd-hint">${cmd.needsArgs ? "args" : "enter"}</span>
    `;
    div.addEventListener("click", () => {
      input.value = "/" + cmd.name + " ";
      input.focus();
      hideCommandPalette();
      input.dispatchEvent(new Event("input"));
    });
    div.addEventListener("mouseenter", () => {
      paletteSelectedIndex = i;
      renderCommandPalette(filter);
    });
    commandPalette.appendChild(div);
  }
  commandPalette.classList.add("open");
}

function updateCommandPalette() {
  const text = input.value;
  if (text.startsWith("/") && !text.includes(" ")) {
    const prefix = text.slice(1);
    renderCommandPalette(prefix);
  } else {
    hideCommandPalette();
  }
}

async function executeSlashCommand(rawText) {
  const parsed = parseSlashCommand(rawText);
  if (!parsed) return false;

  const known = SLASH_COMMANDS.find((c) => c.name === parsed.command);
  if (!known) {
    showError(`Unknown command: /${escapeHtml(parsed.command)}. Type /help for available commands.`);
    return true;
  }

  switch (parsed.command) {
    case "help":
      renderHelpMessage();
      break;
    case "clear":
      await clearChat();
      break;
    case "new":
      await newChat();
      break;
    case "model":
      switchModel(parsed.rest);
      break;
    case "compact":
      await compactConversation(parsed.rest);
      break;
  }
  return true;
}

function renderHelpMessage() {
  if (emptyState) emptyState.style.display = "none";
  const div = document.createElement("div");
  div.className = "message system command-hint";
  const roleLabel = document.createElement("div");
  roleLabel.className = "role";
  roleLabel.textContent = "Slash commands";
  div.appendChild(roleLabel);
  const content = document.createElement("div");
  content.className = "content";
  let html = "";
  for (const cmd of SLASH_COMMANDS) {
    html += `<p><code>/${cmd.name}</code> <strong>-</strong> ${escapeHtml(cmd.desc)}</p>`;
  }
  content.innerHTML = html;
  div.appendChild(content);
  chatLog.appendChild(div);
  if (shouldAutoScroll(chatLog)) autoScroll(chatLog);
}

async function clearChat() {
  conversation.length = 0;
  chatLog.innerHTML = "";
  if (emptyState) emptyState.style.display = "";
  updateCtxMeter(0, models.find((m) => m.id === selectedModel)?.ctx || 131072);
  ctxLabelTps.textContent = "";
  await saveCurrentChat();
}

function switchModel(modelId) {
  if (!modelId) {
    showError("Usage: /model <model-id>");
    return;
  }
  const m = models.find((x) => x.id === modelId || x.id.endsWith("/" + modelId) || (x.display_name && x.display_name.toLowerCase() === modelId.toLowerCase()));
  if (!m) {
    showError(`Model not found: ${escapeHtml(modelId)}`);
    return;
  }
  selectedModel = m.id;
  modelSelect.value = m.id;
  updatePickerStatus();
  if (settingsPanel.classList.contains("open")) renderSettingsPanel();
  restoreDraft();
  ensureModelLoaded().catch((e) => {
    if (e.structured) showStartupFailure(e.structured, { onRetry: () => switchModel(m.id), retryLabel: "Retry load" });
    else showError(e.message);
  });
}

async function compactConversation(instruction) {
  if (conversation.length === 0) {
    showError("Nothing to compact.");
    return;
  }
  if (!selectedModel) {
    showError("Select a model first.");
    return;
  }
  try {
    await ensureModelLoaded();
  } catch (e) {
    if (e.structured) showStartupFailure(e.structured, { onRetry: () => compactConversation(instruction), retryLabel: "Retry load" });
    else showError(e.message);
    return;
  }

  const systemPrompt = instruction
    ? `Summarize the following conversation concisely. Focus on: ${instruction}. Preserve key facts, decisions, code snippets, and user intent. Return only the summary.`
    : "Summarize the following conversation concisely. Preserve key facts, decisions, code snippets, and user intent. Return only the summary.";

  const summaryMsg = { role: "system", content: systemPrompt };
  const messages = [summaryMsg, ...conversation];

  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: selectedModel, messages, stream: false }),
    });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(`${r.status} ${t}`);
    }
    const data = await r.json();
    const summary = data.choices?.[0]?.message?.content?.trim();
    if (!summary) {
      throw new Error("Model returned an empty summary.");
    }

    conversation.length = 0;
    conversation.push({ role: "system", content: "Summary of prior conversation:\n\n" + summary });

    chatLog.innerHTML = "";
    if (emptyState) emptyState.style.display = "none";
    const msg = createMessage("system", "Context compacted. Summary:\n\n" + summary);
    if (shouldAutoScroll(chatLog)) autoScroll(chatLog);

    const m = models.find((x) => x.id === selectedModel);
    updateCtxMeter(estimateTokens(), m?.ctx || 131072);
    await saveCurrentChat();
  } catch (e) {
    showError("Compact failed: " + e.message);
  }
}

input.addEventListener("keydown", async (e) => {
  if (commandPalette.classList.contains("open")) {
    const items = commandPalette.querySelectorAll(".command-item");
    if (e.key === "ArrowDown") {
      e.preventDefault();
      paletteSelectedIndex = (paletteSelectedIndex + 1) % items.length;
      renderCommandPalette(input.value.slice(1));
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      paletteSelectedIndex = (paletteSelectedIndex - 1 + items.length) % items.length;
      renderCommandPalette(input.value.slice(1));
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      hideCommandPalette();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const selected = items[paletteSelectedIndex];
      if (selected) selected.click();
      return;
    }
  }

  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    const text = input.value.trim();
    if (await executeSlashCommand(text)) {
      input.value = "";
      input.style.height = "auto";
      hideCommandPalette();
      return;
    }
    if (isComposerActionActive()) {
      await sendComposerAction();
      return;
    }
    sendMessage();
  }
});

sendButton.addEventListener("click", async () => {
  if (generating && activeAbort) {
    activeAbort.abort();
    return;
  }
  const text = input.value.trim();
  if (await executeSlashCommand(text)) {
    input.value = "";
    input.style.height = "auto";
    hideCommandPalette();
    return;
  }
  if (isComposerActionActive()) {
    await sendComposerAction();
    return;
  }
  sendMessage();
});

$("#theme-toggle").addEventListener("click", () => { const next = document.documentElement.dataset.theme === "light" ? "dark" : "light"; localStorage.setItem(THEME_KEY, next); applyTheme(next); });

input.addEventListener("input", () => {
  refreshBudget();
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 96) + "px";
  updateCommandPalette();
  saveDraft();
});

// Draft persistence: an unsent composer draft survives a page refresh, per
// model, so a reload never eats what the user already typed. Cleared on a
// successful send. sessionStorage (not localStorage) keeps it tab-scoped.
const DRAFT_KEY = "arc-llama-draft-";
function saveDraft() {
  if (!selectedModel) return;
  try { sessionStorage.setItem(DRAFT_KEY + selectedModel, input.value); } catch (_) {}
}
function clearDraft() {
  if (!selectedModel) return;
  try { sessionStorage.removeItem(DRAFT_KEY + selectedModel); } catch (_) {}
}
function restoreDraft() {
  if (!selectedModel) return;
  try {
    const draft = sessionStorage.getItem(DRAFT_KEY + selectedModel);
    input.value = draft || "";
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 96) + "px";
  } catch (_) {}
}

(async function init() {
  await initAdminToken();
  installClientAuth();
  renderingComponent.start();
  attachmentsComponent.start();
  settingsComponent.start();
  historyComponent.start();
  await loadPluginActions();
  await fetchModels();
  await fetchStatus();
  await loadFolders();
  await syncChatsFromServer();
  restoreDraft();
  statusPoller = setInterval(fetchStatus, 3000);
})();
