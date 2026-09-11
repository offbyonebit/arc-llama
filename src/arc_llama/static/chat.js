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

let models = [];
let selectedModel = null;
let loadingModel = null;
let generating = false;
let statusPoller = null;
let adminToken = null;

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
let lastUsage = null;
let streamStartTime = null;
let streamTokenCount = 0;
const conversation = [];
let attachments = [];

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

const HISTORY_KEY    = "arc-llama-chats";
const MAX_HISTORY    = 50;
const ALL_FOLDERS    = "__all__";

let currentFolder    = ALL_FOLDERS;
let folders          = [];

// Configure Markdown renderer with syntax highlighting and safe defaults.
if (typeof marked !== "undefined") {
  marked.use({
    gfm: true,
    breaks: false,
    headerIds: false,
    mangle: false,
  });
}
const mdRenderer = typeof marked !== "undefined" ? new marked.Renderer() : {};
Object.assign(mdRenderer, {
  code(code, language) {
    const validLang = language && hljs.getLanguage(language) ? language : "plaintext";
    const highlighted = hljs.highlight(code, { language: validLang }).value;
    const langLabel = validLang === "plaintext" ? "" : `<span class="code-lang">${escapeHtml(validLang)}</span>`;
    return `<div class="code-block-wrapper">${langLabel}<pre><code class="hljs language-${escapeHtml(validLang)}">${highlighted}</code></pre><button class="copy-code-btn" title="Copy" aria-label="Copy code"><svg viewBox="0 0 24 24" width="14" height="14"><path d="M16 1H4a2 2 0 0 0-2 2v14h2V3h12V1zm3 4H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 16H8V7h11v14z"/></svg></button></div>`;
  },
  blockquote(quote) {
    return `<blockquote>${quote}</blockquote>`;
  },
  html(text) {
    return escapeHtml(text);
  },
  link(href, title, text) {
    return ArcMarkdownSafety.link(href, title, text);
  },
  image(href, title, text) {
    return ArcMarkdownSafety.image(href, title, text);
  },
});

function attachCopyButtons(root) {
  for (const btn of root.querySelectorAll(".copy-code-btn")) {
    btn.addEventListener("click", async () => {
      const code = btn.closest(".code-block-wrapper").querySelector("code");
      const text = code ? code.textContent : "";
      try {
        await navigator.clipboard.writeText(text);
        btn.classList.add("copied");
        btn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14"><path d="M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>`;
        setTimeout(() => {
          btn.classList.remove("copied");
          btn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14"><path d="M16 1H4a2 2 0 0 0-2 2v14h2V3h12V1zm3 4H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 16H8V7h11v14z"/></svg>`;
        }, 1500);
      } catch (e) {
        console.warn("Copy failed", e);
      }
    });
  }
}

let currentChatId    = null;
let chatCache        = loadChatsFromStorage();

const KV_TYPES    = ["f16","f32","q8_0","q5_1","q5_0","q4_1","q4_0"];
const KV_CLASSES  = ["default","moe_a3b","qwen3_27b_dense","gemma_swa"];

settingsToggle.addEventListener("click", () => {
  const open = settingsPanel.classList.toggle("open");
  settingsToggle.classList.toggle("open", open);
  if (open) renderSettingsPanel();
});

function openSettingsFromLink() {
  if (new URLSearchParams(window.location.search).get("settings") !== "1") return;
  settingsPanel.classList.add("open");
  settingsToggle.classList.add("open");
  renderSettingsPanel();
}

historyToggle.addEventListener("click", async () => {
  const open = historyPanel.classList.toggle("open");
  historyToggle.classList.toggle("open", open);
  if (open) {
    await loadFolders();
    await syncChatsFromServer();
    renderHistoryPanel();
  }
});

hNew.addEventListener("click", newChat);

if (hFolder) {
  hFolder.addEventListener("change", () => {
    currentFolder = hFolder.value;
    renderHistoryPanel();
  });
}

if (hNewFolder) {
  hNewFolder.addEventListener("click", createFolder);
}

if (hExport) hExport.addEventListener("click", exportChats);
if (hImport) hImport.addEventListener("click", () => hImportInput?.click());
if (hImportInput) hImportInput.addEventListener("change", importChatsFromFile);

function loadChatsFromStorage() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed;
    if (parsed && Array.isArray(parsed.chats)) return parsed.chats;
  } catch (e) {
    // storage may be full / disabled
  }
  return [];
}

function loadChats() {
  return chatCache;
}

function saveChats(chats) {
  chatCache = chats;
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(chats));
  } catch (e) {
    // storage may be full / disabled
  }
}

function serverChatToLocal(data, modelHint) {
  return {
    id: data.id,
    title: data.title || "New chat",
    folder: data.folder || "",
    model: modelHint || null,
    createdAt: Math.round((data.created_at || Date.now() / 1000) * 1000),
    updatedAt: Math.round((data.updated_at || Date.now() / 1000) * 1000),
    messages: (data.messages || []).map(m => ({ role: m.role, content: m.content })),
  };
}

async function apiRequest(path, options = {}) {
  const r = await fetch(path, options);
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`${r.status} ${t}`);
  }
  return r.json();
}

async function syncChatsFromServer() {
  try {
    const data = await apiRequest("/v1/chats");
    const summaries = data.data || [];
    const map = new Map(chatCache.map(c => [c.id, c]));
    // Server is the source of truth for the chat list. Update titles and
    // ordering from summaries; full messages are lazy-loaded by loadChat().
    for (const s of summaries) {
      const existing = map.get(s.id);
      const updatedAt = Math.round((s.updated_at || 0) * 1000);
      if (existing) {
        existing.title = s.title;
        existing.folder = s.folder || "";
        existing.createdAt = Math.round((s.created_at || 0) * 1000);
        existing.updatedAt = updatedAt;
        existing.message_count = s.message_count;
      } else {
        map.set(s.id, {
          id: s.id,
          title: s.title,
          folder: s.folder || "",
          model: null,
          messages: [],
          createdAt: Math.round((s.created_at || 0) * 1000),
          updatedAt: updatedAt,
          message_count: s.message_count,
        });
      }
    }
    const merged = Array.from(map.values()).sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_HISTORY);
    saveChats(merged);
    if (historyPanel.classList.contains("open")) renderHistoryPanel();
  } catch (e) {
    console.warn("Could not sync chats from server:", e.message);
  }
}

async function ensureServerChat(titleHint, folder) {
  if (currentChatId) return;
  const title = truncateTitle(titleHint || "New chat");
  const chatFolder = folder === ALL_FOLDERS ? "" : folder;
  try {
    const body = { title };
    if (chatFolder !== undefined) body.folder = chatFolder;
    const data = await apiRequest("/v1/chats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    currentChatId = data.id;
    const now = Date.now();
    const chats = loadChats();
    chats.unshift(serverChatToLocal(data, selectedModel));
    chats[0].createdAt = now;
    chats[0].updatedAt = now;
    saveChats(chats);
  } catch (e) {
    console.warn("Could not create chat on server:", e.message);
    // Local-only fallback so the UI keeps working offline.
    const id = generateId();
    currentChatId = id;
    const now = Date.now();
    const chats = loadChats();
    chats.unshift({ id, title, folder: chatFolder || "", model: selectedModel, messages: [], createdAt: now, updatedAt: now });
    saveChats(chats);
  }
}

async function serverAppendMessages(chatId, messages, title) {
  if (!chatId) return;
  if ((!messages || messages.length === 0) && !title) return;
  const body = {};
  if (messages && messages.length > 0) body.messages = messages;
  if (title) body.title = title;
  try {
    await apiRequest(`/v1/chats/${encodeURIComponent(chatId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    console.warn("Could not append messages to server:", e.message);
  }
}

function generateId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function truncateTitle(text, max = 60) {
  if (!text) return "New chat";
  const single = text.replace(/\s+/g, " ").trim();
  if (single.length <= max) return single || "New chat";
  return single.slice(0, max - 1).trimEnd() + "…";
}

function formatRelativeTime(ms) {
  const now = Date.now();
  const diff = now - ms;
  const sec = Math.floor(diff / 1000);
  if (sec < 10) return "just now";
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day === 1) return "yesterday";
  if (day < 7) return `${day} days ago`;
  const d = new Date(ms);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
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
  while (chats.length > MAX_HISTORY) chats.pop();
  saveChats(chats);
}

async function newChat() {
  conversation.length = 0;
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
  await ensureServerChat("New chat", currentFolder);
  input.focus();
}

function renderHistoryPanel() {
  const chats = loadChats().filter(c => currentFolder === ALL_FOLDERS || c.folder === currentFolder);
  hList.innerHTML = "";
  if (chats.length === 0) {
    hList.innerHTML = '<div class="h-empty">No chats in this folder yet.</div>';
    return;
  }
  for (const c of chats) {
    const card = document.createElement("div");
    card.className = "h-card";
    card.dataset.id = c.id;
    const ICON_CLOSE = '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>';
    card.innerHTML = `
      <div class="h-card-title">${escapeHtml(c.title)}</div>
      <div class="h-card-meta">
        <span>${escapeHtml(c.model || "unknown")}</span>
        <span>${formatRelativeTime(c.updatedAt)}</span>
      </div>
      <button class="h-delete" aria-label="Delete chat">${ICON_CLOSE}</button>
    `;
    card.appendChild(buildMoveSelect(c));
    card.addEventListener("click", (e) => {
      if (e.target.closest(".h-delete") || e.target.closest(".h-move")) return;
      loadChat(c.id);
    });
    card.querySelector(".h-delete").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteChat(c.id);
    });
    hList.appendChild(card);
  }
}

async function loadChat(id) {
  // Always refresh from the server so switching browsers / clearing localStorage
  // shows the latest persisted state.
  let chat = null;
  try {
    const data = await apiRequest(`/v1/chats/${encodeURIComponent(id)}`);
    const cached = chatCache.find(c => c.id === id);
    chat = serverChatToLocal(data, cached?.model || null);
    const idx = chatCache.findIndex(c => c.id === id);
    if (idx >= 0) chatCache[idx] = chat; else chatCache.push(chat);
    saveChats(chatCache);
  } catch (e) {
    console.warn("Could not load chat from server:", e.message);
    chat = chatCache.find(c => c.id === id);
    if (!chat) return;
  }
  if (!chat) return;
  conversation.length = 0;
  if (Array.isArray(chat.messages)) {
    conversation.push(...chat.messages);
  }
  currentChatId = chat.id;
  chatLog.innerHTML = "";
  if (conversation.length === 0) {
    chatLog.appendChild(emptyState);
    emptyState.style.display = "";
  } else {
    for (const m of conversation) {
      if (m.role === "assistant") {
        const { div, content } = createMessage("assistant", m.content || "");
        if (m.thinking) renderThinking(div, m.thinking);
        if (m.content) renderMarkdown(content, m.content);
      } else {
        createMessage(m.role, m.content || "");
      }
    }
  }
  if (chat.model && models.some(m => m.id === chat.model)) {
    selectedModel = chat.model;
    modelSelect.value = chat.model;
    loadingModel = null;
    updatePickerStatus();
  }
  historyPanel.classList.remove("open");
  historyToggle.classList.remove("open");
  const m = models.find(x => x.id === selectedModel);
  updateCtxMeter(estimateTokens(), m?.ctx || 131072);
  if (shouldAutoScroll(chatLog)) autoScroll(chatLog);
  input.focus();
}

async function deleteChat(id) {
  try {
    await apiRequest(`/v1/chats/${encodeURIComponent(id)}`, { method: "DELETE" });
  } catch (e) {
    console.warn("Could not delete chat on server:", e.message);
  }
  const chats = loadChats().filter(c => c.id !== id);
  saveChats(chats);
  if (currentChatId === id) {
    currentChatId = null;
  }
  renderHistoryPanel();
}

function getFolderLabel(name) {
  return name || "Default";
}

function populateFolderSelects() {
  if (!hFolder) return;

  const saved = hFolder.value;
  hFolder.innerHTML = `<option value="${ALL_FOLDERS}">All folders</option>`;
  for (const f of folders) {
    const label = getFolderLabel(f.name);
    hFolder.insertAdjacentHTML("beforeend", `<option value="${escapeHtml(f.name)}">${escapeHtml(label)} (${f.count})</option>`);
  }
  if ([...hFolder.options].some(o => o.value === saved)) {
    hFolder.value = saved;
  } else {
    hFolder.value = ALL_FOLDERS;
    currentFolder = ALL_FOLDERS;
  }

}

async function loadFolders() {
  try {
    const data = await apiRequest("/v1/chats/folders");
    folders = data.data || [];
  } catch (e) {
    console.warn("Could not load folders:", e.message);
    folders = [];
  }
  populateFolderSelects();
}

async function createFolder() {
  const name = prompt("Name for the new folder:");
  if (!name || !name.trim()) return;
  const folder = name.trim();
  await ensureServerChat("New chat", folder);
  currentFolder = folder;
  hFolder.value = folder;
  await loadFolders();
  renderHistoryPanel();
  historyPanel.classList.add("open");
  historyToggle.classList.add("open");
}

async function moveChat(chatId, folder) {
  try {
    await apiRequest(`/v1/chats/${encodeURIComponent(chatId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder }),
    });
  } catch (e) {
    console.warn("Could not move chat:", e.message);
    showError("Could not move chat: " + e.message);
    return;
  }
  const chats = loadChats();
  const chat = chats.find(c => c.id === chatId);
  if (chat) {
    chat.folder = folder;
    saveChats(chats);
  }
  await loadFolders();
  renderHistoryPanel();
}

function buildMoveSelect(chat) {
  const select = document.createElement("select");
  select.className = "h-move";
  select.innerHTML = `<option value="">Move to…</option>`;
  for (const f of folders) {
    if (f.name === chat.folder) continue;
    const label = getFolderLabel(f.name);
    select.insertAdjacentHTML("beforeend", `<option value="${escapeHtml(f.name)}">${escapeHtml(label)}</option>`);
  }
  select.insertAdjacentHTML("beforeend", `<option value="__new__">+ New folder</option>`);
  select.addEventListener("change", async (e) => {
    const value = e.target.value;
    e.target.value = "";
    if (value === "__new__") {
      const name = prompt("Name for the new folder:");
      if (!name || !name.trim()) return;
      await moveChat(chat.id, name.trim());
    } else if (value) {
      await moveChat(chat.id, value);
    }
  });
  return select;
}

async function exportChats() {
  try {
    const data = await apiRequest("/v1/chats/export");
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `arc-llama-chats-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    console.warn("Could not export chats:", e.message);
    showError("Export failed: " + e.message);
  }
}

async function importChatsFromFile() {
  const file = hImportInput.files?.[0];
  if (!file) return;
  hImportInput.value = "";
  let body;
  try {
    const text = await file.text();
    body = JSON.parse(text);
  } catch (e) {
    showError("Import failed: invalid JSON file");
    return;
  }
  const chats = body.chats;
  if (!Array.isArray(chats)) {
    showError("Import failed: missing 'chats' array");
    return;
  }
  try {
    const r = await fetch("/v1/chats/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chats, overwrite: false }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    await syncChatsFromServer();
    showError(`Imported ${data.imported || 0}, skipped ${data.skipped || 0}, errors ${data.errors || 0}.`);
  } catch (e) {
    showError("Import failed: " + e.message);
  }
}

function renderSettingsPanel() {
  const m = models.find(m => m.id === selectedModel);
  sModelName.textContent = selectedModel || "Not selected";
  if (!m || (m.owned_by && m.owned_by.startsWith("upstream:"))) {
    sFields.innerHTML = '<div class="s-upstream">Settings not available for upstream models.</div>';
    return;
  }
  const ctx        = m.ctx        ?? 32768;
  const ctk        = m.cache_type_k ?? "q8_0";
  const ctv        = m.cache_type_v ?? "q8_0";
  const parallel   = m.parallel   ?? 1;
  const kvClass    = m.kv_class   ?? "default";

  const kvOpts = KV_TYPES.map(v => `<option value="${v}"${v===ctk?" selected":""}>${v}</option>`).join("");
  const kvOptsV = KV_TYPES.map(v => `<option value="${v}"${v===ctv?" selected":""}>${v}</option>`).join("");
  const classOpts = KV_CLASSES.map(v => `<option value="${v}"${v===kvClass?" selected":""}>${v}</option>`).join("");

  sFields.innerHTML = `
    <div class="s-field"><label>Context (tokens)</label>
      <input id="s-ctx" type="number" min="256" max="1048576" step="1024" value="${ctx}"></div>
    <div class="s-field"><label>KV Cache K</label>
      <select id="s-ctk">${kvOpts}</select></div>
    <div class="s-field"><label>KV Cache V</label>
      <select id="s-ctv">${kvOptsV}</select></div>
    <div class="s-field"><label>Parallel slots</label>
      <input id="s-par" type="number" min="1" max="32" value="${parallel}"></div>
    <div class="s-field"><label>KV Class</label>
      <select id="s-kvc">${classOpts}</select></div>
    <button class="s-apply" id="s-apply">Apply</button>
    <div class="s-note">Takes effect on next model load.</div>
  `;
  $("#s-apply").addEventListener("click", applySettings);
}

async function applySettings() {
  const m = models.find(m => m.id === selectedModel);
  if (!m) return;
  const btn = $("#s-apply");
  btn.disabled = true;
  sFeedback.textContent = "";
  const body = {
    ctx:          parseInt($("#s-ctx").value, 10),
    cache_type_k: $("#s-ctk").value,
    cache_type_v: $("#s-ctv").value,
    parallel:     parseInt($("#s-par").value, 10),
    kv_class:     $("#s-kvc").value,
  };
  try {
    const r = await fetch(`/admin/models/${encodeURIComponent(selectedModel)}/edit`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.status);
    m.ctx          = body.ctx;
    m.cache_type_k = body.cache_type_k;
    m.cache_type_v = body.cache_type_v;
    m.parallel     = body.parallel;
    m.kv_class     = body.kv_class;
    sFeedback.style.color = "var(--accent-bright)";
    sFeedback.textContent = "Saved.";
  } catch (e) {
    sFeedback.style.color = "#e8b0b0";
    sFeedback.textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
  }
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
    models = models.map(m => {
      const s = modelMap.get(m.id);
      if (s) {
        m.loaded        = s.loaded;
        m.ctx           = s.ctx           ?? m.ctx;
        m.cache_type_k  = s.cache_type_k  ?? m.cache_type_k;
        m.cache_type_v  = s.cache_type_v  ?? m.cache_type_v;
        m.kv_class      = s.kv_class      ?? m.kv_class;
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
});

function createMessage(role, text = "") {
  if (emptyState) emptyState.style.display = "none";
  const div = document.createElement("div");
  div.className = "message " + role;
  const roleLabel = document.createElement("div");
  roleLabel.className = "role";
  roleLabel.textContent = role === "user" ? "You" : role === "system" ? "System" : "Assistant";
  div.appendChild(roleLabel);
  if (role === "assistant") {
    const indicator = document.createElement("span");
    indicator.id = "streaming-indicator";
    indicator.textContent = "●";
    indicator.style.color = "var(--accent-bright)";
    indicator.style.opacity = "0";
    roleLabel.appendChild(indicator);
    const thinkingBlock = document.createElement("div");
    thinkingBlock.className = "thinking-block";
    thinkingBlock.style.display = "none";
    const thinkingToggle = document.createElement("div");
    thinkingToggle.className = "thinking-toggle";
    thinkingToggle.innerHTML = '<span class="chevron">▶</span><span>Thinking</span>';
    thinkingToggle.addEventListener("click", () => {
      thinkingToggle.classList.toggle("open");
      thinkingContent.classList.toggle("open");
    });
    const thinkingContent = document.createElement("div");
    thinkingContent.className = "thinking-content";
    thinkingBlock.appendChild(thinkingToggle);
    thinkingBlock.appendChild(thinkingContent);
    div.appendChild(thinkingBlock);
  }
  const content = document.createElement("div");
  content.className = "content";
  content.textContent = text;
  div.appendChild(content);
  chatLog.appendChild(div);
  if (shouldAutoScroll(chatLog)) autoScroll(chatLog);
  return { div, content };
}

function showError(text) {
  const { content } = createMessage("error", text);
  content.parentElement.classList.add("error-card");
  content.parentElement.querySelector(".role").textContent = "Error";
}

async function ensureModelLoaded() {
  const m = models.find(m => m.id === selectedModel);
  if (!m) throw new Error("No model selected");
  if (m.loaded || (m.owned_by && m.owned_by.startsWith("upstream:"))) return;
  loadingModel = selectedModel;
  updatePickerStatus();
  try {
    const r = await fetch(`/admin/load/${encodeURIComponent(selectedModel)}`, {
      method: "POST",
      headers: authHeaders(),
    });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(`Load failed: ${r.status} ${t}`);
    }
    m.loaded = true;
  } catch (e) {
    throw e;
  } finally {
    loadingModel = null;
    updatePickerStatus();
  }
}

async function sendMessage() {
  const text = input.value.trim();
  if (generating || !selectedModel) return;
  if (!text && !hasReadyAttachments()) return;
  if (hasProcessingAttachments()) {
    showError("Please wait for attachments to finish processing.");
    return;
  }

  const attachmentText = buildAttachmentText();
  const fullText = text
    ? attachmentText ? `${text}\n\n${attachmentText}` : text
    : attachmentText;

  input.value = "";
  input.style.height = "auto";
  clearAttachments();
  conversation.push({ role: "user", content: fullText });
  createMessage("user", fullText);

  // Persist this conversation on the server (best-effort).
  await ensureServerChat(fullText, currentFolder);
  serverAppendMessages(currentChatId, [{ role: "user", content: fullText }]);

  generating = true;
  sendButton.disabled = true;
  inputWrap.classList.add("generating");
  const streamingDot = $("#streaming-indicator");
  if (streamingDot) streamingDot.style.opacity = "1";

  try {
    await ensureModelLoaded();
  } catch (e) {
    showError(e.message);
    finishGeneration();
    return;
  }

  const assistantMsg = createMessage("assistant");
  conversation.push({ role: "assistant", content: "", thinking: "" });
  const convoIndex = conversation.length - 1;

  let streamRaw = "";
  let lastDisplayedContent = "";
  let currentThinking = "";

  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: selectedModel,
        messages: conversation.slice(0, -1),
        stream: true,
        stream_options: { include_usage: true },
      }),
    });
    if (!r.ok) {
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
    await serverAppendMessages(currentChatId, [{ role: "assistant", content: conversation[convoIndex].content }]);
    await saveCurrentChat();
  } catch (e) {
    assistantMsg.div.remove();
    conversation.pop();
    showError("Generation failed: " + e.message);
  } finally {
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

function parseThinking(text) {
  const tail = text.slice(-15);
  const lastLt = tail.lastIndexOf("<");
  if (lastLt !== -1) {
    const afterLt = tail.slice(lastLt);
    const possible = ["<think>", "<thinking>", "</think>", "</thinking>"];
    for (const tag of possible) {
      if (tag.startsWith(afterLt) && afterLt.length < tag.length) {
        return { thinking: "", content: text, hasPartialTag: true };
      }
    }
  }
  let thinking = "";
  let content = text;
  const thinkMatches = [...text.matchAll(/<think>([\s\S]*?)<\/think>/g)];
  // Reasoning arrives in small SSE deltas. Preserve the model's whitespace;
  // adding a newline for every delta turns normal prose into a column.
  for (const m of thinkMatches) thinking += m[1];
  content = content.replace(/<think>[\s\S]*?<\/think>/g, "");
  const thinkingMatches = [...text.matchAll(/<thinking>([\s\S]*?)<\/thinking>/g)];
  for (const m of thinkingMatches) thinking += m[1];
  content = content.replace(/<thinking>[\s\S]*?<\/thinking>/g, "");
  const unclosedThink = content.match(/<think>([\s\S]*)$/);
  const unclosedThinking = content.match(/<thinking>([\s\S]*)$/);
  if (unclosedThink) {
    thinking += unclosedThink[1];
    content = content.replace(/<think>[\s\S]*$/, "");
  } else if (unclosedThinking) {
    thinking += unclosedThinking[1];
    content = content.replace(/<thinking>[\s\S]*$/, "");
  }
  return { thinking: thinking.trim(), content: content.trimEnd(), hasPartialTag: false };
}

function appendChunk(container, text) {
  const span = document.createElement("span");
  span.className = "token-chunk";
  span.textContent = text;
  container.appendChild(span);
  requestAnimationFrame(() => span.classList.add("revealed"));
}

function renderMarkdown(container, text) {
  if (typeof marked === "undefined" || typeof hljs === "undefined") {
    container.textContent = text;
    return;
  }
  const raw = text
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/<thinking>[\s\S]*?<\/thinking>/g, "")
    .replace(/[\n\r]+$/, "")
    .trimEnd();
  try {
    let html = marked.parse(raw, { renderer: mdRenderer });
    html = html.replace(/<p>\s*<\/p>/g, "").replace(/<p><br\s*\/?><\/p>/g, "");
    container.innerHTML = html;
    attachCopyButtons(container);
  } catch (e) {
    console.warn("Markdown render failed, falling back to plain text", e);
    container.textContent = text;
  }
}

function escapeHtml(s) {
  return ArcMarkdownSafety.escapeHtml(s);
}

function renderThinking(messageDiv, thinkingText) {
  const thinkingBlock = messageDiv.querySelector(".thinking-block");
  if (!thinkingBlock) return;
  const thinkingContent = thinkingBlock.querySelector(".thinking-content");
  const trimmed = String(thinkingText || "").trim();
  thinkingContent.textContent = trimmed;
  thinkingBlock.style.display = trimmed ? "" : "none";
}

function finishGeneration() {
  generating = false;
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

// ------------------------------------------------------------------
// File attachments
// ------------------------------------------------------------------

const TEXT_EXTENSIONS = new Set([".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv"]);

function isTextFile(file) {
  if (file.type.startsWith("text/")) return true;
  const name = file.name.toLowerCase();
  for (const ext of TEXT_EXTENSIONS) {
    if (name.endsWith(ext)) return true;
  }
  return false;
}

function isPdfFile(file) {
  return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}

function generateAttachmentId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function renderAttachments() {
  attachmentStrip.innerHTML = "";
  if (attachments.length === 0) return;
  for (const a of attachments) {
    const chip = document.createElement("div");
    chip.className = "attachment-chip" + (a.error ? " error" : a.processing ? " processing" : "");
    chip.dataset.id = a.id;
    const ICON_CLIP = '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M16.5 6v11.5c0 2.485-2.015 4.5-4.5 4.5S7.5 19.985 7.5 17.5V5c0-1.657 1.343-3 3-3s3 1.343 3 3v12.5c0 .828-.672 1.5-1.5 1.5s-1.5-.672-1.5-1.5V6h-2v11.5c0 1.933 1.567 3.5 3.5 3.5s3.5-1.567 3.5-3.5V5c0-2.761-2.239-5-5-5S5 2.239 5 5v12.5c0 3.59 2.91 6.5 6.5 6.5s6.5-2.91 6.5-6.5V6h-2z"/></svg>';
    const ICON_CLOSE = '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>';
    const icon = a.processing ? '<span class="spinner"></span>'
                 : a.error ? '<span>!</span>'
                 : `<span class="attachment-icon">${ICON_CLIP}</span>`;
    chip.innerHTML = `
      ${icon}
      <span class="filename" title="${escapeHtml(a.file.name)}">${escapeHtml(a.file.name)}</span>
      <button class="remove" aria-label="Remove attachment">${ICON_CLOSE}</button>
    `;
    chip.querySelector(".remove").addEventListener("click", () => removeAttachment(a.id));
    attachmentStrip.appendChild(chip);
  }
}

function addAttachment(file) {
  const id = generateAttachmentId();
  const a = { id, file, text: "", processing: true, error: "" };
  attachments.push(a);
  renderAttachments();
  processAttachment(a).finally(renderAttachments);
}

async function processAttachment(a) {
  try {
    if (isPdfFile(a.file)) {
      const form = new FormData();
      form.append("file", a.file);
      const r = await fetch("/admin/parse-pdf", {
        method: "POST",
        headers: authHeaders(),
        body: form,
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      a.text = data.text || "";
    } else if (isTextFile(a.file)) {
      a.text = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("Could not read file"));
        reader.readAsText(a.file);
      });
    } else {
      throw new Error("Unsupported file type");
    }
    a.error = "";
  } catch (e) {
    a.error = e.message;
    a.text = "";
  } finally {
    a.processing = false;
  }
}

function removeAttachment(id) {
  attachments = attachments.filter(a => a.id !== id);
  renderAttachments();
}

function clearAttachments() {
  attachments = [];
  renderAttachments();
}

function buildAttachmentText() {
  const parts = [];
  for (const a of attachments) {
    if (a.error || a.processing || !a.text) continue;
    parts.push(`[Attachment: ${a.file.name}]\n${a.text.trim()}`);
  }
  return parts.join("\n\n");
}

function hasReadyAttachments() {
  return attachments.some(a => !a.processing && !a.error && a.text);
}

function hasProcessingAttachments() {
  return attachments.some(a => a.processing);
}

attachButton.addEventListener("click", () => pdfInput.click());

pdfInput.addEventListener("change", () => {
  const files = Array.from(pdfInput.files || []);
  pdfInput.value = "";
  for (const file of files) addAttachment(file);
});

// Drag-and-drop on the input area
inputWrap.addEventListener("dragover", (e) => {
  e.preventDefault();
  e.stopPropagation();
  inputWrap.style.borderColor = "var(--accent-bright)";
});
inputWrap.addEventListener("dragleave", (e) => {
  e.preventDefault();
  e.stopPropagation();
  inputWrap.style.borderColor = "";
});
inputWrap.addEventListener("drop", (e) => {
  e.preventDefault();
  e.stopPropagation();
  inputWrap.style.borderColor = "";
  const files = Array.from(e.dataTransfer?.files || []);
  for (const file of files) addAttachment(file);
});

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
  ensureModelLoaded().catch((e) => showError(e.message));
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
    showError(e.message);
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
    sendMessage();
  }
});

sendButton.addEventListener("click", async () => {
  const text = input.value.trim();
  if (await executeSlashCommand(text)) {
    input.value = "";
    input.style.height = "auto";
    hideCommandPalette();
    return;
  }
  sendMessage();
});

$("#theme-toggle").addEventListener("click", () => { const next = document.documentElement.dataset.theme === "light" ? "dark" : "light"; localStorage.setItem(THEME_KEY, next); applyTheme(next); });

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 96) + "px";
  updateCommandPalette();
});

(async function init() {
  await initAdminToken();
  await fetchModels();
  await fetchStatus();
  await loadFolders();
  await syncChatsFromServer();
  statusPoller = setInterval(fetchStatus, 3000);
})();
