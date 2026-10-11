(function (root) {
  const ArcChat = root.ArcChat = root.ArcChat || {};
  ArcChat.createHistory = function (ctx) {
    const HISTORY_KEY = "arc-llama-chats";
    const MAX_HISTORY = 50;
    const ALL_FOLDERS = "__all__";
    let chatCache = [];
    let currentFolder = ALL_FOLDERS;
    let folders = [];
    function loadChatsFromStorage() {
      try {
        const raw = ctx.storage.getItem(HISTORY_KEY);
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
        ctx.storage.setItem(HISTORY_KEY, JSON.stringify(chats));
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
      const r = await ctx.request(path, options);
        if (!r.ok) {
          const structured = await ctx.parseStructuredFailure(r);
          if (structured) {
            const err = new Error(structured.message);
            err.structured = structured;
            throw err;
          }
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
        if (ctx.historyPanel.classList.contains("open")) renderHistoryPanel();
      } catch (e) {
        console.warn("Could not sync chats from server:", e.message);
      }
    }

    async function ensureServerChat(titleHint, folder) {
      if (ctx.currentChatId) return;
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
        ctx.currentChatId = data.id;
        const now = Date.now();
        const chats = loadChats();
        chats.unshift(serverChatToLocal(data, ctx.selectedModel));
        chats[0].createdAt = now;
        chats[0].updatedAt = now;
        saveChats(chats);
      } catch (e) {
        console.warn("Could not create chat on server:", e.message);
        // Local-only fallback so the UI keeps working offline.
        const id = generateId();
        ctx.currentChatId = id;
        const now = Date.now();
        const chats = loadChats();
        chats.unshift({ id, title, folder: chatFolder || "", model: ctx.selectedModel, messages: [], createdAt: now, updatedAt: now });
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

    function renderHistoryPanel() {
      const chats = loadChats().filter(c => currentFolder === ALL_FOLDERS || c.folder === currentFolder);
      ctx.hList.innerHTML = "";
      if (chats.length === 0) {
        ctx.hList.innerHTML = '<div class="h-empty">No chats in this folder yet.</div>';
        return;
      }
      for (const c of chats) {
        const card = document.createElement("div");
        card.className = "h-card";
        card.dataset.id = c.id;
        const ICON_CLOSE = '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>';
        card.innerHTML = `
          <div class="h-card-title">${ctx.escapeHtml(c.title)}</div>
          <div class="h-card-meta">
            <span>${ctx.escapeHtml(c.model || "unknown")}</span>
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
        ctx.hList.appendChild(card);
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
      ctx.conversation.length = 0;
      ctx.budgetBase = null;
      ctx.lastUserDiv = null;
      ctx.lastAssistantDiv = null;
      if (Array.isArray(chat.messages)) {
        ctx.conversation.push(...chat.messages);
      }
      ctx.currentChatId = chat.id;
      ctx.chatLog.innerHTML = "";
      if (ctx.conversation.length === 0) {
        ctx.chatLog.appendChild(ctx.emptyState);
        ctx.emptyState.style.display = "";
      } else {
        for (const m of ctx.conversation) {
          if (m.role === "assistant") {
            const { div, content } = ctx.createMessage("assistant", m.content || "");
            if (m.thinking) ctx.renderThinking(div, m.thinking);
            if (m.content) ctx.renderMarkdown(content, m.content);
          } else {
            ctx.createMessage(m.role, m.content || "");
          }
        }
      }
      if (chat.model && ctx.models.some(m => m.id === chat.model)) {
        ctx.selectedModel = chat.model;
        ctx.modelSelect.value = chat.model;
        ctx.loadingModel = null;
        ctx.updatePickerStatus();
      }
      ctx.historyPanel.classList.remove("open");
      ctx.historyToggle.classList.remove("open");
      const m = ctx.models.find(x => x.id === ctx.selectedModel);
      ctx.updateCtxMeter(ctx.estimateTokens(), m?.ctx || 131072);
      if (ctx.shouldAutoScroll(ctx.chatLog)) ctx.autoScroll(ctx.chatLog);
      ctx.input.focus();
    }

    async function deleteChat(id) {
      try {
        await apiRequest(`/v1/chats/${encodeURIComponent(id)}`, { method: "DELETE" });
      } catch (e) {
        console.warn("Could not delete chat on server:", e.message);
      }
      const chats = loadChats().filter(c => c.id !== id);
      saveChats(chats);
      if (ctx.currentChatId === id) {
        ctx.currentChatId = null;
      }
      renderHistoryPanel();
    }

    function getFolderLabel(name) {
      return name || "Default";
    }

    function populateFolderSelects() {
      if (!ctx.hFolder) return;

      const saved = ctx.hFolder.value;
      ctx.hFolder.innerHTML = `<option value="${ALL_FOLDERS}">All folders</option>`;
      for (const f of folders) {
        const label = getFolderLabel(f.name);
        ctx.hFolder.insertAdjacentHTML("beforeend", `<option value="${ctx.escapeHtml(f.name)}">${ctx.escapeHtml(label)} (${f.count})</option>`);
      }
      if ([...ctx.hFolder.options].some(o => o.value === saved)) {
        ctx.hFolder.value = saved;
      } else {
        ctx.hFolder.value = ALL_FOLDERS;
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
      ctx.hFolder.value = folder;
      await loadFolders();
      renderHistoryPanel();
      ctx.historyPanel.classList.add("open");
      ctx.historyToggle.classList.add("open");
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
        ctx.showError("Could not move chat: " + e.message);
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
        select.insertAdjacentHTML("beforeend", `<option value="${ctx.escapeHtml(f.name)}">${ctx.escapeHtml(label)}</option>`);
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
        ctx.showError("Export failed: " + e.message);
      }
    }

    async function importChatsFromFile() {
      const file = ctx.hImportInput.files?.[0];
      if (!file) return;
      ctx.hImportInput.value = "";
      let body;
      try {
        const text = await file.text();
        body = JSON.parse(text);
      } catch (e) {
        ctx.showError("Import failed: invalid JSON file");
        return;
      }
      const chats = body.chats;
      if (!Array.isArray(chats)) {
        ctx.showError("Import failed: missing 'chats' array");
        return;
      }
      try {
        const r = await ctx.request("/v1/chats/import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ chats, overwrite: false }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
        await syncChatsFromServer();
        ctx.showError(`Imported ${data.imported || 0}, skipped ${data.skipped || 0}, errors ${data.errors || 0}.`);
      } catch (e) {
        ctx.showError("Import failed: " + e.message);
      }
    }
    let started = false;
    function start() {
      if (started) return;
      started = true;
          chatCache = loadChatsFromStorage();
          ctx.hNew?.addEventListener("click", ctx.newChat);
          if (ctx.hFolder) ctx.hFolder.addEventListener("change", () => { currentFolder = ctx.hFolder.value; renderHistoryPanel(); });
          if (ctx.hNewFolder) ctx.hNewFolder.addEventListener("click", createFolder);
          if (ctx.hExport) ctx.hExport.addEventListener("click", exportChats);
          if (ctx.hImport) ctx.hImport.addEventListener("click", () => ctx.hImportInput?.click());
          if (ctx.hImportInput) ctx.hImportInput.addEventListener("change", importChatsFromFile);
          ctx.historyToggle?.addEventListener("click", async () => { const open = ctx.historyPanel.classList.toggle("open"); ctx.historyToggle.classList.toggle("open", open); if (open) { await loadFolders(); await syncChatsFromServer(); renderHistoryPanel(); } });
    }
    return {
      getChats: () => chatCache,
      getCurrentFolder: () => currentFolder,
      maxHistory: () => MAX_HISTORY,
      loadChatsFromStorage,
      loadChats,
      saveChats,
      serverChatToLocal,
      apiRequest,
      syncChatsFromServer,
      ensureServerChat,
      serverAppendMessages,
      generateId,
      truncateTitle,
      formatRelativeTime,
      renderHistoryPanel,
      loadChat,
      deleteChat,
      getFolderLabel,
      populateFolderSelects,
      loadFolders,
      createFolder,
      moveChat,
      buildMoveSelect,
      exportChats,
      importChatsFromFile,
      start,
    };
  };
})(window);
