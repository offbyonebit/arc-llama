(function (root) {
  const ArcChat = root.ArcChat = root.ArcChat || {};
  ArcChat.createAttachments = function (ctx) {
    const IMAGE_LIMIT_BYTES = 10 * 1024 * 1024;
    const TEXT_EXTENSIONS = new Set([".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv"]);
    let attachments = [];
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
      ctx.attachmentStrip.innerHTML = "";
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
          <span class="filename" title="${ctx.escapeHtml(a.file.name)}">${ctx.escapeHtml(a.file.name)}</span>
          <button class="remove" aria-label="Remove attachment">${ICON_CLOSE}</button>
        `;
        chip.querySelector(".remove").addEventListener("click", () => removeAttachment(a.id));
        ctx.attachmentStrip.appendChild(chip);
      }
    }

    function addAttachment(file) {
      const id = generateAttachmentId();
      const a = { id, file, text: "", processing: true, error: "" };
      attachments.push(a);
      renderAttachments();
      processAttachment(a).finally(() => { renderAttachments(); ctx.refreshBudget(); });
    }

    function isImageFile(file) {
      return (file.type || "").startsWith("image/");
    }

    async function processAttachment(a) {
      try {
        if (isImageFile(a.file)) {
          if (!ctx.modelCanSeeImages(ctx.selectedModel)) throw new Error("This model cannot read images");
          if (a.file.size > IMAGE_LIMIT_BYTES) throw new Error("Image is larger than 10 MB");
          a.image = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result || ""));
            reader.onerror = () => reject(new Error("Could not read image"));
            reader.readAsDataURL(a.file);
          });
          a.text = "";
        } else if (isPdfFile(a.file)) {
          const form = new FormData();
          form.append("file", a.file);
          const r = await ctx.request("/admin/parse-pdf", {
            method: "POST",
            headers: ctx.authHeaders(),
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
        if (a.error || a.processing) continue;
        if (a.image) { parts.push(`[Image: ${a.file.name}]`); continue; }
        if (!a.text) continue;
        parts.push(`[Attachment: ${a.file.name}]\n${a.text.trim()}`);
      }
      return parts.join("\n\n");
    }

    function hasReadyAttachments() {
      return attachments.some(a => !a.processing && !a.error && (a.text || a.image));
    }

    function hasProcessingAttachments() {
      return attachments.some(a => a.processing);
    }
    let started = false;
    function start() {
      if (started) return;
      started = true;
          ctx.attachButton?.addEventListener("click", () => ctx.pdfInput?.click());
          ctx.pdfInput?.addEventListener("change", () => { const files = Array.from(ctx.pdfInput.files || []); ctx.pdfInput.value = ""; for (const file of files) addAttachment(file); });
          ctx.inputWrap?.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); ctx.inputWrap.style.borderColor = "var(--accent-bright)"; });
          ctx.inputWrap?.addEventListener("dragleave", (e) => { e.preventDefault(); e.stopPropagation(); ctx.inputWrap.style.borderColor = ""; });
          ctx.inputWrap?.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); ctx.inputWrap.style.borderColor = ""; for (const file of Array.from(e.dataTransfer?.files || [])) addAttachment(file); });
    }
    return { getItems: () => attachments.slice(), isTextFile, isPdfFile, generateAttachmentId, renderAttachments, addAttachment, isImageFile, processAttachment, removeAttachment, clearAttachments, buildAttachmentText, hasReadyAttachments, hasProcessingAttachments, start };
  };
})(window);
