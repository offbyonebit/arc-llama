(function (root) {
  const ArcChat = root.ArcChat = root.ArcChat || {};
  ArcChat.createRendering = function (ctx) {
    let mdRenderer = typeof marked !== "undefined" ? new marked.Renderer() : {};

    function configureMarkdown() {
      if (typeof marked !== "undefined") {
        marked.use({
          gfm: true,
          breaks: false,
          headerIds: false,
          mangle: false,
        });
      }
      mdRenderer = typeof marked !== "undefined" ? new marked.Renderer() : {};
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
    }
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

    function createMessage(role, text = "") {
      if (ctx.emptyState) ctx.emptyState.style.display = "none";
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
      ctx.chatLog.appendChild(div);
      if (ctx.shouldAutoScroll(ctx.chatLog)) ctx.autoScroll(ctx.chatLog);
      return { div, content };
    }

    function showError(text) {
      const { content } = createMessage("error", text);
      content.parentElement.classList.add("error-card");
      content.parentElement.querySelector(".role").textContent = "Error";
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
    let started = false;
    function start() {
      if (started) return;
      started = true;
      configureMarkdown();
    }
    return { attachCopyButtons, createMessage, showError, parseThinking, appendChunk, renderMarkdown, escapeHtml, renderThinking, start };
  };
})(window);
