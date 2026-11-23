(function (root) {
  const ArcChat = root.ArcChat = root.ArcChat || {};
  ArcChat.createSettings = function (ctx) {
    const KV_TYPES = ["f16","f32","q8_0","q5_1","q5_0","q4_1","q4_0"];
    const KV_CLASSES = ["default","moe_a3b","qwen3_27b_dense","gemma_swa"];
    const PRESETS_KEY = "arc-llama-presets";
    let settingsDirty = false;
    let runtimeBinaryDirty = false, runtimeEnvDirty = false;
    let settingsDraftModel = null;
    let previewTimer = null, previewAbort = null, previewSequence = 0;
    let pendingReloadModel = null;
    function cancelPreview() {
      clearTimeout(previewTimer); previewAbort?.abort(); previewSequence++;
    }
    function draftValues() {
      return { ctx: Number(ctx.$("#s-ctx").value), cache_type_k: ctx.$("#s-ctk").value,
        cache_type_v: ctx.$("#s-ctv").value, parallel: Number(ctx.$("#s-par").value),
        kv_class: ctx.$("#s-kvc").value };
    }
    function reloadNotice(text = "", offer = false) {
      const note = ctx.$("#s-reload-note"), button = ctx.$("#s-reload-now");
      if (note) { note.textContent = text; note.hidden = !text; }
      if (button) button.hidden = !offer;
    }
    function previewFit() {
      cancelPreview();
      const sequence = previewSequence, modelId = ctx.selectedModel;
      previewTimer = setTimeout(async () => {
        const controller = new AbortController(); previewAbort = controller;
        try {
          const response = await ctx.request(`/admin/models/${encodeURIComponent(modelId)}/edit`, {
            method: "POST", headers: ctx.authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ ...draftValues(), preview: true }), signal: controller.signal,
          });
          const data = await response.json();
          if (sequence !== previewSequence || modelId !== ctx.selectedModel) return;
          ctx.$("#s-fit").textContent = response.ok
            ? `Unsaved draft. ${vramFitText({ vram_estimate: data.vram_estimate })}`
            : `Unsaved settings. Memory fit unavailable: ${data.detail || response.status}`;
        } catch (error) {
          if (sequence === previewSequence && modelId === ctx.selectedModel && error.name !== "AbortError")
            ctx.$("#s-fit").textContent = "Unsaved settings. Memory fit is unavailable.";
        }
      }, 300);
    }
    async function reloadNow() {
      const modelId = ctx.selectedModel, button = ctx.$("#s-reload-now");
      button.disabled = true;
      try {
        for (const action of ["stop", "load"]) {
          const response = await ctx.request(`/admin/${action}/${encodeURIComponent(modelId)}`, {
            method: "POST", headers: ctx.authHeaders(),
          });
          if (!response.ok) { const data = await response.json(); throw new Error(data.detail || response.status); }
        }
        pendingReloadModel = null;
        if (modelId === ctx.selectedModel) { reloadNotice(); ctx.sFeedback.textContent = "Reloaded with saved settings."; }
        ctx.fetchStatus().catch(() => {});
      } catch (error) {
        if (modelId === ctx.selectedModel) ctx.sFeedback.textContent = `Reload failed: ${error.message}`;
      } finally { button.disabled = false; }
    }
    function renderPresetPanel() {
      const host = ctx.$("#s-preset");
      if (!host || !ctx.selectedModel) return;
      if (host.dataset.model === ctx.selectedModel && host.contains(document.activeElement)) return;
      host.dataset.model = ctx.selectedModel;
      const preset = presetFor(ctx.selectedModel);
      host.innerHTML = `
        <div class="s-title">Chat preset</div>
        <div class="s-field"><label for="s-system">System prompt</label>
          <textarea id="s-system" rows="4" placeholder="Optional instructions sent before every chat"></textarea></div>
        <div class="s-field"><label for="s-temp">Temperature</label>
          <input id="s-temp" type="number" min="0" max="2" step="0.05" placeholder="model default"></div>
        <div class="s-field"><label for="s-max-tokens">Reply token limit</label>
          <input id="s-max-tokens" type="number" min="1" step="1" placeholder="1024"></div>
        <button class="s-apply" id="s-preset-save" type="button">Save preset</button>
        <div class="s-note" id="s-preset-note">Saved in this browser; applies to the next message.</div>
      `;
      ctx.$("#s-system").value = preset.system;
      ctx.$("#s-max-tokens").value = String(preset.max_tokens);
      ctx.$("#s-temp").value = preset.temperature == null ? "" : String(preset.temperature);
      ctx.$("#s-preset-save").addEventListener("click", () => {
        const raw = ctx.$("#s-temp").value.trim();
        const temp = raw === "" ? null : Number(raw);
        if (temp != null && !(temp >= 0 && temp <= 2)) {
          ctx.$("#s-preset-note").textContent = "Temperature must be between 0 and 2.";
          return;
        }
        const maxTokens = Number(ctx.$("#s-max-tokens").value || 1024);
        if (!Number.isSafeInteger(maxTokens) || maxTokens <= 0) {
          ctx.$("#s-preset-note").textContent = "Reply token limit must be a positive whole number.";
          return;
        }
        savePreset(ctx.selectedModel, { system: ctx.$("#s-system").value.trim(), temperature: temp, max_tokens: maxTokens });
        ctx.$("#s-preset-note").textContent = "Preset saved.";
        ctx.budgetBase = null;
        ctx.refreshBudget();
      });
    }

    // A status poll passes { poll: true }. It used to rebuild the open panel
    // every 3 seconds, closing "Runtime for this model" and discarding preset
    // text typed but not yet saved once focus moved away.
    let renderedSettingsSignature = null;
    function renderSettingsPanel({ poll = false } = {}) {
      const m = ctx.models.find(m => m.id === ctx.selectedModel);
      if (settingsDraftModel === ctx.selectedModel) {
        if (!m?.loaded) reloadNotice();
        else if (pendingReloadModel === ctx.selectedModel) reloadNotice("Saved launch settings take effect on the next load.", true);
      }
      const signature = JSON.stringify([ctx.selectedModel, m && ["ctx", "cache_type_k", "cache_type_v", "parallel", "reasoning_budget", "kv_class", "llama_server", "env", "owned_by"].map(k => m[k])]);
      if (poll && signature === renderedSettingsSignature) {
        const fitLine = ctx.$("#s-fit");
        if (fitLine && !settingsDirty) fitLine.textContent = vramFitText(m || {});
        return;
      }
      renderPresetPanel();
      if (settingsDraftModel === ctx.selectedModel && (settingsDirty || ctx.sFields.contains(document.activeElement))) {
        const fitLine = ctx.$("#s-fit");
        if (fitLine) fitLine.textContent = settingsDirty
          ? "Unsaved settings. Checking memory fit before Apply."
          : vramFitText(m || {});
        return;
      }
      cancelPreview();
      settingsDraftModel = ctx.selectedModel;
      renderedSettingsSignature = signature;
      settingsDirty = false;
      runtimeBinaryDirty = false; runtimeEnvDirty = false;
      ctx.sModelName.textContent = ctx.selectedModel || "Not selected";
      if (!m || (m.owned_by && m.owned_by.startsWith("upstream:"))) {
        ctx.sFields.innerHTML = '<div class="s-upstream">Settings not available for upstream models.</div>';
        return;
      }
      const contextTokens = m.ctx        ?? 32768;
      const ctk        = m.cache_type_k ?? "q8_0";
      const ctv        = m.cache_type_v ?? "q8_0";
      const parallel   = m.parallel   ?? 1;
      const kvClass    = m.kv_class   ?? "default";

      const kvOpts = KV_TYPES.map(v => `<option value="${v}"${v===ctk?" selected":""}>${v}</option>`).join("");
      const kvOptsV = KV_TYPES.map(v => `<option value="${v}"${v===ctv?" selected":""}>${v}</option>`).join("");
      const classNames = { default: "General model", moe_a3b: "Mixture of experts, about 3B active", qwen3_27b_dense: "Qwen 3, 27B dense", gemma_swa: "Gemma with sliding-window attention" };
      const classOpts = KV_CLASSES.map(v => `<option value="${v}"${v===kvClass?" selected":""}>${classNames[v]} (${v})</option>`).join("");

      ctx.sFields.innerHTML = `
        <div class="s-field"><label for="s-ctx">Conversation capacity <small>ctx</small></label><div class="s-hint">More tokens keep longer conversations but use more memory.</div>
          <input id="s-ctx" type="number" min="256" max="1048576" step="1024" value="${contextTokens}"></div>
        <div class="s-field"><label for="s-ctk">Key cache precision <small>cache_type_k</small></label><div class="s-hint">Lower precision uses less memory for remembered conversation keys.</div>
          <select id="s-ctk">${kvOpts}</select></div>
        <div class="s-field"><label for="s-ctv">Value cache precision <small>cache_type_v</small></label><div class="s-hint">Lower precision uses less memory for remembered conversation values.</div>
          <select id="s-ctv">${kvOptsV}</select></div>
        <div class="s-field"><label for="s-par">Simultaneous conversations <small>parallel</small></label><div class="s-hint">More slots share the available context and memory.</div>
          <input id="s-par" type="number" min="1" max="32" value="${parallel}"></div>
        <div class="s-field"><label for="s-rbudget">Thinking budget <small>reasoning_budget</small></label><div class="s-hint">-1 no limit, 0 off</div>
          <input id="s-rbudget" type="number" min="-1" max="131072" step="1"></div>
        <div class="s-field"><label for="s-kvc">Memory estimate model family <small>kv_class</small></label><div class="s-hint">Choose the architecture used to estimate conversation memory; this does not change the model.</div>
          <select id="s-kvc">${classOpts}</select></div>
        <div class="s-field s-fit" id="s-fit">${vramFitText(m)}</div>
        <details class="s-runtime"><summary>Runtime for this model</summary>
          <div class="s-field"><label for="s-llama-server">Runtime executable <small>llama_server</small></label><div class="s-hint">Choose a different runtime for this model; empty uses the global runtime.</div>
            <input id="s-llama-server" type="text" placeholder="Use the global runtime"></div>
          <div class="s-field"><label for="s-runtime-env">Runtime environment <small>env</small></label><div class="s-hint">One literal KEY=value per line, applied only to this model.</div>
            <textarea id="s-runtime-env" rows="4" spellcheck="false" placeholder="KEY=value, one per line"></textarea></div>
          <div class="s-note">Applies only to this model. Empty fields use the defaults. Values are literal; no shell expansion.</div>
        </details>
        <button class="s-apply" id="s-apply">Apply</button>
        <div id="s-reload-note" class="s-note" role="status" hidden></div><button id="s-reload-now" type="button" class="s-apply" hidden>Reload now</button>
      `;
      ctx.$("#s-llama-server").value = m.llama_server || "";
      ctx.$("#s-rbudget").value = m.reasoning_budget ?? "";
      ctx.$("#s-llama-server").placeholder = m.effective_llama_server || "Use the global runtime";
      ctx.$("#s-runtime-env").value = Object.entries(m.env || {}).map(([key, value]) => `${key}=${value}`).join("\n");
      ctx.$("#s-apply").addEventListener("click", applySettings);
      ctx.$("#s-reload-now").addEventListener("click", reloadNow);
      if (m.loaded && pendingReloadModel === ctx.selectedModel)
        reloadNotice("Saved launch settings take effect on the next load.", true);
      ctx.sFields.querySelectorAll("input, select, textarea").forEach((field) => {
        const markDirty = () => {
          settingsDirty = true;
          if (field.id === "s-llama-server") runtimeBinaryDirty = true;
          if (field.id === "s-runtime-env") runtimeEnvDirty = true;
          ctx.$("#s-fit").textContent = "Unsaved settings. Checking memory fit before Apply.";
          reloadNotice(ctx.models.find(model => model.id === ctx.selectedModel)?.loaded ? "Apply restarts this loaded model with the new launch settings." : "");
          if (["s-ctx", "s-ctk", "s-ctv", "s-par", "s-kvc"].includes(field.id)) previewFit();
        };
        field.addEventListener("input", markDirty);
        field.addEventListener("change", markDirty);
      });
    }

    // Honest VRAM fit line for the settings panel, from /admin/status's
    // vram_estimate block. Never invents a number: when the server could not
    // estimate, says so plainly.
    function vramFitText(m) {
      const fit = m.vram_estimate;
      if (!fit || fit.estimated_mb == null) return "Memory fit: not estimated yet.";
      const est = `est. ${fit.estimated_mb.toLocaleString()} MiB`;
      if (fit.fit === false) {
        const head = fit.headroom_mb != null
          ? `exceeds the GPU by ${Math.abs(fit.headroom_mb).toLocaleString()} MiB`
          : "will not fit on the configured GPU";
        return `Memory fit: ${est}, ${head}. Reduce context or KV size, or pick a smaller model.`;
      }
      if (fit.fit === true && fit.headroom_mb != null) {
        return `Memory fit: ${est}, ${fit.headroom_mb.toLocaleString()} MiB headroom on the assigned GPU.`;
      }
      return `Memory fit: ${est}. ${fit.detail || "GPU capacity unknown; no fit verdict."}`;
    }

    async function applySettings() {
      const m = ctx.models.find(m => m.id === ctx.selectedModel);
      if (!m) return;
      const env = Object.create(null);
      for (const line of (runtimeEnvDirty ? ctx.$("#s-runtime-env").value.split("\n") : [])) {
        if (!line.trim()) continue;
        const separator = line.indexOf("=");
        const key = line.slice(0, separator).trim();
        if (separator < 1 || !key || /[=\0]/.test(key) || Object.hasOwn(env, key)) {
          ctx.sFeedback.textContent = "Use one unique KEY=value per line.";
          return;
        }
        env[key] = line.slice(separator + 1);
      }
      cancelPreview();
      const modelId = ctx.selectedModel;
      const wasLoaded = !!m.loaded;
      const rawBudget = ctx.$("#s-rbudget").value.trim();
      const reasoningBudget = rawBudget === "" ? null : Number(rawBudget);
      if (reasoningBudget !== null && (!Number.isInteger(reasoningBudget) || reasoningBudget < -1 || reasoningBudget > 131072)) {
        ctx.sFeedback.textContent = "Thinking budget must be a whole number from -1 to 131072.";
        return;
      }
      const btn = ctx.$("#s-apply");
      btn.disabled = true;
      ctx.sFeedback.textContent = "";
      const body = {
        ...draftValues(),
        reasoning_budget: reasoningBudget,
        reload:       true,
        ...(runtimeBinaryDirty ? { llama_server: ctx.$("#s-llama-server").value.trim() } : {}),
        ...(runtimeEnvDirty ? { env } : {}),
      };
      const launchChanged = Object.entries(draftValues()).some(([key, value]) => value !== (m[key] ?? ({ctx:32768,cache_type_k:"q8_0",cache_type_v:"q8_0",parallel:1,kv_class:"default"}[key])))
        || reasoningBudget !== (m.reasoning_budget ?? null) || runtimeBinaryDirty || runtimeEnvDirty;
      try {
        const r = await ctx.request(`/admin/models/${encodeURIComponent(modelId)}/edit`, {
          method: "POST",
          headers: ctx.authHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify(body),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || r.status);
        m.ctx          = body.ctx;
        m.cache_type_k = body.cache_type_k;
        m.cache_type_v = body.cache_type_v;
        m.parallel     = body.parallel;
        m.reasoning_budget = body.reasoning_budget;
        m.kv_class     = body.kv_class;
        if (Object.hasOwn(body, "llama_server")) m.llama_server = body.llama_server;
        if (Object.hasOwn(body, "env")) m.env = body.env;
        if (ctx.selectedModel !== modelId) return;
        ctx.sFeedback.style.color = "var(--accent-bright)";
        ctx.sFeedback.textContent = data.reloading ? "Saved. Reloading the model..." : "Saved.";
        pendingReloadModel = wasLoaded && launchChanged && !data.reloading ? modelId : null;
        reloadNotice(pendingReloadModel ? "Saved launch settings take effect on the next load." : "", !!pendingReloadModel);
        settingsDirty = false;
        runtimeBinaryDirty = false; runtimeEnvDirty = false;
        // The VRAM estimate depends on ctx/KV; refresh status so the settings
        // panel re-renders an honest fit line instead of the stale one.
        ctx.fetchStatus().catch(() => {});
      } catch (e) {
        if (ctx.selectedModel !== modelId) return;
        ctx.sFeedback.style.color = "#e8b0b0";
        ctx.sFeedback.textContent = "Error: " + e.message;
      } finally {
        btn.disabled = false;
      }
    }

    function loadPresets() {
      try { return JSON.parse(ctx.storage.getItem(PRESETS_KEY) || "{}") || {}; } catch (_) { return {}; }
    }

    function presetFor(model) {
      const p = loadPresets()[model] || {};
      return {
        system: typeof p.system === "string" ? p.system : "",
        temperature: typeof p.temperature === "number" ? p.temperature : null,
        max_tokens: Number.isSafeInteger(p.max_tokens) && p.max_tokens > 0 ? p.max_tokens : 1024,
      };
    }

    function savePreset(model, preset) {
      const all = loadPresets();
      if (!preset.system && preset.temperature == null && (!preset.max_tokens || preset.max_tokens === 1024)) delete all[model];
      else all[model] = preset;
      try { ctx.storage.setItem(PRESETS_KEY, JSON.stringify(all)); } catch (_) {}
    }
    let started = false;
    function start() {
      if (started) return;
      started = true;
          ctx.settingsToggle?.addEventListener("click", () => { const open = ctx.settingsPanel.classList.toggle("open"); ctx.settingsToggle.classList.toggle("open", open); if (open) renderSettingsPanel(); });
    }
    return { renderPresetPanel, renderSettingsPanel, vramFitText, applySettings, loadPresets, presetFor, savePreset, start };
  };
})(window);
