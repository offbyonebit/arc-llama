(function (root) {
  const ArcChat = root.ArcChat = root.ArcChat || {};
  ArcChat.createSettings = function (ctx) {
    const KV_TYPES = ["f16","f32","q8_0","q5_1","q5_0","q4_1","q4_0"];
    const KV_CLASSES = ["default","moe_a3b","qwen3_27b_dense","gemma_swa"];
    const PRESETS_KEY = "arc-llama-presets";
    let settingsDirty = false;
    let settingsDraftModel = null;
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
        <button class="s-apply" id="s-preset-save" type="button">Save preset</button>
        <div class="s-note" id="s-preset-note">Saved in this browser; applies to the next message.</div>
      `;
      ctx.$("#s-system").value = preset.system;
      ctx.$("#s-temp").value = preset.temperature == null ? "" : String(preset.temperature);
      ctx.$("#s-preset-save").addEventListener("click", () => {
        const raw = ctx.$("#s-temp").value.trim();
        const temp = raw === "" ? null : Number(raw);
        if (temp != null && !(temp >= 0 && temp <= 2)) {
          ctx.$("#s-preset-note").textContent = "Temperature must be between 0 and 2.";
          return;
        }
        savePreset(ctx.selectedModel, { system: ctx.$("#s-system").value.trim(), temperature: temp });
        ctx.$("#s-preset-note").textContent = "Preset saved.";
        ctx.budgetBase = null;
        ctx.refreshBudget();
      });
    }

    function renderSettingsPanel() {
      renderPresetPanel();
      const m = ctx.models.find(m => m.id === ctx.selectedModel);
      if (settingsDraftModel === ctx.selectedModel && (settingsDirty || ctx.sFields.contains(document.activeElement))) {
        const fitLine = ctx.$("#s-fit");
        if (fitLine) fitLine.textContent = settingsDirty
          ? "Unsaved settings. Increasing context raises KV memory use; apply to refresh the estimate."
          : vramFitText(m || {});
        return;
      }
      settingsDraftModel = ctx.selectedModel;
      settingsDirty = false;
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
      const classOpts = KV_CLASSES.map(v => `<option value="${v}"${v===kvClass?" selected":""}>${v}</option>`).join("");

      ctx.sFields.innerHTML = `
        <div class="s-field"><label>Context (tokens)</label>
          <input id="s-ctx" type="number" min="256" max="1048576" step="1024" value="${contextTokens}"></div>
        <div class="s-field"><label>KV Cache K</label>
          <select id="s-ctk">${kvOpts}</select></div>
        <div class="s-field"><label>KV Cache V</label>
          <select id="s-ctv">${kvOptsV}</select></div>
        <div class="s-field"><label>Parallel slots</label>
          <input id="s-par" type="number" min="1" max="32" value="${parallel}"></div>
        <div class="s-field"><label>KV Class</label>
          <select id="s-kvc">${classOpts}</select></div>
        <div class="s-field s-fit" id="s-fit">${vramFitText(m)}</div>
        <button class="s-apply" id="s-apply">Apply</button>
        <div class="s-note">Takes effect on next model load.</div>
      `;
      ctx.$("#s-apply").addEventListener("click", applySettings);
      ctx.sFields.querySelectorAll("input, select").forEach((field) => {
        const markDirty = () => {
          settingsDirty = true;
          ctx.$("#s-fit").textContent = "Unsaved settings. Increasing context raises KV memory use; apply to refresh the estimate.";
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
      const btn = ctx.$("#s-apply");
      btn.disabled = true;
      ctx.sFeedback.textContent = "";
      const body = {
        ctx:          parseInt(ctx.$("#s-ctx").value, 10),
        cache_type_k: ctx.$("#s-ctk").value,
        cache_type_v: ctx.$("#s-ctv").value,
        parallel:     parseInt(ctx.$("#s-par").value, 10),
        kv_class:     ctx.$("#s-kvc").value,
      };
      try {
        const r = await ctx.request(`/admin/models/${encodeURIComponent(ctx.selectedModel)}/edit`, {
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
        m.kv_class     = body.kv_class;
        ctx.sFeedback.style.color = "var(--accent-bright)";
        ctx.sFeedback.textContent = "Saved.";
        settingsDirty = false;
        // The VRAM estimate depends on ctx/KV; refresh status so the settings
        // panel re-renders an honest fit line instead of the stale one.
        ctx.fetchStatus().catch(() => {});
      } catch (e) {
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
      };
    }

    function savePreset(model, preset) {
      const all = loadPresets();
      if (!preset.system && preset.temperature == null) delete all[model];
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
