// Plugin catalog and dashboard layout editor controller.
window.ArcDashboard = window.ArcDashboard || {};
window.ArcDashboard.createPluginsController = function createPluginsController({
  document: doc, request, authHeaders, makeButton, navigate,
}) {
  let pluginList = null;
  let uiLayout = null;
  let bound = false;
  let startPromise = null;

  // Plugins panel. The backend's /admin/plugins catalog reflects what was
  // discovered at app creation; here we only render it. Fetch failures are
  // isolated: the panel quietly stays empty and the rest of the page is
  // unaffected.
  const PLUGIN_LABELS = {
    active: "Active",
    error: "Failed to load",
    registered: "Registered",
    failed: "Failed to load",
    disabled: "Disabled",
    incompatible: "Incompatible",
  };

  function pluginStatusLabel(status) {
    return PLUGIN_LABELS[status] || "Registered";
  }

  function createPluginCard(plugin) {
    const card = doc.createElement("article");
    card.className = "plugin-card";

    const body = doc.createElement("div");
    body.className = "plugin-card-main";
    const title = doc.createElement("h3");
    title.textContent = plugin.name;
    body.appendChild(title);
    if (plugin.description) {
      const description = doc.createElement("p");
      description.className = "plugin-meta";
      description.textContent = plugin.description;
      body.appendChild(description);
    }
    if (plugin.error) {
      // Discovery-recorded failure: show the short error so the operator can
      // fix it from the dashboard. No plugin code was executed to collect it.
      const errorEl = doc.createElement("p");
      errorEl.className = "plugin-error";
      errorEl.textContent = plugin.error;
      body.appendChild(errorEl);
    }
    const extra = [plugin.version ? `v${plugin.version}` : null, plugin.ui?.actions?.length ? `${plugin.ui.actions.length} action(s)` : null]
      .filter(Boolean)
      .join(" · ");
    if (extra) {
      const meta = doc.createElement("p");
      meta.className = "plugin-meta";
      meta.textContent = extra;
      body.appendChild(meta);
    }
    const apiRoutes = Array.isArray(plugin.api)
      ? plugin.api.filter((route) => typeof route === "string").slice(0, 8)
      : [];
    if (apiRoutes.length) {
      const api = doc.createElement("p");
      api.className = "plugin-meta";
      const shown = apiRoutes.map((route) => route.slice(0, 128));
      api.textContent = `API: ${shown.join(", ")}${plugin.api.length > shown.length ? ", …" : ""}`;
      body.appendChild(api);
    }

    const side = doc.createElement("div");
    side.className = "plugin-card-side";
    const pill = doc.createElement("span");
    pill.className = `status-pill ${(plugin.status === "error" || plugin.status === "failed") ? "error" : ((plugin.status === "disabled" || plugin.status === "incompatible") ? "warn" : "ready")}`;
    pill.textContent = pluginStatusLabel(plugin.status);
    side.appendChild(pill);
    // Plugin pages: same-origin links validated server-side to stay under
    // /plugins/. They open in a new tab; no plugin markup runs in the dashboard.
    for (const page of (plugin.ui?.pages || []).slice(0, 4)) {
      if (typeof page.path !== "string" || !page.path.startsWith("/plugins/")) continue;
      const link = doc.createElement("a");
      link.className = "secondary plugin-page-link";
      link.href = page.path;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = page.label || "Open";
      side.appendChild(link);
    }

    card.append(body, side);
    return card;
  }

  function renderPlugins() {
    const list = doc.querySelector("#plugin-list");
    if (!list || pluginList == null) return;
    list.replaceChildren();
    if (!pluginList.length) {
      const empty = doc.createElement("p");
      empty.className = "plugin-empty";
      empty.textContent = "No plugins installed. Add-ons exposing an arc_llama.plugins entry point appear here.";
      list.appendChild(empty);
      return;
    }
    for (const plugin of pluginList) list.appendChild(createPluginCard(plugin));
  }

  function allPluginActions() {
    return (pluginList || []).flatMap((p) => (p.ui?.actions || []).map((a) => ({...a, plugin: p.name})));
  }

  function renderPluginActions() {
    const toolbar = doc.querySelector(".section-head .toolbar");
    const pluginActions = doc.querySelector("#plugin-action-list");
    if (!toolbar || !pluginActions || !uiLayout) return;
    toolbar.querySelectorAll(".plugin-action").forEach((n) => n.remove());
    pluginActions.replaceChildren();
    const byId = Object.fromEntries(allPluginActions().map((a) => [a.id, a]));
    for (const id of uiLayout.layout.toolbar || []) {
      const action = byId[id];
      if (!action || (uiLayout.hidden || []).includes(id)) continue;
      const node = makeButton(action.label, "secondary plugin-action", () => {
        if (action.route) navigate(action.route);
      });
      toolbar.insertBefore(node, toolbar.lastElementChild);
    }
    for (const id of uiLayout.layout.plugins || []) {
      const action = byId[id];
      if (!action || (uiLayout.hidden || []).includes(id)) continue;
      const node = makeButton(action.label, "secondary plugin-action", () => {
        if (action.route) navigate(action.route);
      });
      pluginActions.appendChild(node);
    }
  }

  function renderLayoutEditor() {
    const list = doc.querySelector("#ui-layout-list");
    if (!list || !uiLayout) return;
    list.replaceChildren();
    const actions = allPluginActions();
    for (const action of actions) {
      const row = doc.createElement("div"); row.className = "plugin-card";
      const label = doc.createElement("label");
      const check = doc.createElement("input"); check.type = "checkbox"; check.checked = !(uiLayout.hidden || []).includes(action.id);
      check.dataset.action = action.id; label.append(check, ` ${action.label} (${action.plugin})`);
      const select = doc.createElement("select"); select.dataset.action = action.id;
      for (const p of ["toolbar", "plugins", "chat"]) { const o = doc.createElement("option"); o.value = p; o.textContent = p; o.selected = (uiLayout.layout[p] || []).includes(action.id); select.append(o); }
      const up = makeButton("↑", "ghost", () => { const prev = row.previousElementSibling; if (prev) row.parentNode.insertBefore(row, prev); });
      const down = makeButton("↓", "ghost", () => { const next = row.nextElementSibling; if (next) row.parentNode.insertBefore(next, row); });
      up.setAttribute("aria-label", `Move ${action.label} up`); down.setAttribute("aria-label", `Move ${action.label} down`);
      row.append(label, select, up, down); list.append(row);
    }
  }

  async function loadUiLayout() {
    try { const r = await request("/admin/ui/layout", {headers: authHeaders()}); if (r.ok) uiLayout = await r.json(); }
    catch (_) { uiLayout = {layout: {toolbar: [], plugins: []}, hidden: []}; }
    renderPluginActions();
  }

  async function poll() {
    try {
      const response = await request("/admin/plugins", { headers: authHeaders() });
      if (!response.ok) throw new Error(`status ${response.status}`);
      const data = await response.json();
      pluginList = data.plugins || [];
      renderPlugins();
      await loadUiLayout();
    } catch (_) {
      // Keep whatever was shown before; discovery is best-effort.
    }
  }

  function bind() {
    if (bound) return;
    bound = true;
    const dialog = doc.querySelector("#ui-layout-dialog");
    doc.querySelector("#customize-ui")?.addEventListener("click", () => { renderLayoutEditor(); dialog.showModal(); });
    doc.querySelector("#ui-layout-close")?.addEventListener("click", () => dialog.close());
    doc.querySelector("#ui-layout-cancel")?.addEventListener("click", () => dialog.close());
    doc.querySelector("#ui-layout-reset")?.addEventListener("click", () => {
      uiLayout.layout = {toolbar: allPluginActions().map(a => a.id), plugins: [], chat: []};
      uiLayout.hidden = [];
      renderLayoutEditor();
    });
    doc.querySelector("#ui-layout-save")?.addEventListener("click", async () => {
      const layout = {toolbar: [], plugins: [], chat: []}, hidden = [];
      doc.querySelectorAll("#ui-layout-list .plugin-card").forEach((row) => {
        const id = row.querySelector("input").dataset.action;
        const p = row.querySelector("select").value;
        if (row.querySelector("input").checked) layout[p].push(id); else hidden.push(id);
      });
      const r = await request("/admin/ui/layout", {
        method: "PUT", headers: {...authHeaders(), "Content-Type": "application/json"},
        body: JSON.stringify({layout, hidden}),
      });
      if (r.ok) { uiLayout = await r.json(); dialog.close(); renderPluginActions(); }
    });
  }

  return {
    bind,
    start() {
      if (startPromise) return startPromise;
      startPromise = poll();
      return startPromise;
    },
    poll,
    render: renderPlugins,
  };
};
