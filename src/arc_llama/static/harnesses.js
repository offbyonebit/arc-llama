// Install Work agents (OpenCode, Hermes) from the browser. Work's Agent menu and System's Agents
// list share this: a dialog that says exactly what will be downloaded from where, then follows the
// server's install job until it is done, cancelled or failed.
(function () {
  "use strict";

  function el(doc, tag, attrs, ...children) {
    const node = doc.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value == null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of children.flat()) if (child != null && child !== false) node.append(child);
    return node;
  }

  const ACTIVE = new Set(["queued", "running"]);
  const mib = (n) => `${(n / 2 ** 20).toFixed(n < 10 * 2 ** 20 ? 1 : 0)} MB`;

  async function detail(response) {
    try { const body = await response.json(); return body.detail || body.error || `HTTP ${response.status}`; }
    catch (_) { return `HTTP ${response.status}`; }
  }

  // request(url, options) must add the admin auth header; onInstalled(name) refreshes the caller.
  function createInstaller({ document: doc, request, onInstalled, pollMs = 1000 }) {
    let overlay = null, timer = null, closed = true;

    function close() {
      closed = true;
      clearTimeout(timer);
      if (overlay) overlay.hidden = true;
    }

    function sheet(title, ...body) {
      if (!overlay) {
        overlay = el(doc, "div", { class: "harness-dialog", role: "dialog", "aria-modal": "true", "aria-labelledby": "harness-dialog-title" });
        overlay.addEventListener("mousedown", (e) => { if (e.target === overlay && !overlay.dataset.busy) close(); });
        doc.addEventListener("keydown", (e) => { if (e.key === "Escape" && overlay && !overlay.hidden && !overlay.dataset.busy) close(); });
        doc.body.append(overlay);
      }
      overlay.replaceChildren(el(doc, "div", { class: "harness-dialog-sheet" }, el(doc, "h2", { id: "harness-dialog-title" }, title), ...body));
      overlay.hidden = false;
      const focus = overlay.querySelector(".harness-btn.primary") || overlay.querySelector("button");
      focus?.focus();
    }

    function open(harness) {
      closed = false;
      const offer = harness.install || {};
      if (!offer.installable) {
        sheet(`${harness.label} can’t be installed here`,
          el(doc, "p", {}, `Arc Llama has no ${harness.label} download for this computer. Install it yourself and make sure it is on PATH.`),
          el(doc, "div", { class: "harness-actions" }, el(doc, "button", { class: "harness-btn primary", type: "button", onclick: close }, "Close")));
        return;
      }
      sheet(`Install ${harness.label}`,
        el(doc, "p", {}, `Arc Llama will download ${harness.label} and check it before Work uses it. Your own ${harness.label}, if you have one, is not changed.`),
        el(doc, "dl", {},
          el(doc, "dt", {}, "Version"), el(doc, "dd", {}, `${offer.version} (tested with this Arc Llama)`),
          el(doc, "dt", {}, "From"), el(doc, "dd", {}, el(doc, "a", { href: offer.source, target: "_blank", rel: "noopener noreferrer" }, offer.source.replace(/^https:\/\//, ""))),
          el(doc, "dt", {}, "License"), el(doc, "dd", {}, offer.license),
          el(doc, "dt", {}, "Size"), el(doc, "dd", {}, offer.size_note),
          el(doc, "dt", {}, "Saved in"), el(doc, "dd", {}, offer.location)),
        offer.experimental ? el(doc, "p", { class: "harness-note" }, `${harness.label} on Windows hasn’t been tested with Arc Llama yet. It may not work.`) : null,
        el(doc, "div", { class: "harness-actions" },
          el(doc, "button", { class: "harness-link harness-secondary-link", type: "button", onclick: () => start(harness, "latest") }, "Install newest instead (untested)"),
          el(doc, "button", { class: "harness-btn", type: "button", onclick: close }, "Cancel"),
          el(doc, "button", { class: "harness-btn primary", type: "button", onclick: () => start(harness, "pinned") }, "Install")));
    }

    async function start(harness, version) {
      closed = false;
      progressSheet(harness, { status: "queued", step: "Starting" });
      let response;
      try {
        response = await request(`/admin/harnesses/${encodeURIComponent(harness.id)}/install`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ version }),
        });
      } catch (_) { return failed(harness, version, { error: "Could not reach Arc Llama." }); }
      if (!response.ok) return failed(harness, version, { error: await detail(response) });
      follow(harness, version, (await response.json()).id);
    }

    function progressSheet(harness, job) {
      const total = job.bytes_total, done = job.bytes_done;
      const bar = el(doc, "div", { class: "harness-progress" + (total ? "" : " indeterminate"), role: "progressbar", "aria-label": job.step || "Installing",
        "aria-valuemin": total ? "0" : null, "aria-valuemax": total ? String(total) : null, "aria-valuenow": total ? String(done || 0) : null }, el(doc, "span"));
      if (total) bar.firstChild.style.width = `${Math.min(100, Math.round(((done || 0) / total) * 100))}%`;
      sheet(`Installing ${harness.label}`,
        el(doc, "p", { role: "status" }, (job.step || "Working") + (total ? ` · ${mib(done || 0)} of ${mib(total)}` : "")),
        bar,
        logView(job),
        el(doc, "div", { class: "harness-actions" },
          el(doc, "button", { class: "harness-btn", type: "button", disabled: !job.id, onclick: () => cancel(job.id) }, "Cancel install")));
      overlay.dataset.busy = "1";
    }

    function logView(job) {
      return job.log && job.log.length ? el(doc, "details", { class: "harness-log" }, el(doc, "summary", {}, "Details"), el(doc, "pre", {}, job.log.join("\n"))) : null;
    }

    async function cancel(id) {
      try { await request(`/admin/harnesses/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }); } catch (_) { /* the next poll reports it */ }
    }

    function follow(harness, version, id) {
      const poll = async () => {
        if (closed) return;
        let job;
        try {
          const response = await request("/admin/harnesses/jobs");
          if (response.ok) job = ((await response.json()).jobs || []).find((j) => j.id === id);
        } catch (_) { /* keep polling */ }
        if (closed) return;
        if (job && ACTIVE.has(job.status)) progressSheet(harness, job);
        else if (job && job.status === "done") return finished(harness, job);
        else if (job) return failed(harness, version, job);
        timer = setTimeout(poll, pollMs);
      };
      poll();
    }

    function finished(harness, job) {
      delete overlay.dataset.busy;
      sheet(`${harness.label} is ready`,
        el(doc, "p", { role: "status" }, `${harness.label} ${job.installed_version} is installed and selected for Work.` + (job.tested ? "" : " This version hasn’t been tested with Arc Llama.")),
        el(doc, "div", { class: "harness-actions" }, el(doc, "button", { class: "harness-btn primary", type: "button", onclick: close }, "Done")));
      onInstalled?.(harness.id);
    }

    function failed(harness, version, job) {
      if (overlay) delete overlay.dataset.busy;
      const cancelled = job.status === "cancelled";
      sheet(cancelled ? `${harness.label} install cancelled` : `${harness.label} didn’t install`,
        el(doc, "p", { class: cancelled ? null : "harness-error", role: "alert" }, cancelled ? "Nothing was selected. You can install it later." : (job.error || "The install failed.")),
        cancelled ? null : el(doc, "p", {}, "Nothing was changed for Work. Check your connection and disk space, then try again."),
        logView(job),
        el(doc, "div", { class: "harness-actions" },
          el(doc, "button", { class: "harness-btn", type: "button", onclick: close }, "Close"),
          el(doc, "button", { class: "harness-btn primary", type: "button", onclick: () => start(harness, version) }, "Try again")));
    }

    return { open, close };
  }

  // System → Agents: what Work will run, where it came from, and install/remove for Arc Llama's copies.
  function createPanel({ document: doc, request, container }) {
    let info = null;
    const installer = createInstaller({ document: doc, request, onInstalled: () => refresh() });

    function describe(h) {
      if (!h.available) return "Not installed.";
      const managed = (h.managed || []).find((m) => m.path === h.path);
      if (h.source === "managed" && managed) return `Installed by Arc Llama: ${managed.version}${managed.tested ? "" : " (untested)"}.`;
      const where = { path: "found on PATH", env: "set by an environment variable", config: "set in the config" }[h.source] || "found";
      return `Your own copy, ${where}: ${h.path}`;
    }

    async function remove(h, version) {
      if (!confirm(`Remove ${h.label} ${version}? Work can’t use it until it’s installed again.`)) return;
      const response = await request(`/admin/harnesses/${encodeURIComponent(h.id)}/${encodeURIComponent(version)}`, { method: "DELETE" });
      if (!response.ok) alert(await detail(response));
      refresh();
    }

    function render() {
      if (!container) return;
      if (!info) { container.replaceChildren(el(doc, "p", { class: "section-copy" }, "Agent status is unavailable.")); return; }
      const external = (info.harnesses || []).filter((h) => h.id !== "builtin");
      if (!info.experimental) {
        container.replaceChildren(el(doc, "p", { class: "section-copy" }, "OpenCode and Hermes in Work are experimental. Start the server with ARC_LLAMA_EXPERIMENTAL_AGENT=1 to install and use them."));
        return;
      }
      container.replaceChildren(...external.map((h) => {
        const pinned = h.install?.version;
        const hasPinned = (h.managed || []).some((m) => m.version === pinned);
        const actions = [];
        if (h.install?.installable && (!h.available || (h.source === "managed" && !hasPinned)))
          actions.push(el(doc, "button", { class: "harness-btn primary", type: "button", onclick: () => installer.open(h) }, h.available ? `Update to ${pinned}` : "Install"));
        if (h.install?.installable && h.available && h.source !== "managed")
          actions.push(el(doc, "button", { class: "harness-btn", type: "button", onclick: () => installer.open(h) }, "Install Arc Llama’s copy"));
        for (const m of h.managed || []) actions.push(el(doc, "button", { class: "harness-btn", type: "button", onclick: () => remove(h, m.version) }, `Remove ${m.version}`));
        return el(doc, "div", { class: "harness-row" },
          el(doc, "div", {}, el(doc, "h3", {}, h.label), el(doc, "p", {}, describe(h))),
          el(doc, "div", { class: "harness-actions" }, ...actions));
      }));
    }

    async function refresh() {
      try {
        const response = await request("/v1/agent/info");
        info = response.ok ? await response.json() : null;
      } catch (_) { info = null; }
      render();
    }

    return { refresh, render };
  }

  window.ArcHarnesses = { createInstaller, createPanel };
})();
