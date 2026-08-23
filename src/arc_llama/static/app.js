// arc-llama model manager — polling client over /admin/status + /admin/load|stop.

const $ = (sel) => document.querySelector(sel);
const fmtVram = (mb) => mb == null ? "?" : `${(mb / 1024).toFixed(1)} GB`;
const fmtPath = (p) => {
  if (!p) return "—";
  // Show just the basename + parent dir, leave full path on hover.
  const parts = p.split("/").filter(Boolean);
  return parts.slice(-2).join("/");
};

let lastStatus = null;
let inflight = false;

async function fetchStatus(force) {
  // Skip the auto-poll while a row is in edit mode so input values don't
  // get clobbered by re-renders. Manual refresh (force=true) still works.
  if (editingModel && !force) return;
  if (inflight) return;
  inflight = true;
  const footer = $("#status-footer");
  try {
    const r = await fetch("/admin/status");
    if (!r.ok) throw new Error(`status ${r.status}`);
    lastStatus = await r.json();
    render(lastStatus);
    $("#last-updated").textContent = `updated ${new Date().toLocaleTimeString()}`;
    footer.classList.remove("error");
    footer.classList.add("online");
  } catch (e) {
    $("#last-updated").textContent = `error: ${e.message}`;
    footer.classList.remove("online");
    footer.classList.add("error");
  } finally {
    inflight = false;
  }
}

async function postAction(path, label) {
  try {
    const r = await fetch(path, { method: "POST" });
    if (!r.ok) {
      const t = await r.text();
      alert(`${label} failed: ${r.status} ${t}`);
      return;
    }
    await fetchStatus(true);
  } catch (e) {
    alert(`${label} error: ${e.message}`);
  }
}

// Track which model is currently in inline-edit mode (only one at a time).
let editingModel = null;

const KV_OPTIONS = ["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"];

function render(s) {
  $("#server-info").textContent = `${s.server.host}:${s.server.port}`;
  $("#policy-info").textContent =
    s.server.single_resident ? "single-resident" : "multi-resident";

  // GPUs
  const gpuBody = $("#gpus tbody");
  gpuBody.innerHTML = "";
  if (!s.gpus || s.gpus.length === 0) {
    renderEmpty(gpuBody, 6, "No GPUs detected", "Enable a device in config to make it available for model loading.");
  } else {
    for (const g of s.gpus) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono">${g.pci_slot}</td>
        <td>${g.arch}</td>
        <td>${g.name || "—"}</td>
        <td class="mono">level_zero:${g.sycl_index}</td>
        <td>${fmtVram(g.vram_mb)}</td>
        <td><span class="pill ${g.enabled ? "loaded" : "idle"}">${g.enabled ? "yes" : "no"}</span></td>
      `;
      gpuBody.appendChild(tr);
    }
  }

  // Models
  const modelBody = $("#models tbody");
  modelBody.innerHTML = "";
  const audio = s.audio_models || [];
  if ((!s.models || s.models.length === 0) && audio.length === 0) {
    renderEmpty(modelBody, 8, "No models registered", "Click Scan for models to discover GGUF files, or add upstream endpoints in config.");
  } else {
    for (const m of s.models || []) {
      const tr = document.createElement("tr");
      if (editingModel === m.name) {
        renderEditRow(tr, m);
      } else {
        renderViewRow(tr, m);
      }
      modelBody.appendChild(tr);
    }
    // Speech backends (STT and TTS). No ctx/KV to show or edit — their launch
    // config is an engine/task/mode triple, not a llama.cpp recipe.
    for (const m of audio) {
      const tr = document.createElement("tr");
      renderAudioRow(tr, m);
      modelBody.appendChild(tr);
    }
  }
}

function renderAudioRow(tr, m) {
  tr.className = m.loaded ? "bright" : "dim";
  const pill = m.loaded
    ? '<span class="pill loaded">loaded</span>'
    : '<span class="pill idle">idle</span>';
  const pinned = m.always_resident ? ' <span class="upstream-hint" title="Exempt from single-resident eviction">pinned</span>' : "";
  tr.innerHTML = `
    <td>${pill}</td>
    <td><strong>${m.name}</strong> <span class="upstream-hint" title="${m.engine || ""}">${m.task || "audio"}</span>${pinned}</td>
    <td class="mono">${m.gpu_pci_slot || "—"}</td>
    <td class="mono">${m.port || "—"}</td>
    <td class="mono">—</td>
    <td class="mono">${m.mode || "—"}</td>
    <td class="path" title="${m.path || ""}">${fmtPath(m.path)}</td>
    <td class="actions"></td>
  `;
  const actions = tr.querySelector(".actions");
  const wrap = document.createElement("div");
  wrap.className = "row-actions";
  if (!m.launchable) {
    const note = document.createElement("span");
    note.className = "upstream-link";
    note.title = m.launch_error || "This model has no runnable backend.";
    note.textContent = "not launchable";
    wrap.appendChild(note);
  } else if (m.loaded) {
    const stop = document.createElement("button");
    stop.className = "danger";
    stop.textContent = "Stop";
    stop.onclick = () => postAction(`/admin/stop/${encodeURIComponent(m.name)}`, "Stop");
    wrap.appendChild(stop);
  } else {
    const load = document.createElement("button");
    load.textContent = "Load";
    load.onclick = () => postAction(`/admin/load/${encodeURIComponent(m.name)}`, "Load");
    wrap.appendChild(load);
  }
  actions.appendChild(wrap);
}

function renderEmpty(tbody, colspan, title, hint) {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td colspan="${colspan}">
      <div class="empty-state">
        <div class="icon">∅</div>
        <div class="title">${title}</div>
        <div class="hint">${hint}</div>
      </div>
    </td>
  `;
  tbody.appendChild(tr);
}

function renderViewRow(tr, m) {
  const isUpstream = m.source && m.source !== "local";
  tr.className = m.loaded ? "bright" : "dim";
  const kv = `${m.cache_type_k || "?"}/${m.cache_type_v || "?"}`;
  const pill = isUpstream
    ? `<span class="pill upstream">${m.source}</span>`
    : m.loaded
      ? '<span class="pill loaded">loaded</span>'
      : '<span class="pill idle">idle</span>';
  const nameCell = isUpstream
    ? `${m.name} <span class="upstream-hint" title="${m.upstream_url}">${m.upstream_name}</span>`
    : m.name;
  tr.innerHTML = `
    <td>${pill}</td>
    <td><strong>${nameCell}</strong></td>
    <td class="mono">${m.gpu_pci_slot || "—"}${isUpstream ? " *" : ""}</td>
    <td class="mono">${m.port || "—"}</td>
    <td class="mono">${m.ctx ?? "?"}</td>
    <td class="mono">${kv}</td>
    <td class="path" title="${m.path || ""}">${isUpstream ? "— " : fmtPath(m.path)}</td>
    <td class="actions"></td>
  `;
  const actions = tr.querySelector(".actions");

  if (isUpstream) {
    // Upstream models show a "via" link instead of load/stop
    const via = document.createElement("span");
    via.className = "upstream-link";
    via.textContent = m.upstream_name;
    actions.appendChild(via);
  } else {
    const wrap = document.createElement("div");
    wrap.className = "row-actions";

    const editBtn = document.createElement("button");
    editBtn.className = "secondary";
    editBtn.textContent = "Edit";
    editBtn.onclick = () => {
      editingModel = m.name;
      fetchStatus(true);
    };
    wrap.appendChild(editBtn);

    if (m.loaded) {
      const stop = document.createElement("button");
      stop.className = "danger";
      stop.textContent = "Stop";
      stop.onclick = () => postAction(`/admin/stop/${encodeURIComponent(m.name)}`, "stop");
      wrap.appendChild(stop);
    } else {
      const load = document.createElement("button");
      load.className = "primary";
      load.textContent = "Load";
      load.onclick = () => postAction(`/admin/load/${encodeURIComponent(m.name)}`, "load");
      wrap.appendChild(load);
    }
    actions.appendChild(wrap);
  }
}

function renderEditRow(tr, m) {
  tr.className = "editing";
  const optsK = KV_OPTIONS.map(o =>
    `<option value="${o}"${o === m.cache_type_k ? " selected" : ""}>${o}</option>`
  ).join("");
  const optsV = KV_OPTIONS.map(o =>
    `<option value="${o}"${o === m.cache_type_v ? " selected" : ""}>${o}</option>`
  ).join("");
  tr.innerHTML = `
    <td><span class="pill warn">editing</span></td>
    <td><strong>${m.name}</strong></td>
    <td class="mono">${m.gpu_pci_slot}</td>
    <td class="mono">${m.port}</td>
    <td><input class="edit-field" type="number" min="256" max="1048576" step="1024" value="${m.ctx ?? 8192}" data-field="ctx"/></td>
    <td>
      <div class="kv-selects">
        <select class="edit-field" data-field="cache_type_k">${optsK}</select>
        <select class="edit-field" data-field="cache_type_v">${optsV}</select>
      </div>
    </td>
    <td class="path" title="${m.path}">${fmtPath(m.path)}</td>
    <td class="actions"></td>
  `;
  const actions = tr.querySelector(".actions");
  const wrap = document.createElement("div");
  wrap.className = "row-actions";

  const save = document.createElement("button");
  save.className = "primary";
  save.textContent = "Save";
  save.onclick = async () => {
    const ctx = parseInt(tr.querySelector('[data-field="ctx"]').value, 10);
    const k = tr.querySelector('[data-field="cache_type_k"]').value;
    const v = tr.querySelector('[data-field="cache_type_v"]').value;
    const wasLoaded = m.loaded;
    if (wasLoaded && !confirm(
      `${m.name} is currently loaded — saving will stop it. Continue?`
    )) {
      return;
    }
    save.disabled = true;
    try {
      const r = await fetch(`/admin/models/${encodeURIComponent(m.name)}/edit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ctx, cache_type_k: k, cache_type_v: v }),
      });
      if (!r.ok) {
        const t = await r.text();
        alert(`edit failed: ${r.status} ${t}`);
        return;
      }
      editingModel = null;
      await fetchStatus(true);
    } finally {
      save.disabled = false;
    }
  };
  wrap.appendChild(save);

  const cancel = document.createElement("button");
  cancel.className = "secondary";
  cancel.textContent = "Cancel";
  cancel.onclick = () => { editingModel = null; fetchStatus(true); };
  wrap.appendChild(cancel);

  actions.appendChild(wrap);
}

$("#refresh").onclick = () => fetchStatus(true);
$("#stop-all").onclick = () => {
  if (confirm("Stop every running llama-server?")) {
    postAction("/admin/stop-all", "stop-all");
  }
};
$("#scan").onclick = async () => {
  const btn = $("#scan");
  btn.disabled = true;
  btn.textContent = "Scanning…";
  try {
    const r = await fetch("/admin/scan", { method: "POST" });
    if (!r.ok) {
      const t = await r.text();
      alert(`scan failed: ${r.status} ${t}`);
      return;
    }
    const j = await r.json();
    const added = j.added || [];
    const msg = added.length
      ? `Found ${j.found} GGUF(s); registered ${added.length} new: ${added.join(", ")}`
      : `Found ${j.found} GGUF(s); nothing new to register.`;
    $("#last-updated").textContent = msg;
    await fetchStatus();
  } catch (e) {
    alert(`scan error: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Scan for models";
  }
};

fetchStatus();
setInterval(fetchStatus, 5000);
