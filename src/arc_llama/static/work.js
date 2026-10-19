"use strict";
(() => {
  const $ = selector => document.querySelector(selector);
  let token = sessionStorage.getItem("arc-work-token") || "";
  let workspaces = [], jobs = [], selected = null, projectId = "", editing = null, job = null, cursor = 0, busy = false, reviewedRevision = null;
  const stateLabel = value => value.replaceAll("_", " ");
  function error(message) { $("#work-error").textContent = message; $("#work-error").hidden = !message; }
  async function request(path, body, method = body ? "POST" : "GET") {
    const response = await fetch(path, { method, headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}), ...(body ? { "Content-Type": "application/json" } : {}) }, ...(body ? { body: JSON.stringify(body) } : {}) });
    if (response.status === 401) $("#work-auth").hidden = false;
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    return data;
  }
  async function act(operation) {
    if (busy) return;
    busy = true; error("");
    try { await operation(); } catch (e) { error(e.message); }
    finally { busy = false; }
  }
  function projectForm(value = null) {
    editing = value?.id || null;
    $("#workspace-name").value = value?.name || "";
    $("#workspace-root").value = value?.root || "";
    $("#workspace-root").disabled = !!value;
    $("#workspace-instructions").value = value?.instructions || "";
    if (value) $("#workspace-model").value = value.model;
    $("#allow-edits").checked = value?.allow_edits ?? true;
    $("#allow-commands").checked = value?.allow_commands ?? false;
    $("#verify-work").checked = value?.verify ?? true;
    $("#max-attempts").value = value?.max_attempts ?? 1;
    $("#time-budget-minutes").value = value?.time_budget_minutes ?? 30;
    attemptSettings();
  }
  function attemptSettings() {
    const enabled = $("#verify-work").checked;
    if (!enabled && !$("#time-budget-minutes").validity.valid) $("#time-budget-minutes").value = 30;
    $("#max-attempts").disabled = !enabled;
    $("#time-budget-minutes").disabled = !enabled;
  }
  function selectProject(id) {
    projectId = id;
    $("#workspace-picker").value = id;
    $("#start-job").disabled = !id;
    projectForm(workspaces.find(w => w.id === id));
    renderJobs();
  }
  function renderJobs() {
    const list = $("#job-list");
    list.replaceChildren();
    for (const value of jobs.filter(j => !projectId || j.workspace.id === projectId)) {
      const button = document.createElement("button");
      button.type = "button"; button.className = "work-job"; button.dataset.jobId = value.id;
      button.setAttribute("aria-current", String(value.id === selected));
      const name = document.createElement("strong"); name.textContent = value.task.slice(0, 100);
      const status = document.createElement("span"); status.textContent = `${stateLabel(value.state)}${value.pause_requested && value.state === "running" ? " (pausing)" : ""}`;
      button.append(name, status);
      button.addEventListener("click", () => act(async () => { selected = value.id; cursor = 0; reviewedRevision = null; $("#acknowledge-uncertain").checked = false; $("#activity").replaceChildren(); $("#diff").textContent = ""; $("#change-summary").textContent = "Use Review changes to compare this copy with its starting point."; await loadJob(); renderJobs(); }));
      list.append(button);
    }
    if (!list.childElementCount) list.textContent = "No jobs yet.";
  }
  function actionButton(label, action) {
    const button = document.createElement("button"); button.type = "button"; button.textContent = label;
    if (action === "apply") button.disabled = !reviewedRevision;
    button.addEventListener("click", () => act(async () => {
      await request(`/admin/work/jobs/${selected}/${action}`, { acknowledge_uncertain: $("#acknowledge-uncertain").checked, review_revision: reviewedRevision });
      await refresh();
    }));
    $("#job-actions").append(button);
  }
  async function loadJob() {
    if (!selected) return;
    const identity = selected;
    const value = await request(`/admin/work/jobs/${identity}?after=${cursor}`);
    if (identity !== selected) return;
    job = value;
    $("#job-title").textContent = value.task;
    $("#job-state").textContent = `${stateLabel(value.state)}: ${value.message}`;
    $("#job-copy").textContent = `Project copy: ${value.project_copy}`;
    $("#job-actions").replaceChildren();
    const active = ["queued", "running", "waiting_approval", "preparing", "verifying", "settling"].includes(value.state);
    if (["queued", "running", "waiting_approval"].includes(value.state)) actionButton("Pause job", "pause");
    if (["paused", "interrupted"].includes(value.state)) actionButton("Resume job", "resume");
    if (!active && value.state !== "discarded") {
      if (value.apply_state === "none") actionButton("Apply reviewed changes", "apply");
      if (["applied", "applying"].includes(value.apply_state)) actionButton("Undo applied changes", "undo");
      if (value.apply_state === "none") actionButton("Discard project copy", "discard");
    }
    renderAttempts(value);
    renderVerification(value);
    $("#uncertain-row").hidden = !value.pending.some(p => p.state === "executing");
    $("#approval").hidden = value.state !== "waiting_approval";
    $("#approval-command").textContent = value.pending[0]?.arguments?.command || "";
    $("#review-changes").disabled = ["preparing", "settling", "discarded"].includes(value.state);
    for (const event of value.events) {
      cursor = event.seq;
      const row = document.createElement("details"); row.className = "work-event";
      const title = document.createElement("summary"); title.textContent = `${event.seq}. ${stateLabel(event.type)}${event.name ? `: ${event.name}` : ""}`;
      const content = document.createElement("pre"); content.textContent = event.content || event.message || JSON.stringify(event.call || event.files || event, null, 2);
      row.append(title, content); $("#activity").append(row);
    }
  }
  function renderAttempts(value) {
    const entries = value.attempts || [], limit = value.workspace.max_attempts ?? 1;
    const chosen = value.chosen_attempt;
    $("#attempt-history").hidden = !entries.length && limit <= 1;
    $("#attempt-progress").textContent = chosen
      ? `Kept attempt ${chosen} of ${entries.length}. Review its changes before applying.`
      : `Attempt ${value.attempt ?? 1} of up to ${limit}. Budget: ${value.workspace.time_budget_minutes ?? 30} minutes; running attempts are allowed to finish.`;
    const list = $("#attempt-list"); list.replaceChildren();
    for (const entry of entries) {
      const row = document.createElement("li");
      row.textContent = `Attempt ${entry.n}${entry.n === chosen ? " (kept)" : ""}: ${entry.label || stateLabel(entry.verdict)}${entry.incomplete ? " (turn limit reached)" : ""}. ${entry.passed ?? 0} checks passed; ${entry.repairs ?? 0} repair pass(es).`;
      list.append(row);
    }
  }
  const MARK = { pass: "Passed", fail: "Failed", unknown: "Could not confirm" };
  const HOW = { machine: "checked by code", judge: "judged by a reviewer model", safety: "safety check" };
  function renderVerification(value) {
    const report = value.verification, planned = value.criteria || [];
    $("#verification").hidden = !report && !planned.length;
    $("#verdict").textContent = report ? report.label + (report.reason ? ` (${report.reason})` : "") : value.state === "verifying" ? "Checking the result…" : "";
    $("#verdict").dataset.verdict = report?.verdict || "";
    const list = $("#check-list"); list.replaceChildren();
    for (const item of report?.results || planned.map(c => ({ text: c.text, status: "pending", kind: c.type === "judge" ? "judge" : c.type === "may_delete" ? "safety" : "machine", evidence: "" }))) {
      const row = document.createElement("li"); row.dataset.status = item.status;
      row.textContent = `${MARK[item.status] || "Planned"}: ${item.text} (${HOW[item.kind]})${item.evidence ? ` - ${item.evidence}` : ""}`;
      list.append(row);
    }
    const repairs = value.chosen_attempt ? (value.attempts || []).find(entry => entry.n === value.chosen_attempt)?.repairs : value.repairs;
    $("#criteria-note").textContent = repairs ? `The agent was sent back to fix problems ${repairs} time(s) in this attempt.` : "";
  }
  async function refresh() {
    const data = await request("/admin/work");
    workspaces = data.workspaces; jobs = data.jobs;
    const picker = $("#workspace-picker");
    if (JSON.stringify([...picker.options].map(o => [o.value, o.textContent])) !== JSON.stringify([["", "Choose a project"], ...workspaces.map(w => [w.id, w.name])])) {
      picker.replaceChildren(new Option("Choose a project", ""), ...workspaces.map(w => new Option(w.name, w.id)));
      picker.value = projectId;
    }
    $("#allow-commands").disabled = !data.commands_available;
    $("#sandbox-note").textContent = data.commands_available ? "Commands need a working Linux sandbox. They have no network, GPU, or host home access." : "OS command isolation is unavailable here. File-only jobs are supported.";
    if (!workspaces.length) $("#project-editor").open = true;
    renderJobs(); await loadJob();
  }
  $("#verify-work").addEventListener("change", attemptSettings);
  $("#workspace-picker").addEventListener("change", event => selectProject(event.target.value));
  $("#new-workspace").addEventListener("click", () => { projectForm(); $("#project-editor").open = true; $("#workspace-name").focus(); });
  $("#workspace-form").addEventListener("submit", event => { event.preventDefault(); act(async () => {
    const body = { name: $("#workspace-name").value, root: $("#workspace-root").value, instructions: $("#workspace-instructions").value, model: $("#workspace-model").value, allow_edits: $("#allow-edits").checked, allow_commands: $("#allow-commands").checked, verify: $("#verify-work").checked, max_attempts: $("#verify-work").checked ? Number($("#max-attempts").value) : 1, time_budget_minutes: Number($("#time-budget-minutes").value) };
    const value = await request(editing ? `/admin/work/workspaces/${editing}` : "/admin/work/workspaces", body, editing ? "PUT" : "POST");
    await refresh(); selectProject(value.id); $("#project-editor").open = false;
  }); });
  $("#task-form").addEventListener("submit", event => { event.preventDefault(); act(async () => {
    const value = await request("/admin/work/jobs", { workspace_id: projectId, task: $("#task").value });
    selected = value.id; cursor = 0; reviewedRevision = null; $("#acknowledge-uncertain").checked = false; $("#activity").replaceChildren(); $("#diff").textContent = "";
    $("#task").value = ""; await refresh();
  }); });
  $("#review-changes").addEventListener("click", () => act(async () => {
    const data = await request(`/admin/work/jobs/${selected}/changes`);
    $("#change-summary").textContent = data.files.length ? `${data.files.length} changed file(s): ${data.files.join(", ")}` : "No file changes.";
    $("#diff").textContent = data.diff;
    reviewedRevision = data.files.length ? data.revision : null;
    await loadJob();
  }));
  for (const [selector, approved] of [["#approve-command", true], ["#deny-command", false]]) $(selector).addEventListener("click", () => act(async () => {
    await request(`/admin/work/jobs/${selected}/approve`, { call_id: job.pending[0].id, approved }); await refresh();
  }));
  $("#work-auth").addEventListener("submit", event => { event.preventDefault(); token = $("#work-token").value; sessionStorage.setItem("arc-work-token", token); $("#work-auth").hidden = true; act(init); });
  $("#theme-toggle").addEventListener("click", () => { const light = document.documentElement.dataset.theme !== "light"; document.documentElement.dataset.theme = light ? "light" : "dark"; $("#theme-toggle").textContent = light ? "Dark theme" : "Light theme"; });
  async function init() {
    const response = await fetch("/admin/session-token");
    if (response.ok) token = (await response.json()).admin_token || token;
    const data = await request("/v1/models");
    $("#workspace-model").replaceChildren(...data.data.filter(m => m.owned_by === "arc-llama" && (!m.metadata?.capabilities || m.metadata.capabilities.includes("chat"))).map(m => new Option(m.metadata?.display_name || m.id, m.id)));
    await refresh();
    if (workspaces.length) selectProject(workspaces[0].id);
  }
  act(init);
  setInterval(() => { if (!document.hidden && !busy) act(refresh); }, 2000);
})();
