// Hugging Face downloads and local model storage controller.
window.ArcDashboard = window.ArcDashboard || {};
window.ArcDashboard.createLibraryController = function createLibraryController({
  document: doc, request, authHeaders, makeButton, getDisplayName, formatMb,
  writeMessage, getSnapshot, fetchStatus, reviewModel, showModelsView, onJobsChange = () => {},
}) {
  // ---------------------------------------------------------------------------
  // Model library. Every string from Hugging Face is set with textContent.
  // ---------------------------------------------------------------------------

  const FIT_LABELS = {
    fits: ["Likely to fit", "ready", "Estimated room for the model and conversation memory."],
    tight: ["Limited memory headroom", "warn", "A shorter conversation limit may be needed to leave enough memory."],
    too_big: ["Likely too large", "error", "The estimate exceeds this GPU's capacity. More memory or CPU offloading may be needed."],
    unknown: ["Fit unknown", "warn", "An estimate needs both the file size and GPU memory capacity."],
  };

  function compressionDescription(quant) {
    const code = String(quant || "unknown").toUpperCase();
    const bits = code.match(/^(?:I?Q)([1-8])(?:_|$)/)?.[1];
    if (bits) {
      const description = Number(bits) <= 3
        ? "Very compact. Strong compression reduces memory use, but can affect answer quality."
        : Number(bits) === 4
          ? "Uses less memory than an 8-bit copy of the same model. Compression can affect answer quality."
          : Number(bits) < 8
            ? "A middle ground between smaller and less compressed copies of the same model."
            : "Keeps more numerical detail than a 4-bit copy of the same model, and uses more memory.";
      return [`${bits}-bit compressed`, description];
    }
    if (/^(BF16|F16|FP16|F32|FP32)$/.test(code)) {
      return [`${code.includes("32") ? "32" : "16"}-bit precision`, "Less compressed; usually a larger download with higher memory use than quantized copies of the same model."];
    }
    return ["Compression not reported", "No plain-language compression estimate is available. Check the technical details before choosing."];
  }

  function modelSize(mb) {
    if (!(mb > 0)) return "Size not reported";
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GiB` : `${mb} MiB`;
  }

  let localCompatibility = null;
  const compatibilityHosts = new Map();
  function invalidateCompatibility() {
    const identity = getSnapshot()?.runtime_compatibility_identity;
    for (const [host, state] of compatibilityHosts) {
      if (!host.isConnected) { state.cancel(); compatibilityHosts.delete(host); continue; }
      if (state.identity !== identity) { state.reset(); state.identity = identity; }
    }
  }
  function compatibilityControl(target) {
    const snapshot = getSnapshot();
    const model = target.name && snapshot?.models?.find(item => item.name === target.name);
    const localKey = target.name ? JSON.stringify([target.name, snapshot?.runtime_compatibility_identity, model?.path, model?.file_readiness, model?.gpu_pci_slot]) : null;
    if (localKey && localCompatibility?.key === localKey) return localCompatibility.host;
    const host = doc.createElement("div");
    host.className = "library-compatibility";
    const message = doc.createElement("p");
    message.className = "plugin-meta";
    message.setAttribute("role", "status");
    message.textContent = "Runtime compatibility has not been checked.";
    const evidence = doc.createElement("details");
    evidence.className = "library-compatibility-evidence";
    evidence.hidden = true;
    const summary = doc.createElement("summary");
    summary.textContent = "What was checked";
    const explanation = doc.createElement("p");
    evidence.append(summary, explanation);
    let activeCheck = null;
    let revision = 0;
    const cancel = () => { revision++; activeCheck?.abort(); activeCheck = null; };
    const reset = () => {
      cancel();
      check.disabled = false;
      evidence.hidden = true;
      message.className = "plugin-meta";
      message.textContent = "Runtime changed. Check compatibility again.";
    };
    compatibilityHosts.set(host, { identity: snapshot?.runtime_compatibility_identity, reset, cancel });
    const check = makeButton("Check runtime compatibility", "secondary", async () => {
      const checkedIdentity = getSnapshot()?.runtime_compatibility_identity;
      const attempt = ++revision;
      const controller = new AbortController();
      activeCheck = controller;
      const timeout = setTimeout(() => controller.abort(), 30000);
      check.disabled = true;
      evidence.hidden = true;
      message.textContent = "Checking model architecture against the installed runtime…";
      try {
        const response = await request("/admin/library/compatibility", {
          method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify(target), signal: controller.signal,
        });
        const result = await response.json().catch(() => ({}));
        if (attempt !== revision) return;
        if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
        if (checkedIdentity !== getSnapshot()?.runtime_compatibility_identity) throw new Error("Runtime changed. Check again.");
        message.textContent = `${result.label}: ${result.detail} ${result.action}`;
        message.className = `plugin-meta${result.status === "incompatible" ? " library-error" : ""}`;
        explanation.textContent = [result.scope, result.source, result.runtime && `Runtime: ${result.runtime} (${result.runtime_fingerprint || "unknown identity"})`, result.backend && `Configured backend: ${result.backend}`, result.architecture && `Architecture: ${result.architecture}`, result.revision && `Model revision: ${result.revision}`].filter(Boolean).join(" · ");
        evidence.hidden = false;
      } catch (e) {
        if (attempt !== revision) return;
        message.textContent = `Compatibility unknown: ${e.name === "AbortError" ? "The check timed out. Try again." : e.message}`;
        message.className = "plugin-meta";
      } finally {
        clearTimeout(timeout);
        if (attempt === revision) { activeCheck = null; check.disabled = false; }
      }
    });
    host.append(check, message, evidence);
    if (localKey) localCompatibility = { key: localKey, host };
    return host;
  }

  async function librarySearch(event) {
    event.preventDefault();
    const host = doc.querySelector("#library-results");
    const query = doc.querySelector("#library-query").value.trim();
    if (query.length < 2) return;
    writeMessage(host, "Searching…");
    try {
      const r = await request(`/admin/library/search?q=${encodeURIComponent(query)}`, { headers: authHeaders() });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      host.replaceChildren();
      if (!data.results.length) { writeMessage(host, "No GGUF repositories matched."); return; }
      for (const item of data.results) {
        const card = doc.createElement("article");
        card.className = "plugin-card library-repo";
        const main = doc.createElement("div");
        const title = doc.createElement("h3");
        title.textContent = item.repo;
        const meta = doc.createElement("p");
        meta.className = "plugin-meta";
        meta.textContent = `${(item.downloads ?? 0).toLocaleString()} downloads · ${(item.likes ?? 0).toLocaleString()} likes`;
        const options = doc.createElement("div");
        options.className = "library-options";
        const identity = doc.createElement("div");
        identity.className = "library-repo-identity";
        const publisher = String(item.repo || "").split("/")[0];
        const avatar = doc.createElement("span");
        avatar.className = "library-publisher-avatar";
        avatar.setAttribute("aria-hidden", "true");
        avatar.textContent = publisher.slice(0, 2).toUpperCase() || "HF";
        // Derive a fixed HF URL from a namespace, never accept an arbitrary image URL.
        if (/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(publisher)) {
          const image = doc.createElement("img");
          image.alt = "";
          image.width = 36;
          image.height = 36;
          image.loading = "lazy";
          image.referrerPolicy = "no-referrer";
          image.addEventListener("error", () => image.remove(), { once: true });
          image.src = `https://huggingface.co/api/avatars/${encodeURIComponent(publisher)}`;
          avatar.appendChild(image);
        }
        const heading = doc.createElement("div");
        const byline = doc.createElement("p");
        byline.className = "plugin-meta library-publisher";
        byline.textContent = `Published by ${publisher || "an unknown account"} on Hugging Face`;
        byline.title = "The repository publisher may differ from the original model developer.";
        heading.append(title, byline);
        identity.append(avatar, heading);
        main.append(identity, meta, options);
        const side = doc.createElement("div");
        side.appendChild(makeButton("Compare versions", "secondary", () => libraryShowRepo(item.repo, options)));
        card.append(main, side);
        host.appendChild(card);
      }
    } catch (e) {
      writeMessage(host, `Search failed: ${e.message}`, "library-note error");
    }
  }

  async function libraryShowRepo(repo, host) {
    writeMessage(host, "Reading files…");
    try {
      const r = await request(`/admin/library/repo?repo=${encodeURIComponent(repo)}`, { headers: authHeaders() });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      host.replaceChildren();
      const estimate = doc.createElement("p");
      estimate.className = "plugin-meta";
      estimate.textContent = data.vram_mb > 0
        ? `GPU capacity used for estimates: ${modelSize(data.vram_mb)}. Fit includes an allowance for conversation memory; actual use depends on the model and settings.`
        : "GPU memory capacity is unavailable. Fit cannot be estimated yet.";
      host.appendChild(estimate);
      if (data.vision) {
        const v = doc.createElement("p");
        v.className = "plugin-meta";
        v.textContent = "Extra vision file: this repository includes an image adapter (projector). A matching adapter downloads automatically; its extra size is not included below. Review the pairing after download.";
        host.appendChild(v);
      }
      if (!data.options.length) { writeMessage(host, "No downloadable GGUF files."); return; }
      for (const option of data.options) {
        const row = doc.createElement("div");
        row.className = "library-option";
        const main = doc.createElement("div");
        main.className = "library-option-main";
        const [plainName, explanation] = compressionDescription(option.quant);
        const name = doc.createElement("strong");
        name.className = "library-file";
        name.textContent = plainName;
        const summary = doc.createElement("p");
        summary.className = "plugin-meta";
        summary.textContent = explanation;
        const size = doc.createElement("p");
        size.className = "library-option-size";
        size.textContent = `Download: ${modelSize(option.size_mb)}${option.shards > 1 ? ` total across ${option.shards} files` : ""}`;
        const [label, tone, fitExplanation] = FIT_LABELS[option.fit] || FIT_LABELS.unknown;
        const badge = doc.createElement("span");
        badge.className = `status-pill ${tone}`;
        badge.textContent = label;
        const fit = doc.createElement("p");
        fit.className = "plugin-meta";
        fit.textContent = `${fitExplanation} This is an estimate, not verified inference.`;
        const details = doc.createElement("details");
        details.className = "library-technical";
        const detailLabel = doc.createElement("summary");
        detailLabel.textContent = "Technical details";
        const filename = doc.createElement("p");
        filename.textContent = `File: ${option.file}`;
        const encoding = doc.createElement("p");
        encoding.textContent = `Encoding: ${option.quant || "unknown"}`;
        details.append(detailLabel, filename, encoding);
        const compatibility = compatibilityControl({ repo, file: option.file });
        main.append(name, summary, size, badge, fit, compatibility, details);
        const download = makeButton("Download", "secondary", async () => {
          download.disabled = true;
          try {
            const resp = await request("/admin/library/download", {
              method: "POST",
              headers: authHeaders({ "Content-Type": "application/json" }),
              body: JSON.stringify({ repo, file: option.file, size_mb: option.size_mb }),
            });
            const job = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(job.detail || `HTTP ${resp.status}`);
            poll({ force: true });
          } catch (e) {
            download.disabled = false;
            badge.textContent = `Download failed: ${e.message}`;
            badge.className = "status-pill error";
          }
        });
        if (option.fit === "too_big") download.title = "Estimated to exceed GPU memory. Review memory and offloading settings before loading.";
        row.append(main, download);
        host.appendChild(row);
      }
    } catch (e) {
      writeMessage(host, `Could not read ${repo}: ${e.message}`, "library-note error");
    }
  }

  const retryingDownloads = new Set();
  const supersededFailures = new Set();
  const retriedJobs = new Set();
  let knownJobs = [];

  function recoveryMessage(text, error = false) {
    const notice = doc.querySelector("#library-recovery-status");
    if (!notice) return;
    notice.hidden = !text;
    notice.textContent = text;
    notice.className = error ? "library-note error" : "section-copy";
    if (error) { notice.scrollIntoView({ behavior: "smooth", block: "center" }); notice.focus({ preventScroll: true }); }
  }

  async function retryDownload(job, control) {
    const key = JSON.stringify([job.repo, job.file]);
    if (retryingDownloads.has(key) || supersededFailures.has(job.id)) return false;
    retryingDownloads.add(key);
    if (control) control.disabled = true;
    recoveryMessage("Starting another attempt for the same file…");
    let accepted = false;
    try {
      const body = { repo: job.repo, file: job.file };
      if (Number.isFinite(job.bytes_total) && job.bytes_total >= 1048576) body.size_mb = Math.floor(job.bytes_total / 1048576);
      const response = await request("/admin/library/download", { method: "POST", headers: authHeaders({ "Content-Type": "application/json" }), body: JSON.stringify(body) });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
      accepted = true;
      supersededFailures.add(job.id);
      for (const previous of knownJobs) {
        if (previous.status === "error" && previous.repo === job.repo && previous.file === job.file) supersededFailures.add(previous.id);
      }
      if (result.id) retriedJobs.add(result.id);
      recoveryMessage("Retry queued. You can keep using the app while it downloads.");
      await poll({ force: true });
      return true;
    } catch (error) {
      recoveryMessage(accepted
        ? "The retry was queued, but its status could not be refreshed. Check downloads again."
        : `Could not start the retry: ${error.message}. Check your connection and try again.`, true);
      return false;
    } finally {
      retryingDownloads.delete(key);
      if (control && !accepted) control.disabled = false;
      // Polling may have replaced the clicked node while the submission waited.
      if (!accepted) await poll({ refreshStatus: false, force: true });
    }
  }

  let libraryJobTimer = null;
  let bound = false;
  let started = false;
  let pollRequest = null;
  let queuedPoll = null;
  let renderedJobsSignature = null;
  const refreshedCompletions = new Set();

  function schedulePoll(delay) {
    clearTimeout(libraryJobTimer);
    libraryJobTimer = setTimeout(poll, doc.hidden ? 30000 : delay);
  }

  function poll(options = {}) {
    if (pollRequest) {
      if (!options.force) return pollRequest;
      if (!queuedPoll) queuedPoll = pollRequest.then(() => poll({ ...options, force: false })).finally(() => { queuedPoll = null; });
      return queuedPoll;
    }
    clearTimeout(libraryJobTimer);
    pollRequest = pollJobs(options).finally(() => { pollRequest = null; });
    return pollRequest;
  }

  async function pollJobs({ refreshStatus = true } = {}) {
    const host = doc.querySelector("#library-jobs");
    if (!host) return;
    let jobs = [];
    try {
      const r = await request("/admin/library/jobs", { headers: authHeaders() });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      jobs = (await r.json()).jobs || [];
      if (!Array.isArray(jobs)) throw new Error("Invalid download status");
    } catch (_) {
      onJobsChange(null);
      writeMessage(host, "Download status unavailable. Check again to refresh it.", "library-note error");
      host.appendChild(makeButton("Check downloads again", "secondary", () => poll()));
      renderedJobsSignature = null;
      schedulePoll(5000);
      return;
    }
    knownJobs = jobs;
    for (const job of jobs) {
      if (!retriedJobs.has(job.id) || !["done", "error"].includes(job.status)) continue;
      retriedJobs.delete(job.id);
      recoveryMessage(job.status === "done" ? "Download finished. Review the model below." : "The download did not finish. You can retry again or choose another model.");
    }
    onJobsChange(jobs);
    const completed = jobs.filter(job => job.status === "done" && !refreshedCompletions.has(job.id));
    let retryCompletion = false;
    if (completed.length && refreshStatus) {
      if (await fetchStatus(true)) for (const job of completed) refreshedCompletions.add(job.id);
      else retryCompletion = true;
    }
    const active = jobs.some(job => ["queued", "downloading", "registering"].includes(job.status));
    if (active || retryCompletion) schedulePoll(active ? 2000 : 5000);
    const visibleJobs = jobs.filter(job => !supersededFailures.has(job.id));
    const signature = JSON.stringify([visibleJobs, [...retryingDownloads], (getSnapshot()?.models || []).map(model => [model.name, getDisplayName(model)])]);
    if (signature === renderedJobsSignature) return;
    host.replaceChildren();
    for (const job of visibleJobs.slice(0, 6)) {
      const row = doc.createElement("div");
      row.className = "library-job";
      const name = doc.createElement("span");
      name.textContent = `${job.repo} · ${job.file.split("/").pop()}`;
      const state = doc.createElement("strong");
      if (job.status === "downloading") {
        const pct = job.bytes_total ? Math.min(99, Math.round(job.bytes_done / job.bytes_total * 100)) : null;
        state.textContent = pct == null ? `downloading ${formatMb(Math.round(job.bytes_done / 1048576))}` : `downloading ${pct}%`;
      } else if (job.status === "done") {
        const registered = Array.isArray(job.registered) ? job.registered : [];
        state.textContent = registered.length ? "downloaded and added" : "downloaded (registration details unavailable)";
        const actions = doc.createElement("div");
        actions.className = "library-job-actions";
        if (!registered.length) {
          actions.appendChild(makeButton("View your models", "secondary", () => showModelsView("#choose-title")));
        } else {
          const visible = registered.filter((modelName) => getSnapshot()?.models?.some((model) => model.name === modelName));
          for (const modelName of visible) {
            const model = getSnapshot().models.find((item) => item.name === modelName);
            const review = makeButton(`Review ${getDisplayName(model)}`, "secondary", async () => {
              if (!(await reviewModel(modelName))) await poll();
            });
            review.setAttribute("aria-label", `Review model ${modelName}`);
            review.title = `Review ${getDisplayName(model)} (${modelName})`;
            actions.appendChild(review);
          }
          if (visible.length < registered.length) {
            const pending = doc.createElement("span");
            pending.className = "library-note";
            pending.textContent = "Registration is not visible yet. Refresh to review the downloaded model.";
            actions.appendChild(pending);
            actions.appendChild(makeButton("Refresh to review", "secondary", async (event) => {
              event.currentTarget.disabled = true;
              await fetchStatus(true);
              await poll({ refreshStatus: false });
            }));
          }
        }
        row.append(name, state, actions);
        host.appendChild(row);
        continue;
      } else if (job.status === "error") {
        state.textContent = "Download did not finish";
        state.className = "library-error";
        const actions = doc.createElement("div");
        actions.className = "library-job-actions";
        const retry = makeButton("Retry download", "secondary", event => retryDownload(job, event.currentTarget));
        retry.disabled = retryingDownloads.has(JSON.stringify([job.repo, job.file]));
        retry.setAttribute("aria-label", `Retry download ${job.file}`);
        const find = makeButton("Find another model", "ghost", () => showModelsView("#library-title"));
        const details = doc.createElement("details");
        details.className = "library-technical";
        const summary = doc.createElement("summary"); summary.textContent = "Technical details";
        const error = doc.createElement("pre"); error.textContent = job.error || "No error details were reported.";
        details.append(summary, error);
        actions.append(retry, find, details);
        row.append(name, state, actions);
        host.appendChild(row);
        continue;
      } else {
        state.textContent = job.status;
      }
      row.append(name, state);
      host.appendChild(row);
    }
    renderedJobsSignature = signature;
  }

  async function loadLibraryDisk() {
    const host = doc.querySelector("#library-disk");
    if (!host) return;
    try {
      const r = await request("/admin/library/disk", { headers: authHeaders() });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      host.replaceChildren();
      const summary = doc.createElement("p");
      summary.className = "plugin-meta";
      const used = data.models.reduce((n, m) => n + m.size_mb, 0);
      summary.textContent = `${formatMb(used)} in ${data.models.length} model(s) · ${formatMb(data.free_mb)} free in ${data.models_dir}`;
      host.appendChild(summary);
      for (const m of data.models) {
        const row = doc.createElement("div");
        row.className = "library-option";
        const name = doc.createElement("span");
        const when = m.last_used ? new Date(m.last_used * 1000).toLocaleDateString() : "no recorded use";
        name.textContent = `${m.name} · ${formatMb(m.size_mb)} · ${m.missing ? "file missing" : when}`;
        const remove = makeButton(m.managed ? "Delete" : "Unregister", "ghost", async () => {
          const verb = m.managed ? `Delete ${m.name} and its files` : `Unregister ${m.name} (files stay where they are)`;
          if (!confirm(`${verb}?`)) return;
          remove.disabled = true;
          const resp = await request(`/admin/library/models/${encodeURIComponent(m.name)}?delete_files=${m.managed}`, {
            method: "DELETE", headers: authHeaders(),
          });
          if (resp.ok) { await loadLibraryDisk(); fetchStatus(true); }
          else remove.disabled = false;
        });
        remove.title = m.managed ? "Files inside the models folder are deleted" : "Outside the models folder: only the registration is removed";
        row.append(name, remove);
        host.appendChild(row);
      }
    } catch (e) {
      writeMessage(host, `Disk usage unavailable: ${e.message}`, "library-note error");
    }
  }

  function bind() {
    if (bound) return;
    bound = true;
    doc.querySelector("#library-search")?.addEventListener("submit", librarySearch);
    doc.querySelector("#library-disk-details")?.addEventListener("toggle", (event) => {
      if (event.target.open) loadLibraryDisk();
    });
  }

  return {
    bind,
    compatibilityControl,
    invalidateCompatibility,
    start() {
      if (started) return;
      started = true;
      doc.addEventListener("visibilitychange", () => {
        clearTimeout(libraryJobTimer);
        if (!doc.hidden) poll();
        else schedulePoll(30000);
      });
      poll();
    },
    poll,
    retry: retryDownload,
    loadDisk: loadLibraryDisk,
  };
};
