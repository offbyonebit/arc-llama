// First-answer coordinator. Reads never start downloads, loads or inference.
window.ArcDashboard = window.ArcDashboard || {};
window.ArcDashboard.createOnboardingController = function ({ document: doc, request, authHeaders, chooseAnother, scanExisting, refreshStatus }) {
  const host = doc.querySelector("#first-answer");
  const section = doc.querySelector(".first-answer-section");
  const next = doc.querySelector(".next-step-section");
  const $ = name => host?.querySelector(`[data-setup="${name}"]`);
  const busyPhases = ["checking", "runtime", "downloading", "registering", "loading"];
  let data = null, snapshot = null, online = false, pending = null, timer = null, unavailable = false, localError = "", mutating = false;
  let lastPoll = 0, sequence = 0, statusKnown = false;
  const key = "arc-llama.first-answer.start";
  const mib = value => `${(Math.max(0, Number(value) || 0) / 1048576).toFixed(0)} MiB`;
  const size = value => `${(value / 1024).toFixed(1)} GiB`;
  const busy = () => busyPhases.includes(data?.phase);
  function remember(value) { try { value ? sessionStorage.setItem(key, value) : sessionStorage.removeItem(key); } catch { /* optional storage */ } }
  function remembered() { try { return sessionStorage.getItem(key); } catch { return null; } }
  function render() {
    if (!host) return;
    const show = !unavailable && data && (!snapshot?.models?.length || data.phase !== "idle");
    section.hidden = !show;
    if (next) next.classList.toggle("setup-coordinated", !!show);
    doc.querySelector("#choose-title")?.closest("section")?.classList.toggle("setup-coordinated", !!show && !snapshot?.models?.length);
    if (!show) return;
    const gpu = data.gpu, starter = data.starter;
    const discovering = data.discovery?.status === "searching";
    const discoveryLabels = {
      searching: "Searching the current public model catalog…",
      no_match: "No suitable starter found in the current catalog search. Retry, compare models in Discover, or scan for an existing download.",
      unavailable: "The public model catalog is unavailable. Check your connection and retry, or use Discover.",
    };
    $("hardware").textContent = gpu
      ? `Your device: ${gpu.name || gpu.pci_slot}${gpu.vram_mb ? ` · ${size(gpu.vram_mb)}` : " · memory capacity unavailable"}`
      : "No Intel GPU is configured yet.";
    $("model").textContent = starter ? starter.name : discovering ? "Finding a model for your card…" : "Choose a model";
    $("description").textContent = starter
      ? `A chat model from the current public catalog. Download: approximately ${size(starter.size_mb)}. Estimated to fit with conversation memory; actual use varies. Ranked by catalog popularity and estimated fit, not benchmarked quality.`
      : discoveryLabels[data.discovery?.status] || "Run arc-llama doctor to check drivers and permissions. If GPU memory is unknown, use Discover to compare models.";
    const source = $("source"); source.hidden = !starter;
    if (starter) { source.href = starter.source; source.textContent = `Model details · License: ${starter.license || "Not stated"}`; }
    const phase = data.phase;
    $("scan").hidden = busy();
    $("scan").disabled = !online || mutating;
    $("title").textContent = phase === "ready" ? "Ready for your first question"
      : phase === "error" ? "Let’s finish your setup"
      : phase === "paused" ? "Setup paused" : "Let’s get your first answer";
    const labels = { idle: "Download and start", checking: "Checking the current model files…", runtime: "Preparing the runtime…", downloading: "Downloading your model…", registering: "Adding your model…", loading: "Loading your model…", ready: "Your model is loaded. Open Chat and ask a question.", paused: "The current step finished safely. Continue whenever you’re ready.", error: "Setup stopped. Check the message below, then retry." };
    $("status").textContent = (!online || !statusKnown) ? "Connection lost. Setup may still be running; reconnect before continuing."
      : data.pause_requested && busy() ? "Pausing after this step. The current download or load will finish; no question will be sent."
      : labels[phase] || "Checking setup…";
    $("status").hidden = phase === "idle" && online && statusKnown;
    const progress = $("progress"), bytes = $("bytes");
    const job = data.download;
    const count = phase === "runtime" ? data.runtime_bytes_done : job?.bytes_done;
    const total = phase === "runtime" ? data.runtime_bytes_total : job?.bytes_total;
    progress.hidden = !["runtime", "downloading"].includes(phase);
    if (!progress.hidden && total > 0) { progress.max = total; progress.value = Math.min(count || 0, total * .99); }
    else progress.removeAttribute("value");
    bytes.textContent = !progress.hidden && count > 0
      ? `${mib(count)}${total > 0 ? ` of ${mib(total)} · ${Math.min(99, Math.floor(count / total * 100))}%` : " downloaded"}` : "";
    $("steps").querySelectorAll("li").forEach(node => {
      const order = ["runtime", "downloading", "loading", "ready"];
      const current = phase === "registering" ? 1 : order.indexOf(phase);
      node.dataset.current = String(current === order.indexOf(node.dataset.phase));
    });
    const error = $("error"); error.textContent = localError || data.error || data.discovery?.error || ""; error.hidden = !error.textContent;
    const start = $("start");
    start.hidden = phase === "ready" || busy();
    start.disabled = mutating || !online || !statusKnown || (discovering || (!starter && !!gpu && !gpu.vram_mb));
    start.textContent = !gpu ? "Check hardware again" : phase === "error" ? "Retry setup" : phase === "paused" ? "Continue setup" : !starter ? discovering ? "Searching…" : "Retry model search" : "Download and start";
    const pause = $("pause"); pause.hidden = !busy(); pause.disabled = mutating || !online || !statusKnown || data.pause_requested;
    const chat = $("chat"); chat.hidden = phase !== "ready" || !data.chat_url || !online || !statusKnown;
    if (!chat.hidden) chat.href = data.chat_url;
    $("consent").hidden = phase !== "idle";
    if (phase === "ready" && data.model && remembered() === starter?.id && online && statusKnown) {
      remember(null);
      location.assign(data.chat_url);
    }
  }
  async function poll(force = false) {
    if (document.hidden || unavailable || (!force && Date.now() - lastPoll < 1500)) return pending;
    if (pending) return pending;
    const mine = sequence;
    lastPoll = Date.now();
    pending = (async () => {
      try {
        const response = await request("/admin/onboarding", { headers: authHeaders() });
        if (mine !== sequence) return;
        if (response.status === 404) { unavailable = true; return; }
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Setup status is unavailable.");
        data = result; statusKnown = true;
        if (localError.startsWith("Setup status is unavailable")) localError = "";
      } catch { statusKnown = false; localError = "Setup status is unavailable. Check your connection and try again."; }
      finally { if (mine === sequence) render(); }
    })().finally(() => { pending = null; });
    return pending;
  }
  async function act(action) {
    if (mutating || !online) return;
    mutating = true; localError = ""; sequence++;
    if (action === "start") remember(data?.starter?.id);
    render();
    try {
      const response = await request("/admin/onboarding", { method: "POST", headers: authHeaders({ "Content-Type": "application/json" }), body: JSON.stringify({ action, starter: data?.starter?.id }) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Setup could not start.");
      data = result; statusKnown = true;
      if (action === "pause") remember(null);
      if (action === "detect") await refreshStatus();
    } catch (error) { localError = error.message; if (action === "start") remember(null); }
    finally { mutating = false; render(); }
  }
  $("start")?.addEventListener("click", () => act(!data?.gpu ? "detect" : data?.starter ? "start" : "refresh"));
  $("pause")?.addEventListener("click", () => act("pause"));
  $("another")?.addEventListener("click", chooseAnother);
  $("scan")?.addEventListener("click", scanExisting);
  return {
    update(value, connected) {
      snapshot = value; online = connected;
      // Read once after a reload even if registration already finished: the
      // live server can still have a paused or failed setup awaiting Continue.
      const ongoing = !value?.models?.length || remembered() || (data && data.phase !== "idle");
      if (online && (ongoing || !data)) poll();
      if (ongoing && online) {
        if (!timer) timer = setInterval(() => { if (online && !document.hidden) poll(); }, 2000);
      } else if (timer) { clearInterval(timer); timer = null; }
      render();
    }, poll,
  };
};
