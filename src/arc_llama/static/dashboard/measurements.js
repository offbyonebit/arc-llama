// Dashboard measurements/history controller. Loaded as a classic script; it
// performs no requests until the coordinator explicitly calls start/poll.
window.ArcDashboard = window.ArcDashboard || {};
window.ArcDashboard.createMeasurementsController = function createMeasurementsController({
  document: doc, request, authHeaders, intervalMs = 15000,
}) {
  // ---------------------------------------------------------------------------
  // Measurements panel: real, bounded per-model usage measurements from
  // /admin/metrics. Never invents throughput: the panel renders only what the
  // server recorded from actual traffic.
  // ---------------------------------------------------------------------------

  function fmtSeconds(value) {
    if (value == null) return "n/a";
    if (value >= 10) return `${value.toFixed(0)}s`;
    if (value >= 1) return `${value.toFixed(1)}s`;
    return `${Math.round(value * 1000)}ms`;
  }

  function timingRow(label, summary, unit = "") {
    if (!summary) return null;
    const item = doc.createElement("div");
    item.className = "measure-row";
    const term = doc.createElement("span");
    term.textContent = label;
    const detail = doc.createElement("strong");
    const fmt = unit === " tok/s"
      ? (v) => v == null ? "n/a" : Number(v).toFixed(1)
      : fmtSeconds;
    const suffix = unit === " tok/s" ? "tok_s" : "s";
    detail.textContent = `median ${fmt(summary[`median_${suffix}`])}${unit} · p95 ${fmt(summary[`p95_${suffix}`])}${unit} · last ${fmt(summary[`last_${suffix}`])}${unit} (${summary.count})`;
    item.append(term, detail);
    return item;
  }

  // Hourly generation-speed medians from /admin/metrics/history, by model.
  let generationHistory = new Map();

  function sparkline(points) {
    const values = points.map((p) => p.median);
    const min = Math.min(...values), max = Math.max(...values);
    const width = 160, height = 32, span = max - min || 1;
    const step = width / (values.length - 1);
    const coords = values.map((v, i) => `${(i * step).toFixed(1)},${(height - 2 - (v - min) / span * (height - 4)).toFixed(1)}`);
    const ns = "http://www.w3.org/2000/svg";
    const svg = doc.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(height));
    svg.setAttribute("class", "measure-sparkline");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `Generation speed trend: ${min.toFixed(1)} to ${max.toFixed(1)} tok/s`);
    const line = doc.createElementNS(ns, "polyline");
    line.setAttribute("points", coords.join(" "));
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", "currentColor");
    line.setAttribute("stroke-width", "1.5");
    svg.appendChild(line);
    return svg;
  }

  function historyRow(name) {
    const points = generationHistory.get(name) || [];
    if (points.length < 2) return null;
    const row = doc.createElement("div");
    row.className = "measure-row measure-trend";
    const label = doc.createElement("span");
    const days = Math.max(1, Math.round((points[points.length - 1].t - points[0].t) / 86400));
    label.textContent = `Generation trend (${points.length} hours over ${days} day${days === 1 ? "" : "s"})`;
    const latest = doc.createElement("strong");
    latest.textContent = `now ${points[points.length - 1].median.toFixed(1)} tok/s`;
    row.append(label, sparkline(points), latest);
    return row;
  }

  function renderMeasurements(metrics) {
    const host = doc.querySelector("#measurements");
    if (!host) return;
    host.replaceChildren();
    const timings = metrics?.timings || {};
    const timingModels = timings?.models || {};
    const entries = Object.entries(timingModels);
    const queue = timings?.queue_wait;
    const tuned = (metrics?.autotune?.models || []).filter(m => m.before_after);

    const card = doc.createElement("div");
    card.className = "measurements-card";
    if (!entries.length && !queue && !tuned.length) {
      const empty = doc.createElement("p");
      empty.className = "measurements-empty";
      empty.textContent = "No measurements yet. Send chat messages and load models; real timings appear here.";
      card.appendChild(empty);
      host.appendChild(card);
      return;
    }
    for (const [name, entry] of entries) {
      const block = doc.createElement("div");
      block.className = "measure-block";
      const title = doc.createElement("h3");
      title.textContent = name;
      block.appendChild(title);
      const rows = doc.createElement("div");
      rows.className = "measure-rows";
      if (entry.cold_start) rows.appendChild(timingRow("Cold start", entry.cold_start));
      if (entry.ttft) rows.appendChild(timingRow("Time to first token", entry.ttft));
      if (entry.model_wait) rows.appendChild(timingRow("Model wait (load/switch included)", entry.model_wait));
      if (entry.generation_tok_s) rows.appendChild(timingRow("Generation speed", entry.generation_tok_s, " tok/s"));
      const trend = historyRow(name);
      if (trend) rows.appendChild(trend);
      block.appendChild(rows);
      card.appendChild(block);
    }
    if (queue) card.appendChild(timingRow("Model wait across requests (load/switch included)", queue));
    for (const model of tuned) {
      const result = model.before_after;
      const block = doc.createElement("div");
      block.className = "measure-block";
      const title = doc.createElement("h3");
      title.textContent = `${model.name} · last completed autotune this session`;
      block.appendChild(title);
      for (const [key, label] of [["prompt_tok_s", "Prompt processing"], ["generation_tok_s", "Generation"]]) {
        const before = result.before?.[key], after = result.after?.[key];
        if (!Number.isFinite(before) || !Number.isFinite(after)) continue;
        const row = doc.createElement("p");
        row.textContent = `${label}: ${before.toFixed(1)} → ${after.toFixed(1)} tok/s (${result.applied ? "applied" : "measured only"})`;
        block.appendChild(row);
      }
      card.appendChild(block);
    }
    host.appendChild(card);
  }

  let pollRequest = null;
  function poll() {
    if (!pollRequest) pollRequest = pollMeasurements().finally(() => { pollRequest = null; });
    return pollRequest;
  }
  async function pollMeasurements() {
    try {
      const [response, history] = await Promise.all([
        request("/admin/metrics", { headers: authHeaders() }),
        request("/admin/metrics/history?metric=generation_tok_s&days=30", { headers: authHeaders() }).catch(() => null),
      ]);
      if (!response.ok) throw new Error(`status ${response.status}`);
      if (history && history.ok) {
        const grouped = new Map();
        for (const point of (await history.json()).points || []) {
          if (!grouped.has(point.model)) grouped.set(point.model, []);
          grouped.get(point.model).push(point);
        }
        generationHistory = grouped;
      }
      renderMeasurements(await response.json());
    } catch (_) {
      // Measurements are best-effort; keep whatever was rendered.
    }
  }

  let timer = null;
  let started = false;
  function schedule() {
    clearTimeout(timer);
    timer = null;
    if (started) timer = setTimeout(async () => { await poll(); schedule(); }, doc.hidden ? Math.max(30000, intervalMs) : intervalMs);
  }
  async function visibilityChanged() {
    if (!started) return;
    clearTimeout(timer);
    if (!doc.hidden) await poll();
    schedule();
  }
  return {
    poll,
    start() {
      if (started) return;
      started = true;
      doc.addEventListener("visibilitychange", visibilityChanged);
      schedule();
    },
    stop() {
      started = false;
      doc.removeEventListener("visibilitychange", visibilityChanged);
      clearTimeout(timer);
      timer = null;
    },
  };
};
