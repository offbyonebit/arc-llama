// Guided frontend connection and copy-only integration dialog controller.
window.ArcDashboard = window.ArcDashboard || {};
window.ArcDashboard.createIntegrationController = function createIntegrationController({
  document: doc, request, authHeaders, browserNavigator, scheduleTimeout = setTimeout,
}) {
  let integrationLoaded = false;
  let activeFrontendTab = "openwebui";

  // Connect-a-frontend guided panel. The endpoint returns only locally computed
  // discovery data (base URL from the configured host/port, loopback Ollama
  // reachability, registered upstreams); the panel is copy-only and never
  // mutates anything on the user's side or ours.
  const frontendTabs = [
    ["#tab-openwebui", "#panel-openwebui"],
    ["#tab-ollama", "#panel-ollama"],
    ["#tab-generic", "#panel-generic"],
  ];

  function isWindows() {
    return browserNavigator.platform && /win/i.test(browserNavigator.platform);
  }

  async function copyToClipboard(text) {
    try {
      await browserNavigator.clipboard.writeText(text);
      return true;
    } catch (_) {
      // Older or non-secure contexts may lack the async clipboard API.
      try {
        const helper = doc.createElement("textarea");
        helper.value = text;
        helper.setAttribute("readonly", "true");
        helper.style.position = "fixed";
        helper.style.opacity = "0";
        doc.body.appendChild(helper);
        helper.select();
        const ok = doc.execCommand("copy");
        helper.remove();
        return ok;
      } catch (_) {
        return false;
      }
    }
  }

  function markCopied(buttonNode) {
    const original = buttonNode.textContent;
    buttonNode.textContent = "Copied";
    buttonNode.classList.add("copied");
    scheduleTimeout(() => {
      buttonNode.textContent = original;
      buttonNode.classList.remove("copied");
    }, 1500);
  }

  function bindCopyButton(id, getText) {
    const node = doc.querySelector(id);
    if (!node) return;
    node.addEventListener("click", async () => {
      if (await copyToClipboard(getText())) markCopied(node);
    });
  }

  function selectFrontendTab(kind) {
    activeFrontendTab = kind;
    for (const [tabId, panelId] of frontendTabs) {
      const tab = doc.querySelector(tabId);
      const panel = doc.querySelector(panelId);
      const active = tabId.includes(kind);
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
      panel.toggleAttribute("hidden", !active);
    }
  }

  function renderOllamaStatus(ollama) {
    const status = doc.querySelector("#ollama-status");
    const registered = doc.querySelector("#ollama-registered-note");
    if (!ollama?.reachable) {
      status.textContent =
        "No Ollama found on this machine (checked its default address). " +
        "The command below still works later; run it once Ollama is installed and running.";
      status.classList.add("unreachable");
    } else {
      status.textContent = `Ollama detected${ollama.version ? ` (version ${ollama.version})` : ""}.`;
      status.classList.remove("unreachable");
    }
    registered.hidden = !ollama?.already_registered;
  }

  function renderIntegration(data) {
    const openwebuiUrl = doc.querySelector("#openwebui-url");
    const genericUrl = doc.querySelector("#generic-url");
    const genericCurl = doc.querySelector("#generic-curl");
    const ollamaCommand = doc.querySelector("#ollama-command");
    const portHint = doc.querySelector("#frontend-port-hint");
    if (!openwebuiUrl || !data) return;
    openwebuiUrl.textContent = data.base_url;
    genericUrl.textContent = data.base_url;
    genericCurl.textContent = isWindows() ? data.curl_example_windows : data.curl_example;
    ollamaCommand.textContent = data.ollama?.upstream_add_command || "";
    portHint.textContent = `port ${data.server?.port}, at /v1/chat/completions and /v1/models`;
    const lanNote = doc.querySelector("#openwebui-lan-note");
    if (data.lan_note) {
      lanNote.textContent = data.lan_note;
      lanNote.hidden = false;
    } else {
      lanNote.hidden = true;
    }
    const apiKeyNote = doc.querySelector("#openwebui-key-note");
    if (data.api_key_guidance) {
      apiKeyNote.textContent = `API Key: ${data.api_key_guidance}`;
    }
    renderOllamaStatus(data.ollama);
    integrationLoaded = true;
  }

  async function loadIntegration(force = false) {
    if (integrationLoaded && !force) return;
    try {
      const response = await request("/admin/integration", { headers: authHeaders() });
      if (!response.ok) throw new Error(`status ${response.status}`);
      renderIntegration(await response.json());
    } catch (_) {
      renderIntegration(null);
    }
  }

  function openFrontendDialog() {
    doc.querySelector("#frontend-dialog").showModal();
    loadIntegration();
  }

  let bound = false;
  function bind() {
    if (bound) return;
    bound = true;
    doc.querySelector("#connect-frontend")?.addEventListener("click", openFrontendDialog);
    doc.querySelector("#frontend-close")?.addEventListener("click", () => doc.querySelector("#frontend-dialog").close());
    for (const [tabId] of frontendTabs) {
      doc.querySelector(tabId)?.addEventListener("click", () => selectFrontendTab(tabId.split("-")[1]));
    }
    bindCopyButton("#copy-openwebui-url", () => doc.querySelector("#openwebui-url")?.textContent);
    bindCopyButton("#copy-ollama-command", () => doc.querySelector("#ollama-command")?.textContent);
    bindCopyButton("#copy-generic-url", () => doc.querySelector("#generic-url")?.textContent);
    bindCopyButton("#copy-generic-curl", () => doc.querySelector("#generic-curl")?.textContent);
  }

  return { bind, open: openFrontendDialog, load: loadIntegration };
};
