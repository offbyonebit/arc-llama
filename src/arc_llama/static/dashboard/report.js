// "Report a problem" dialog. The server builds prefilled GitHub links locally
// (/admin/report-link); this file only shows the preview and sets the link
// targets. The user opens GitHub and submits the report themselves, so
// Arc Llama needs no online service and never holds a GitHub token.
(function reportProblem() {
  const dialog = document.querySelector("#report-dialog");
  const trigger = document.querySelector("#report-problem");
  if (!dialog || !trigger) return;

  const status = dialog.querySelector("#report-status");
  const preview = dialog.querySelector("#report-preview");
  const issueLink = dialog.querySelector("#report-issue");
  const discussionLink = dialog.querySelector("#report-discussion");

  function headers() {
    // authHeaders() is defined by app.js, which loads after this file but
    // before anyone can click the button.
    return typeof window.authHeaders === "function" ? window.authHeaders() : {};
  }

  async function fetchLink(kind) {
    const response = await fetch(`/admin/report-link?kind=${kind}`, { headers: headers() });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  async function open() {
    status.textContent = "Collecting details…";
    preview.hidden = true;
    dialog.showModal();
    try {
      const [issue, discussion] = await Promise.all([fetchLink("issue"), fetchLink("discussion")]);
      issueLink.href = issue.url;
      discussionLink.href = discussion.url;
      const p = issue.preview;
      preview.querySelector("code").textContent = [p.environment, "", p.gpu, "", p.log].join("\n");
      preview.hidden = false;
      status.textContent = issue.trimmed
        ? `Included the last ${issue.log_lines_included} log lines so the link stays short. You can edit everything on GitHub.`
        : "This is what will be filled in. You can edit everything on GitHub.";
    } catch (_) {
      // Keep the plain links from the markup so reporting still works.
      status.textContent = "Couldn't collect details from the server. The links below open an empty report.";
    }
  }

  trigger.addEventListener("click", open);
  dialog.querySelector("#report-close")?.addEventListener("click", () => dialog.close());
})();
