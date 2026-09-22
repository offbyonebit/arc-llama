(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.ArcMarkdownSafety = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function safeUrl(value, allowMail = true) {
    if (typeof value !== "string") return null;
    const url = value.trim();
    if (!url || /[\u0000-\u001f\u007f]/.test(url)) return null;
    if (/^(?:#|\/|\.\/|\.\.\/|\?)/.test(url)) return url;
    try {
      const parsed = new URL(url, "http://arc-llama.local/");
      const allowed = allowMail
        ? ["http:", "https:", "mailto:"]
        : ["http:", "https:"];
      return allowed.includes(parsed.protocol) ? url : null;
    } catch (_) {
      return null;
    }
  }

  function link(href, title, text) {
    const url = safeUrl(href, true);
    if (!url) return `<span class="unsafe-link">${text}</span>`;
    const titleAttr = title ? ` title="${escapeHtml(title)}"` : "";
    return `<a href="${escapeHtml(url)}"${titleAttr} rel="noopener noreferrer">${text}</a>`;
  }

  function image(href, title, alt) {
    const url = safeUrl(href, false);
    if (!url) return escapeHtml(alt || "");
    const titleAttr = title ? ` title="${escapeHtml(title)}"` : "";
    return `<img src="${escapeHtml(url)}" alt="${escapeHtml(alt || "")}"${titleAttr}>`;
  }

  return { escapeHtml, safeUrl, link, image };
});
