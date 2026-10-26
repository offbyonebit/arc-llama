// Welcome prompts are suggestions only; a click fills an empty composer.
window.ArcFirstAnswer = {
  begin({ input, selectedModel }) {
    if (location.pathname !== "/chat" || new URLSearchParams(location.search).get("welcome") !== "1") return;
    const panel = document.querySelector("#first-question");
    if (!panel || !selectedModel) return;
    document.body.classList.add("first-answer-welcome");
    this.model = selectedModel;
    panel.hidden = false;
    panel.querySelectorAll("button[data-prompt]").forEach(button => button.addEventListener("click", () => {
      if (input.value.trim()) return;
      input.value = button.dataset.prompt;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.focus();
    }));
    input.focus();
  },
  completed(model, text, evidence) {
    if (!this.model || model !== this.model || !text?.trim()) return;
    const panel = document.querySelector("#first-question");
    panel.querySelector("h2").textContent = "Your first answer is here";
    panel.querySelector("p").textContent = [`Reply from ${model}`, evidence?.configuredDevice && `Configured device: ${evidence.configuredDevice}`, evidence?.configuredBackend && `Configured backend: ${evidence.configuredBackend}`].filter(Boolean).join(" · ");
    panel.querySelector(".first-question-prompts").hidden = true;
    panel.classList.add("answered");
    const url = new URL(location.href); url.searchParams.delete("welcome");
    window.history.replaceState(null, "", url.pathname + url.search + url.hash);
    document.body.classList.remove("first-answer-welcome");
    this.model = null;
  },
};
