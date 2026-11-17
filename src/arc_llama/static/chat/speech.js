(function (root) {
  "use strict";
  const ArcChat = root.ArcChat = root.ArcChat || {};
  ArcChat.createSpeech = function ({ document: doc, chatLog, settings }) {
    const synth = root.speechSynthesis;
    const KEY = "arc-llama-local-voice";
    let preferred = "", current = null, started = false;
    try { preferred = root.localStorage.getItem(KEY) || ""; } catch (_) {}

    function voices() {
      try { return synth ? synth.getVoices().filter(voice => voice.localService === true) : []; }
      catch (_) { return []; }
    }
    function label(button, active) {
      const text = active ? "Stop reading" : "Read aloud";
      if (button.textContent !== text) button.textContent = text;
      button.setAttribute("aria-pressed", String(active));
    }
    function stop() {
      const old = current; current = null;
      if (old) { label(old.button, false); synth.cancel(); }
    }
    function replyText(message) {
      const content = message.querySelector(".content");
      if (!content) return "";
      const clone = content.cloneNode(true);
      for (const node of clone.querySelectorAll("pre")) node.replaceWith(doc.createTextNode(" code block. "));
      for (const node of clone.querySelectorAll(".thinking-block, .thinking-content, button, script, style, [aria-hidden=true]")) node.remove();
      for (const node of clone.querySelectorAll("p, h1, h2, h3, h4, h5, h6, li, tr, td, th, blockquote, br")) node.append(doc.createTextNode(" "));
      return (clone.textContent || "").replace(/\s+/g, " ").trim();
    }
    function read(message, button) {
      if (current?.button === button) { stop(); return; }
      stop();
      // Recheck at the click: a saved preference or changed voice list never selects a network voice.
      const local = voices();
      const voice = local.find(v => v.voiceURI === preferred) || local[0];
      const text = replyText(message);
      if (!voice || !text || !root.SpeechSynthesisUtterance) { refresh(); return; }
      const utterance = new root.SpeechSynthesisUtterance(text);
      utterance.voice = voice;
      utterance.lang = voice.lang;
      current = { message, button, utterance };
      label(button, true);
      const ended = () => { if (current?.utterance === utterance) { current = null; label(button, false); } };
      utterance.onend = ended; utterance.onerror = ended;
      try { synth.speak(utterance); } catch (_) { ended(); }
    }
    function refresh() {
      const local = voices();
      if (current && (!current.message.isConnected || !local.some(v => v.voiceURI === current.utterance.voice.voiceURI)
          || chatLog.querySelector(".live-activity"))) stop();
      for (const message of chatLog.querySelectorAll(".message.assistant")) {
        let button = message.querySelector(".read-aloud");
        if (!local.length || !root.SpeechSynthesisUtterance || message.querySelector(".live-activity") || (!button && !replyText(message))) {
          button?.remove(); continue;
        }
        if (!button) {
          button = doc.createElement("button"); button.type = "button"; button.className = "read-aloud";
          label(button, false); button.addEventListener("click", () => read(message, button));
          message.append(button);
        }
      }
    }
    function picker() {
      if (!settings) return;
      const local = voices();
      settings.hidden = !local.length;
      settings.replaceChildren();
      if (!local.length) return;
      const title = doc.createElement("label"); title.htmlFor = "speech-voice"; title.textContent = "Read-aloud voice";
      const select = doc.createElement("select"); select.id = "speech-voice";
      for (const voice of local) {
        const option = doc.createElement("option"); option.value = voice.voiceURI;
        option.textContent = `${voice.name} (${voice.lang})`; select.append(option);
      }
      select.value = local.some(v => v.voiceURI === preferred) ? preferred : local[0].voiceURI;
      select.addEventListener("change", () => {
        if (!voices().some(v => v.voiceURI === select.value)) { picker(); refresh(); return; }
        preferred = select.value;
        try { root.localStorage.setItem(KEY, preferred); } catch (_) {}
        stop();
      });
      const note = doc.createElement("p"); note.className = "s-note"; note.textContent = "Only voices installed on this device are offered.";
      settings.append(title, select, note);
    }
    function start() {
      if (started || !synth || !root.SpeechSynthesisUtterance) return;
      started = true;
      synth.addEventListener("voiceschanged", () => { picker(); refresh(); });
      new MutationObserver(refresh).observe(chatLog, { childList: true, subtree: true });
      root.addEventListener("beforeunload", stop);
      picker(); refresh();
    }
    return { start, stop };
  };
})(window);
