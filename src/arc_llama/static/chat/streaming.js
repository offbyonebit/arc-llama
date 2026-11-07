/* Bounded SSE framing for Chat completions. No requests or DOM work on import. */
window.ArcChat = window.ArcChat || {};
window.ArcChat.createCompletionStream = function (onChunk) {
  const limit = 1024 * 1024; // UTF-16 code units per line/event, including data separators.
  const decoder = new TextDecoder();
  let lineParts = [], lineSize = 0, data = [], dataSize = 0;
  let skipLF = false, done = false;
  const isObject = value => value !== null && typeof value === "object" && !Array.isArray(value);
  const optionalText = value => value == null || typeof value === "string";
  const invalid = () => { throw new Error("The server sent an invalid completion event. Your partial answer was kept."); };

  function dispatch() {
    if (!data.length) return;
    const payload = data.join("\n");
    data = []; dataSize = 0;
    if (payload === "[DONE]") { done = true; return; }
    let chunk;
    try { chunk = JSON.parse(payload); }
    catch (_) { throw new Error("The server sent malformed streaming JSON. Your partial answer was kept."); }
    if (!isObject(chunk)) invalid();
    if (chunk.error) {
      throw new Error(typeof chunk.error.message === "string" ? chunk.error.message : "Generation was interrupted.");
    }
    if (chunk.usage != null) {
      if (!isObject(chunk.usage)) invalid();
      for (const key of ["prompt_tokens", "completion_tokens", "total_tokens"]) {
        if (chunk.usage[key] != null && (!Number.isSafeInteger(chunk.usage[key]) || chunk.usage[key] < 0)) invalid();
      }
    }
    if (chunk.choices == null && chunk.usage == null) invalid();
    if (chunk.choices !== undefined && !Array.isArray(chunk.choices)) invalid();
    for (const choice of chunk.choices || []) {
      if (!isObject(choice) || !optionalText(choice.finish_reason)) invalid();
      if (choice.index !== undefined && (!Number.isSafeInteger(choice.index) || choice.index < 0)) invalid();
      if (choice.delta !== undefined && !isObject(choice.delta)) invalid();
      const delta = choice.delta || {};
      for (const key of ["role", "content", "reasoning_content", "reasoning"]) {
        if (!optionalText(delta[key])) invalid();
      }
      if (delta.tool_calls != null) {
        if (!Array.isArray(delta.tool_calls)) invalid();
        for (const tool of delta.tool_calls) {
          if (!isObject(tool) || !optionalText(tool.id) || !optionalText(tool.type)) invalid();
          if (tool.index !== undefined && (!Number.isSafeInteger(tool.index) || tool.index < 0)) invalid();
          if (tool.function !== undefined) {
            if (!isObject(tool.function) || !optionalText(tool.function.name) || !optionalText(tool.function.arguments)) invalid();
          }
        }
      }
    }
    onChunk(chunk);
  }

  function line(value) {
    if (value === "") { dispatch(); return; }
    if (value.startsWith(":")) return;
    const colon = value.indexOf(":");
    const field = colon < 0 ? value : value.slice(0, colon);
    let content = colon < 0 ? "" : value.slice(colon + 1);
    if (content.startsWith(" ")) content = content.slice(1);
    if (field !== "data") return;
    dataSize += content.length + 1;
    if (dataSize > limit) throw new Error("The server sent a streaming event that is too large. Your partial answer was kept.");
    data.push(content);
  }

  function append(value) {
    lineSize += value.length;
    if (lineSize > limit) throw new Error("The server sent a streaming line that is too large. Your partial answer was kept.");
    lineParts.push(value);
  }

  function consume(text) {
    let start = 0;
    for (let i = 0; i < text.length && !done; i++) {
      const char = text[i];
      if (skipLF) {
        skipLF = false;
        if (char === "\n") { start = i + 1; continue; }
      }
      if (char !== "\r" && char !== "\n") continue;
      append(text.slice(start, i));
      const complete = lineParts.join("");
      lineParts = []; lineSize = 0;
      skipLF = char === "\r";
      start = i + 1;
      line(complete);
    }
    if (!done && start < text.length) append(text.slice(start));
  }

  return {
    get done() { return done; },
    push(bytes) { if (!done) consume(decoder.decode(bytes, { stream: true })); },
    end() {
      if (!done) consume(decoder.decode());
      // SSE never dispatches an unterminated event at EOF; finish_reason alone
      // also cannot establish that the transport delivered the final event.
      if (!done) throw new Error("The stream ended before completion. Your partial answer was kept.");
    },
  };
};
