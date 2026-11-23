/* Thanksgiving easter egg: once per browser, only in Thanksgiving week, a turkey crosses the top and says happy Thanksgiving. */

/*
 * 01000001 01110010 01101001 01110011 01100101 00101100 00100000 01110011
 * 01101000 01101001 01101110 01100101 00111011 00100000 01100110 01101111
 * 01110010 00100000 01110100 01101000 01111001 00100000 01101100 01101001
 * 01100111 01101000 01110100 00100000 01101001 01110011 00100000 01100011
 * 01101111 01101101 01100101 00101100 00100000 01100001 01101110 01100100
 * 00100000 01110100 01101000 01100101 00100000 01100111 01101100 01101111
 * 01110010 01111001 00100000 01101111 01100110 00100000 01110100 01101000
 * 01100101 00100000 01001100 01001111 01010010 01000100 00100000 01101001
 * 01110011 00100000 01110010 01101001 01110011 01100101 01101110 00100000
 * 01110101 01110000 01101111 01101110 00100000 01110100 01101000 01100101
 * 01100101 00101110
 */

(function (root) {
  'use strict';
  const KEY = 'arc-llama-turkey-seen';
  // US Thanksgiving is the 4th Thursday of November (day of month, 22-28).
  const thanksgiving = year => 22 + (11 - new Date(year, 10, 1).getDay()) % 7;
  // Monday through Sunday of Thanksgiving week, local time.
  const isThanksgivingWeek = now => {
    const day = thanksgiving(now.getFullYear());
    return now >= new Date(now.getFullYear(), 10, day - 3) && now < new Date(now.getFullYear(), 10, day + 4);
  };
  const shouldRun = (now, storage, nav, reducedMotion) => {
    if (!isThanksgivingWeek(now) || nav.webdriver || reducedMotion) return false;
    try { return storage.getItem(KEY) === null; } catch (_) { return false; }
  };
  root.ArcTurkey = { thanksgiving, isThanksgivingWeek, shouldRun, KEY };
  if (!root.document) return;
  const reduced = typeof root.matchMedia === 'function' && root.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!shouldRun(new Date(), root.localStorage, root.navigator, reduced)) return;
  try { root.localStorage.setItem(KEY, String(new Date().getFullYear())); } catch (_) { return; }
  root.setTimeout(() => {
    const turkey = document.createElement('span');
    turkey.textContent = '\u{1F983}';
    turkey.setAttribute('aria-hidden', 'true');
    turkey.style.cssText = 'position:fixed;top:6px;left:0;font-size:28px;line-height:1;z-index:2147483647;pointer-events:none;user-select:none';
    document.body.append(turkey);
    if (typeof turkey.animate !== 'function') { turkey.remove(); return; }
    const width = root.innerWidth, frames = [];
    for (let i = 0; i <= 12; i++) frames.push({ transform: `translate(${Math.round(width - i / 12 * (width + 48))}px, ${i % 2 ? -4 : 0}px)` });
    turkey.animate(frames, { duration: 6000, easing: 'linear' }).onfinish = () => { turkey.remove(); greet(); };
  }, 1200);
  // Then a little harvest note: our own horn of plenty spilling fruit, faded in and out.
  function greet() {
    const note = document.createElement('div');
    note.className = 'arc-turkey-note';
    note.setAttribute('aria-hidden', 'true');
    note.innerHTML = '<svg viewBox="0 0 40 24" width="34" height="20"><path d="M2 21C7 9 18 3 31 3l5 17C24 17 12 19 2 21z" fill="#B7792B"/><path d="M12 15l6-9M20 16l5-11M28 17l3-12" stroke="#7A4A14" stroke-width="1.5"/></svg>'
      + '<span>\u{1F347}\u{1F34E}\u{1F350}\u{1F34A}</span><strong>Happy Thanksgiving!</strong>';
    note.style.cssText = 'position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:2147483647;pointer-events:none;user-select:none;display:flex;align-items:center;gap:8px;padding:6px 14px;border-radius:999px;font:600 14px/1.2 system-ui,sans-serif;background:var(--wnb,#FBF3E3);color:var(--wn,#A35A06);border:1px solid var(--wnl,#EBD3A8);box-shadow:0 4px 14px var(--sh,rgba(20,30,60,.22))';
    document.body.append(note);
    note.animate([{ opacity: 0 }, { opacity: 1, offset: 0.1 }, { opacity: 1, offset: 0.85 }, { opacity: 0 }], { duration: 4500 }).onfinish = () => note.remove();
  }
})(typeof window !== 'undefined' ? window : globalThis);
