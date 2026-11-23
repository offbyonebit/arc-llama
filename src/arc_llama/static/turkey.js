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
  const TURKEY = '/assets/turkey-walk.png?v=turkey-2', CORN = '/assets/cornucopia.png?v=turkey-2';
  const art = new Image(); art.src = TURKEY; new Image().src = CORN;
  root.setTimeout(() => (art.decode ? art.decode() : Promise.resolve()).catch(() => {}).then(() => {
    // A 6-frame walk strip (left-facing) at 56 px, flipped through at 9 fps while it crosses.
    const turkey = document.createElement('div');
    turkey.className = 'arc-turkey';
    turkey.setAttribute('aria-hidden', 'true');
    turkey.style.cssText = `position:fixed;top:4px;left:0;width:56px;height:56px;background:url("${TURKEY}") 0 0/336px 56px no-repeat;z-index:2147483647;pointer-events:none`;
    document.body.append(turkey);
    if (typeof turkey.animate !== 'function') { turkey.remove(); return; }
    const width = root.innerWidth, frames = [];
    for (let i = 0; i <= 12; i++) frames.push({ transform: `translate(${Math.round(width - i / 12 * (width + 64))}px, ${i % 2 ? -4 : 0}px)` });
    turkey.animate([{ backgroundPosition: '0 0' }, { backgroundPosition: '-336px 0' }], { duration: 667, easing: 'steps(6)', iterations: Infinity });
    turkey.animate(frames, { duration: 6000, easing: 'linear' }).onfinish = () => { turkey.remove(); greet(); };
  }), 1200);
  // Then a little harvest note with a horn of plenty, faded in and out.
  function greet() {
    const note = document.createElement('div');
    note.className = 'arc-turkey-note';
    note.setAttribute('aria-hidden', 'true');
    note.innerHTML = `<img src="${CORN}" width="36" height="36" alt=""><strong>Happy Thanksgiving!</strong>`;
    note.style.cssText = 'position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:2147483647;pointer-events:none;user-select:none;display:flex;align-items:center;gap:8px;padding:4px 16px 4px 8px;border-radius:999px;font:600 14px/1.2 system-ui,sans-serif;background:var(--wnb,#FBF3E3);color:var(--wn,#A35A06);border:1px solid var(--wnl,#EBD3A8);box-shadow:0 4px 14px var(--sh,rgba(20,30,60,.22))';
    document.body.append(note);
    note.animate([{ opacity: 0 }, { opacity: 1, offset: 0.1 }, { opacity: 1, offset: 0.85 }, { opacity: 0 }], { duration: 4500 }).onfinish = () => note.remove();
  }
})(typeof window !== 'undefined' ? window : globalThis);
