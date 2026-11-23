/* Thanksgiving easter egg: a turkey crosses the top once per browser, only in Thanksgiving week. */
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
    turkey.animate(frames, { duration: 6000, easing: 'linear' }).onfinish = () => turkey.remove();
  }, 1200);
})(typeof window !== 'undefined' ? window : globalThis);
