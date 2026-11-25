/* Once per browser, in December only, a short decorative animation along the top of the page. */

/*
 * 01010111 01101000 01100101 01101110 00100000 01110100 01101000 01100101
 * 01111001 00100000 01110011 01100001 01110111 00100000 01110100 01101000
 * 01100101 00100000 01110011 01110100 01100001 01110010 00101100 00100000
 * 01110100 01101000 01100101 01111001 00100000 01110010 01100101 01101010
 * 01101111 01101001 01100011 01100101 01100100 00100000 01110111 01101001
 * 01110100 01101000 00100000 01100101 01111000 01100011 01100101 01100101
 * 01100100 01101001 01101110 01100111 00100000 01100111 01110010 01100101
 * 01100001 01110100 00100000 01101010 01101111 01111001 00101110
 */

(function (root) {
  'use strict';
  const KEY = 'arc-llama-ambient-seen';
  const isDecember = now => now.getMonth() === 11;
  const shouldRun = (now, storage, nav, reducedMotion) => {
    if (!isDecember(now) || nav.webdriver || reducedMotion) return false;
    try { return storage.getItem(KEY) === null; } catch (_) { return false; }
  };
  root.ArcAmbient = { isDecember, shouldRun, KEY };
  if (!root.document) return;
  const reduced = typeof root.matchMedia === 'function' && root.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const now = new Date();
  // Never share the screen with the November one in years where that week reaches December 1.
  if (root.ArcTurkey && root.ArcTurkey.isThanksgivingWeek(now)) return;
  if (!shouldRun(now, root.localStorage, root.navigator, reduced)) return;
  try { root.localStorage.setItem(KEY, String(now.getFullYear())); } catch (_) { return; }

  const INK = '#3B2414';
  let points = '';
  for (let i = 0; i < 16; i++) {
    const r = i % 4 === 0 ? 18 : i % 4 === 2 ? 10.5 : 5, a = (i * 22.5 - 90) * Math.PI / 180;
    points += `${(20 + r * Math.cos(a)).toFixed(1)},${(20 + r * Math.sin(a)).toFixed(1)} `;
  }
  const STAR = `<svg width="132" height="40" viewBox="0 0 132 40"><defs>
    <linearGradient id="arc-am-t"><stop offset="0" stop-color="#FFE9A8" stop-opacity=".95"/><stop offset="1" stop-color="#FFE9A8" stop-opacity="0"/></linearGradient>
    <radialGradient id="arc-am-g"><stop offset="0" stop-color="#FFF6D0" stop-opacity=".9"/><stop offset="1" stop-color="#FFD866" stop-opacity="0"/></radialGradient></defs>
    <path d="M20 15 L132 20 L20 25 Z" fill="url(#arc-am-t)"/><circle cx="20" cy="20" r="19" fill="url(#arc-am-g)"/>
    <polygon points="${points}" fill="#FFD447" stroke="#9A6A08" stroke-width="1.4" stroke-linejoin="round"/>
    <circle cx="20" cy="20" r="3.2" fill="#FFFDF2"/></svg>`;
  // Facing left, toward the star. [robe, trim, skin, beard, gift]
  const kings = [
    ['#7B3FA8', '#F2C14E', '#F1C9A5', '#F4F1EA', '<rect x="1.5" y="33" width="10" height="7" rx="1.2" fill="#F2C14E" stroke="' + INK + '" stroke-width="1.2"/><path d="M1.5 35.5h10" stroke="#B07A0C"/>'],
    ['#C0392B', '#F2C14E', '#C68B5E', '', '<path d="M3 41 Q2 35 6.5 33 Q11 35 10 41 Z" fill="#E7D7B5" stroke="' + INK + '" stroke-width="1.2"/><rect x="5" y="30.5" width="3" height="3" fill="#B07A0C" stroke="' + INK + '" stroke-width="1"/>'],
    ['#1F7A5A', '#F2C14E', '#7A4A2B', '#2A1A10', '<path d="M2.5 34 h8 l-1 7 h-6 Z" fill="#8C5A3C" stroke="' + INK + '" stroke-width="1.2"/><ellipse cx="6.5" cy="34" rx="4.6" ry="1.4" fill="#F2C14E" stroke="' + INK + '" stroke-width="1"/>'],
  ].map(([robe, trim, skin, beard, gift]) => `<svg width="50" height="66" viewBox="0 0 44 58" style="overflow:visible">
    <ellipse cx="23" cy="56" rx="13" ry="2" fill="rgba(0,0,0,.16)"/>
    <path d="M11 55 L15 28 Q23 21 31 28 L36 55 Z" fill="${robe}" stroke="${INK}" stroke-width="1.5" stroke-linejoin="round"/>
    <path d="M11.4 52 L35.6 52 L36 55 L11 55 Z" fill="${trim}" stroke="${INK}" stroke-width="1"/>
    <ellipse cx="13" cy="55.5" rx="4" ry="1.8" fill="${INK}"/>
    <circle cx="23" cy="20" r="7" fill="${skin}" stroke="${INK}" stroke-width="1.5"/>
    <circle cx="16.4" cy="21" r="1.7" fill="${skin}" stroke="${INK}" stroke-width="1"/><circle cx="19" cy="18.6" r="1" fill="${INK}"/>
    ${beard ? `<path d="M17 23 Q16.5 32 23 33 Q28.5 30 28.5 23 Q23 26 17 23 Z" fill="${beard}" stroke="${INK}" stroke-width="1.2"/>` : ''}
    <path d="M16 14 L16 7 L19.2 10.2 L23 5 L26.8 10.2 L30 7 L30 14 Z" fill="#F6C343" stroke="#7A4E00" stroke-width="1.2" stroke-linejoin="round"/>
    <circle cx="23" cy="11.2" r="1.3" fill="#D63A3A"/>
    ${gift}<path d="M17 29 Q9 33 9.5 38 L13.5 38.5 Q16 34.5 21 33.5 Z" fill="${robe}" stroke="${INK}" stroke-width="1.3" stroke-linejoin="round"/>
    <circle cx="10" cy="37.6" r="1.9" fill="${skin}" stroke="${INK}" stroke-width="1"/></svg>`);

  root.setTimeout(() => {
    const width = root.innerWidth, base = 'position:fixed;left:0;z-index:2147483647;pointer-events:none;';
    const star = document.createElement('div');
    star.className = 'arc-ambient-star';
    star.setAttribute('aria-hidden', 'true');
    star.style.cssText = base + 'top:0;width:132px;height:40px';
    star.innerHTML = STAR;
    const train = document.createElement('div');
    train.className = 'arc-ambient-train';
    train.setAttribute('aria-hidden', 'true');
    train.style.cssText = base + 'top:10px;display:flex;gap:2px;transform:translateX(' + width + 'px)';
    train.innerHTML = kings.join('');
    const snow = document.createElement('div');
    snow.className = 'arc-ambient-snow';
    snow.setAttribute('aria-hidden', 'true');
    snow.style.cssText = base + 'top:0;width:100%;height:100%;overflow:hidden;z-index:2147483646';
    document.body.append(snow, star, train);
    if (typeof star.animate !== 'function') { snow.remove(); star.remove(); train.remove(); return; }

    // The star streaks right to left on a shallow arc, twinkling, and leaves first.
    const arc = [];
    for (let i = 0; i <= 10; i++) arc.push({ transform: `translate(${Math.round(width - i / 10 * (width + 150))}px, ${Math.round(14 * (1 - Math.sin(i / 10 * Math.PI)) - 4)}px)` });
    star.firstChild.animate([{ transform: 'scale(1)' }, { transform: 'scale(1.12)' }], { duration: 380, direction: 'alternate', iterations: Infinity });
    star.animate(arc, { duration: 3600, easing: 'ease-in' }).onfinish = () => star.remove();

    // Three kings follow at a walking pace, each bobbing a little out of step.
    [...train.children].forEach((k, i) => (k.style.transformOrigin = '50% 100%') && k.animate(
      [{ transform: 'translateY(0) rotate(-2deg)' }, { transform: 'translateY(-3px) rotate(0deg)' }, { transform: 'translateY(0) rotate(2deg)' }],
      { duration: 520, delay: i * 170, direction: 'alternate', iterations: Infinity, easing: 'ease-in-out' }));
    train.animate([{ transform: `translateX(${width}px)` }, { transform: 'translateX(-150px)' }], { duration: 9000, delay: 700, easing: 'linear', fill: 'both' }).onfinish = () => train.remove();

    // Snow falls behind them for about ten seconds.
    const height = root.innerHeight;
    for (let i = 0; i < 34; i++) {
      const f = document.createElement('span'), size = 9 + Math.random() * 9, fall = 5500 + Math.random() * 3000;
      f.textContent = '❄︎';
      f.style.cssText = `position:absolute;top:-24px;left:${(Math.random() * 100).toFixed(1)}%;font:${size.toFixed(0)}px/1 system-ui,sans-serif;color:#A9CBEA;text-shadow:0 0 2px rgba(60,90,130,.35);opacity:.9`;
      snow.append(f);
      const drift = Math.round((Math.random() - 0.5) * 80);
      f.animate([{ transform: 'translate(0,0) rotate(0deg)', opacity: 0 }, { opacity: 0.9, offset: 0.1 },
        { transform: `translate(${drift}px,${height + 48}px) rotate(${drift > 0 ? 180 : -180}deg)`, opacity: 0.6 }],
      { duration: fall, delay: Math.random() * (10000 - fall), easing: 'linear', fill: 'both' });
    }
    root.setTimeout(() => snow.remove(), 10400);
  }, 1200);
})(typeof window !== 'undefined' ? window : globalThis);
