import { openChat, choosePaperModel, ADMIN_STATUS } from './harness.mjs';
const json = body => ({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });

export async function testPerModelRuntimeSettings({ page, origin }) {
  const runtime = new Map([['qwen', { llama_server: '/patched/ornith server', env: { LD_LIBRARY_PATH: '/patched/lib', FLAG: 'literal=value' } }], ['gemma', { llama_server: '', env: {} }]]);
  let editCount = 0;
  await page.route('**/admin/status', route => route.fulfill(json({ ...ADMIN_STATUS, models: ADMIN_STATUS.models.map(m => ({ ...m, ...runtime.get(m.name), effective_llama_server: runtime.get(m.name)?.llama_server || '/global/server' })) })));
  await page.route('**/admin/models/*/edit', route => {
    const name = new URL(route.request().url()).pathname.split('/')[3];
    const body = route.request().postDataJSON();
    runtime.set(name, { ...runtime.get(name), ...(Object.hasOwn(body, "llama_server") ? { llama_server: body.llama_server } : {}), ...(Object.hasOwn(body, "env") ? { env: body.env } : {}) });
    editCount++;
    return route.fulfill(json({ changed: ['llama_server', 'env'] }));
  });
  await openChat(page, origin);
  await page.click('#settings-toggle');
  await page.locator('.s-runtime summary').click();
  await page.waitForFunction(() => document.querySelector('#s-llama-server')?.value === '/patched/ornith server');
  if (await page.inputValue('#s-runtime-env') !== 'LD_LIBRARY_PATH=/patched/lib\nFLAG=literal=value') throw new Error('stored overrides not restored');
  await page.fill('#s-llama-server', '/new patched/server');
  await page.fill('#s-runtime-env', 'LD_LIBRARY_PATH=/new/lib\nFLAG=$(literal); value=1\nEMPTY=');
  await page.evaluate(() => document.activeElement.blur());
  await page.evaluate(() => fetchStatus());
  if (await page.inputValue('#s-llama-server') !== '/new patched/server') throw new Error('poll erased unsaved runtime');
  await page.click('#s-apply');
  await page.waitForFunction(() => document.querySelector('#s-feedback').textContent === 'Saved.');
  const saved = runtime.get('qwen');
  if (saved.llama_server !== '/new patched/server' || saved.env.FLAG !== '$(literal); value=1' || saved.env.EMPTY !== '') throw new Error('runtime values changed');
  if (runtime.get('gemma').llama_server !== '' || Object.keys(runtime.get('gemma').env).length) throw new Error('other model changed');
  await page.locator('.s-runtime summary').click();
  await page.fill('#s-runtime-env', 'INVALID');
  await page.click('#s-apply');
  if (editCount !== 1 || !(await page.locator('#s-feedback').innerText()).includes('KEY=value')) throw new Error('invalid env submitted');
  await page.fill('#s-llama-server', '');
  await page.fill('#s-runtime-env', '');
  await page.click('#s-apply');
  await page.waitForFunction(() => document.querySelector('#s-feedback').textContent === 'Saved.');
  if (runtime.get('qwen').llama_server !== '' || Object.keys(runtime.get('qwen').env).length) throw new Error('clearing did not restore defaults');
  await page.keyboard.press('Escape');
  await choosePaperModel(page, 'gemma');
  await page.click('#settings-toggle');
  await page.locator('.s-runtime summary').click();
  if (await page.inputValue('#s-llama-server') !== '' || await page.inputValue('#s-runtime-env') !== '') throw new Error('model switch leaked runtime settings');
}


export async function testRuntimeOverridesSurviveUnrelatedEdits({ page, origin }) {
  let edited;
  // A failed status fetch must not turn unavailable runtime fields into a reset.
  await page.route('**/admin/status', route => route.abort('failed'));
  await page.route('**/admin/models/*/edit', route => {
    edited = route.request().postDataJSON();
    return route.fulfill(json({}));
  });
  await openChat(page, origin);
  await page.click('#settings-toggle');
  await page.fill('#s-ctx', '16384');
  await page.click('#s-apply');
  await page.waitForFunction(() => document.querySelector('#s-feedback').textContent === 'Saved.');
  if (edited.ctx !== 16384 || Object.hasOwn(edited, 'env') || Object.hasOwn(edited, 'llama_server')) throw new Error('unrelated edit overwrote unobserved runtime settings');
}

export async function testPhoneNetworkAndManifest({ page, origin }) {
  let payload = { enabled: false, addresses: [], hint: 'Run arc-llama serve --lan to enable phone access.' };
  const headers = [];
  await page.route('**/admin/network', route => {
    headers.push(route.request().headers().authorization);
    return route.fulfill(json(payload));
  });
  await page.goto(`${origin}/#system`);
  await page.getByText('Run arc-llama serve --lan to enable phone access.', { exact: true }).waitFor();
  if (await page.locator('.network-qr svg').count()) throw new Error('Loopback-only server displayed QR');
  await page.evaluate(() => {
    const make = window.qrcode;
    window.qrInputs = [];
    window.qrcode = (...args) => {
      const qr = make(...args);
      const add = qr.addData;
      qr.addData = (text, ...rest) => { window.qrInputs.push(text); return add(text, ...rest); };
      return qr;
    };
  });
  payload = { enabled: true, hint: 'Connect your phone to this LAN or Tailscale.',
    admin_token: 'never-render-this-key', addresses: [
      { url: 'http://192.168.1.8:11437/chat', kind: 'lan', interface: 'Ethernet' },
      { url: 'http://100.101.102.103:11437/chat', kind: 'tailscale', interface: 'Tailscale' },
      { url: 'http://192.168.1.8:11437/chat?api_key=never-render-this-key', kind: 'lan' },
    ] };
  await page.click('#network-refresh');
  await page.locator('.network-qr svg').nth(1).waitFor();
  const links = await page.locator('.network-address a').evaluateAll(nodes => nodes.map(n => n.href));
  const inputs = await page.evaluate(() => window.qrInputs);
  if (JSON.stringify(links) !== JSON.stringify(inputs) || inputs.length !== 2) throw new Error('QR input differed from clean address');
  if (inputs.some(value => value.includes('?') || value.includes('key'))) throw new Error('QR contains credentials');
  if ((await page.locator('body').innerText()).includes('never-render-this-key')) throw new Error('Network panel rendered a key');
  if (!headers.every(value => value === 'Bearer browser-test-token')) throw new Error('Network request happened before token initialization');
  const manifest = await page.evaluate(async () => {
    const link = document.querySelector('link[rel="manifest"]');
    const response = await fetch(link.href);
    return { type: response.headers.get('content-type'), body: await response.json() };
  });
  if (!manifest.type.includes('application/manifest+json') || manifest.body.start_url !== '/chat' || manifest.body.display !== 'standalone') throw new Error('App manifest missing/wrong');
  await page.setViewportSize({width:390,height:844});
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Phone panel overflowed');
  await page.goto(`${origin}/chat`);
  if (await page.locator('link[rel="manifest"]').count() !== 1 || await page.locator('link[rel="apple-touch-icon"]').count() !== 1) throw new Error('Chat lacks install metadata');
}
