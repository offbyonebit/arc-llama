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
