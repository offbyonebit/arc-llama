import { openChat, choosePaperModel, ADMIN_STATUS } from './harness.mjs';
const json = body => ({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });

export async function testRuntimeRollback({ page, origin }) {
  const build = tag => ({ build: tag, path: `/runtime/${tag}/llama-server`, available: true,
    installed_at: tag === 'b100' ? null : 1700000000, date_source: tag === 'b100' ? null : 'update history' });
  let current = 'b300', previous = 'b200', busy = null, mode = 'success';
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const calls = [];
  const history = () => ({ current: build(current), previous: previous ? build(previous) : null,
    builds: ['b300', 'b200', 'b100'].map(build), busy_reason: busy });
  await page.route('**/admin/runtime/history', route => route.fulfill(json(history())));
  await page.route('**/admin/runtime/rollback', async route => {
    calls.push({ body: route.request().postDataJSON(), auth: route.request().headers().authorization });
    if (mode === 'refuse') return route.fulfill({ ...json({ detail: 'A tune is running. Finish or stop it before rolling back.' }), status: 409 });
    await gate;
    const from = build(current).path;
    current = previous;
    previous = current === 'b200' ? 'b100' : null;
    return route.fulfill(json({ from, to: build(current).path, restarted: mode === 'partial' ? [] : ['qwen'],
      restart_errors: mode === 'partial' ? [{ model: 'qwen', message: 'controlled reload failure' }] : [], history: history() }));
  });
  await page.goto(`${origin}/#system`);
  await page.waitForFunction(() => document.querySelector('#runtime-current').textContent.includes('b300'));
  if (!(await page.locator('#runtime-builds').innerText()).includes('Install date not recorded')) throw new Error('unknown install date was fabricated');
  await page.click('#runtime-rollback');
  if (!(await page.locator('#runtime-confirm').innerText()).includes('Loaded models will stop and restart')) throw new Error('restart warning missing');
  await page.click('#runtime-confirm-cancel');
  if (calls.length) throw new Error('cancel initiated rollback');
  await page.click('#runtime-rollback');
  const request = page.waitForRequest('**/admin/runtime/rollback');
  await page.click('#runtime-confirm-yes');
  await request;
  if (!(await page.locator('#runtime-rollback').isDisabled()) || !(await page.locator('#runtime-action-status').innerText()).includes('Stopping loaded models')) throw new Error('rollback progress was not shown');
  if (calls[0].body.expected_previous !== '/runtime/b200/llama-server' || !calls[0].auth?.startsWith('Bearer ')) throw new Error('rollback target/auth missing');
  release();
  await page.waitForFunction(() => document.querySelector('#runtime-action-status').textContent.includes('Restarted 1'));
  await page.waitForFunction(() => document.querySelector('#runtime-current').textContent.includes('b200'));
  busy = 'A download is running.';
  await page.evaluate(() => loadRuntimeHistory());
  if (!(await page.locator('#runtime-rollback').isDisabled()) || !(await page.locator('#runtime-busy-reason').innerText()).includes('download')) throw new Error('busy history did not disable rollback with a reason');
  busy = null;
  mode = 'refuse';
  await page.evaluate(() => loadRuntimeHistory());
  await page.click('#runtime-rollback');
  await page.click('#runtime-confirm-yes');
  await page.waitForFunction(() => document.querySelector('#runtime-action-status').textContent.includes('A tune is running'));
  if (current !== 'b200') throw new Error('refused rollback changed the runtime');
  mode = 'partial';
  await page.click('#runtime-rollback');
  await page.click('#runtime-confirm-yes');
  await page.waitForFunction(() => document.querySelector('#runtime-action-status').textContent.includes('Some models could not restart'));
  await page.waitForFunction(() => document.querySelector('#runtime-current').textContent.includes('b100'));
  if (!(await page.locator('#runtime-rollback').isHidden())) throw new Error('rollback button remained without an earlier runtime');
}

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
