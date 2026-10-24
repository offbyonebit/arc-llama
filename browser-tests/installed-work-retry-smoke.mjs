// Installed wheel + real local inference. Faults are actual copy-only I/O failures;
// no API or model responses are mocked and source files are never faulted.
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

const origin = process.argv[2], root = process.env.ARC_WORK_SMOKE_ROOT;
const mode = process.env.ARC_WORK_NATIVE_CASE || 'retry';
assert.ok(origin?.startsWith('http://127.0.0.1:'));
assert.ok(root?.startsWith('/tmp/arc-work-'));
assert.ok(['normal', 'retry', 'keep-best'].includes(mode));
await mkdir(root, { recursive: true });
assert.deepEqual(await readdir(root), [], 'Use a fresh empty /tmp/arc-work-* project folder');
await writeFile(path.join(root, 'README.txt'), 'Original project\n', {flag:'wx'});
for (const name of ['A.txt', 'B.txt']) await assert.rejects(stat(path.join(root, name)), {code:'ENOENT'});
const options = { headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] };
const browser = await chromium.launch({...options,channel:'chrome'}).catch(() => chromium.launch(options));
let last = null;
const faults = [], injected = new Set(), errors = [];
const protocolPauses = [];
const reportPath = `/tmp/arc-work-inference-${mode}-${process.env.ARC_WORK_SMOKE_MODEL_SIZE || "0.5b"}-report.json`;
try {
  const page = await browser.newPage();
  page.on('pageerror', e => errors.push(e.message));
  const api = async (route, body) => {
    const response = await page.request.fetch(origin + route, {method:body?'POST':'GET', headers:{Authorization:`Bearer ${process.env.ARC_LLAMA_SMOKE_TOKEN}`}, ...(body?{data:body}:{})});
    assert.equal(response.ok(), true, `${route}: ${await response.text()}`);
    return response.json();
  };
  await page.goto(origin + '/work');
  await page.waitForSelector('#workspace-model option', {state:'attached'});
  await page.locator('#project-editor').evaluate(node => { node.open = true; });
  await page.click('#new-workspace');
  await page.fill('#workspace-name', `Native inference ${mode}`);
  await page.fill('#workspace-root', root);
  await page.selectOption('#workspace-model', 'published-smoke');
  await page.fill('#workspace-instructions', 'Use the file tools to make the requested changes.');
  await page.fill('#max-attempts', mode === 'normal' ? '1' : '2');
  await page.fill('#time-budget-minutes', '10');
  const workspacePromise = page.waitForResponse(r => r.url().endsWith('/admin/work/workspaces') && r.request().method()==='POST');
  await page.click('#save-workspace');
  const workspace = await (await workspacePromise).json();
  assert.equal(workspace.verify, true);
  assert.equal(workspace.max_attempts, mode === 'normal' ? 1 : 2);
  const task = 'Create A.txt with alpha and B.txt with beta. Both files must exist and contain the specified text.';
  await page.fill('#task', task);
  const jobPromise = page.waitForResponse(r => r.url().endsWith('/admin/work/jobs') && r.request().method()==='POST');
  await page.click('#start-job');
  const created = await (await jobPromise).json();
  const start = Date.now();
  let lastProgress = 0;
  while (Date.now() - start < 400000) {
    last = await api(`/admin/work/jobs/${created.id}`);
    if (mode !== 'normal' && last.state === 'running' && !injected.has(last.attempt)) {
      assert.notEqual(path.resolve(last.project_copy), path.resolve(root));
      assert.ok(path.resolve(last.project_copy).startsWith('/tmp/arc-work-'));
      const blocked = last.attempt === 1 ? ['B.txt'] : mode === 'keep-best' ? ['A.txt','B.txt'] : [];
      for (const name of blocked) {
        const target = path.join(last.project_copy, name);
        assert.equal(path.dirname(last.project_copy), path.dirname(created.project_copy));
        await mkdir(target); // Writing a file at this path must now fail with EISDIR.
        faults.push({attempt:last.attempt,path:name,fault:'directory blocks file write'});
      }
      injected.add(last.attempt);
    }
    if (Date.now() - lastProgress > 10000) {
      console.log(JSON.stringify({mode,state:last.state,attempt:last.attempt,turns:last.turns,repairs:last.repairs}));
      lastProgress = Date.now();
    }
    if (last.state === 'paused' && protocolPauses.length < 2 && /Unterminated string|Invalid tool|Expecting|empty answer/.test(last.message)) {
      protocolPauses.push({attempt:last.attempt, message:last.message, pending:last.pending});
      console.log(JSON.stringify({mode, explicit_admin_resume:last.message}));
      await api(`/admin/work/jobs/${created.id}/resume`, {});
      await new Promise(resolve => setTimeout(resolve, 100));
      continue;
    }
    if (['completed','failed','paused','interrupted'].includes(last.state)) break;
    await new Promise(resolve => setTimeout(resolve, 40));
  }
  assert.equal(last.state, 'completed', JSON.stringify(last));
  assert.ok(last.criteria.some(c => c.path === 'A.txt'), 'No machine criterion for A.txt');
  assert.ok(last.criteria.some(c => c.path === 'B.txt'), 'No machine criterion for B.txt');
  if (mode === 'normal') {
    assert.equal(last.attempts.length, 1);
    assert.equal(last.chosen_attempt, 1);
    assert.equal(last.verification.verdict, 'verified');
  } else {
    assert.equal(last.attempts.length, 2);
    assert.equal(last.attempts[0].verdict, 'failed');
    assert.equal(last.attempts[0].repairs, 2);
    if (mode === 'retry') {
      assert.equal(last.chosen_attempt, 2);
      assert.equal(last.attempts[1].verdict, 'verified');
    } else {
      assert.equal(last.chosen_attempt, 1);
      assert.equal(last.attempts[1].verdict, 'failed');
      assert.ok(last.attempts[0].passed > last.attempts[1].passed);
    }
    assert.ok(last.events.some(e => e.type==='tool_result' && e.error), 'No real tool failure recorded');
  }
  const expected = mode === 'keep-best' ? ['A.txt'] : ['A.txt','B.txt'];
  for (const name of expected) {
    assert.equal((await readFile(path.join(last.project_copy,name),'utf8')).trim(), name==='A.txt'?'alpha':'beta');
    await assert.rejects(stat(path.join(root,name)), {code:'ENOENT'});
  }
  await page.reload();
  await page.waitForFunction(id => !!document.querySelector(`#workspace-picker option[value="${id}"]`), workspace.id);
  await page.selectOption('#workspace-picker', workspace.id);
  await page.locator(`[data-job-id="${created.id}"]`).click();
  await page.waitForFunction(n => document.querySelector('#attempt-progress').textContent.includes(`Kept attempt ${n}`), last.chosen_attempt);
  if (mode === 'keep-best') assert.match(await page.textContent('#verdict'), /Failed verification/);
  else assert.match(await page.textContent('#verdict'), /Verified: checked by code/);
  await page.click('#review-changes');
  await page.waitForFunction(() => document.querySelector('#diff').textContent.includes('+alpha'));
  await page.getByRole('button', {name:'Apply reviewed changes'}).click();
  await page.getByRole('button', {name:'Undo applied changes'}).waitFor();
  for (const name of expected) assert.equal((await readFile(path.join(root,name),'utf8')).trim(),name==='A.txt'?'alpha':'beta');
  if (mode === 'keep-best') await assert.rejects(stat(path.join(root,'B.txt')), {code:'ENOENT'});
  await page.getByRole('button', {name:'Undo applied changes'}).click();
  await page.getByRole('button', {name:'Apply reviewed changes'}).waitFor();
  for (const name of expected) await assert.rejects(stat(path.join(root,name)), {code:'ENOENT'});
  assert.equal((await readFile(path.join(root,'README.txt'),'utf8')).trim(),'Original project');
  assert.deepEqual(errors, []);
  await page.screenshot({path:`/tmp/arc-work-inference-${mode}-${process.env.ARC_WORK_SMOKE_MODEL_SIZE || "0.5b"}.png`, fullPage:true});
  const report = {mode,real_inference:true,model:`Qwen2.5-${process.env.ARC_WORK_SMOKE_MODEL_SIZE || '0.5b'}-Instruct-Q4_K_M`,backend:'CPU',faults,protocol_pauses:protocolPauses,elapsed_seconds:(Date.now()-start)/1000,criteria:last.criteria,attempts:last.attempts,chosen_attempt:last.chosen_attempt,verification:last.verification,tool_results:last.events.filter(e=>e.type==='tool_result'),reload:'pass',reviewed_apply:'pass',undo:'pass',source_preserved:'pass'};
  await writeFile(reportPath,JSON.stringify(report,null,2));
  console.log(JSON.stringify({mode,result:'pass',attempts:last.attempts,chosen:last.chosen_attempt,report:reportPath}));
} catch (error) {
  await writeFile(reportPath,JSON.stringify({mode,result:'failed',faults,protocol_pauses:protocolPauses,error:String(error),last},null,2));
  throw error;
} finally { await browser.close(); }
