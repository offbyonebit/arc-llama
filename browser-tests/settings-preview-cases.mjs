import assert from 'node:assert/strict';
import { openChat, MODELS, ADMIN_STATUS } from './harness.mjs';
const json = body => ({status:200,contentType:'application/json',body:JSON.stringify(body)});
async function fixture(page, origin, loaded=true, reloading=false) {
  const calls=[]; let context=8192;
  await page.route('**/v1/models',route=>route.fulfill(json({...MODELS,data:MODELS.data.map(m=>({...m,metadata:{...m.metadata,loaded,ctx:context}}))})));
  await page.route('**/admin/status',route=>route.fulfill(json({...ADMIN_STATUS,models:ADMIN_STATUS.models.map(m=>({...m,loaded,ctx:context}))})));
  await page.route('**/admin/models/qwen/edit',route=>{
    const body=route.request().postDataJSON();calls.push({action:body.preview?'preview':'save',body});
    if(!body.preview)context=body.ctx;
    return route.fulfill(json(body.preview?{preview:true,vram_estimate:{estimated_mb:7000,headroom_mb:17576,fit:true}}:{reloading}));
  });
  for(const action of ['stop','load'])await page.route(`**/admin/${action}/qwen`,route=>{
    calls.push({action,auth:route.request().headers().authorization});return route.fulfill(json({loaded:action==='load'}));
  });
  await openChat(page,origin);await page.click('#settings-toggle');
  calls.setLoaded = value => { loaded = value; };
  return calls;
}
export async function testSettingsLivePreview({page,origin}) {
  const calls=await fixture(page,origin);
  for(const name of ['Conversation capacity','Key cache precision','Value cache precision','Simultaneous conversations','Memory estimate model family'])
    assert.equal(await page.getByLabel(new RegExp(name)).count(),1);
  assert.equal(await page.locator('#s-fields .s-hint').count(),7);
  assert.match(await page.locator('#s-fields').innerText(),/cache_type_k/);
  await page.fill('#s-ctx','16384');await page.selectOption('#s-ctk','q4_0');await page.selectOption('#s-ctv','q4_0');
  await page.waitForFunction(()=>document.querySelector('#s-fit').textContent.includes('7,000'));
  assert.equal(calls.length,1);assert.equal(calls[0].action,'preview');
  assert.equal(calls[0].body.ctx,16384);assert.equal(calls[0].body.cache_type_k,'q4_0');
  assert.match(await page.locator('#s-fit').innerText(),/Unsaved draft.*17,576/);
  assert.match(await page.locator('#s-reload-note').innerText(),/Apply restarts/);
  assert.equal(await page.locator('#s-reload-now').isVisible(),false);
}
export async function testSettingsPendingReload({page,origin}) {
  const calls=await fixture(page,origin);
  assert.equal(await page.locator('#s-reload-note').isVisible(),false);
  await page.selectOption('#s-ctk','q4_0');await page.click('#s-apply');
  await page.getByRole('button',{name:'Reload now',exact:true}).waitFor();
  assert.match(await page.locator('#s-reload-note').innerText(),/next load/);
  calls.setLoaded(false); await page.evaluate(()=>fetchStatus());
  assert.equal(await page.locator('#s-reload-note').isVisible(),false);
  calls.setLoaded(true); await page.evaluate(()=>fetchStatus());
  assert.equal(await page.locator('#s-reload-now').isVisible(),true);
  await page.getByRole('button',{name:'Reload now',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#s-feedback').textContent.includes('Reloaded'));
  assert.deepEqual(calls.filter(c=>['stop','load'].includes(c.action)),[
    {action:'stop',auth:'Bearer browser-test-token'},{action:'load',auth:'Bearer browser-test-token'}]);
  assert.equal(await page.locator('#s-reload-now').isVisible(),false);
}
export async function testSettingsUnloadedNoReload({page,origin}) {
  await fixture(page,origin,false);await page.fill('#s-ctx','16384');
  assert.equal(await page.locator('#s-reload-note').isVisible(),false);
  await page.click('#s-apply');await page.waitForFunction(()=>document.querySelector('#s-feedback').textContent==='Saved.');
  assert.equal(await page.locator('#s-reload-now').isVisible(),false);
}
export async function testSettingsAutomaticReload({page,origin}) {
  await fixture(page,origin,true,true);await page.selectOption('#s-ctk','q4_0');await page.click('#s-apply');
  await page.waitForFunction(()=>document.querySelector('#s-feedback').textContent.includes('Reloading'));
  assert.equal(await page.locator('#s-reload-now').isVisible(),false);
}
