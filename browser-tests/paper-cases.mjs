import assert from 'node:assert/strict';
import { openChat, choosePaperModel, openContext, ADMIN_STATUS, MODELS } from './harness.mjs';
const json=body=>({status:200,contentType:'application/json',body:JSON.stringify(body)});
const rail=page=>page.locator('.paper-rail');

export async function testPaperShellAndModelPicker({page,origin}) {
  await page.goto(origin);await page.waitForSelector('.model-card');
  assert.deepEqual(await rail(page).locator('.nav-link').evaluateAll(links=>links.map(l=>l.getAttribute('aria-label'))),['Chat','Work','Models','Discover','System']);
  assert.equal(Math.round((await rail(page).boundingBox()).width),68);
  assert.equal(await page.locator('#paper-model-pill').innerText().then(t=>t.includes('Running')),true);
  await choosePaperModel(page,'gemma');
  assert.equal(await page.locator('.model-card.selected').getAttribute('data-model-name'),'gemma');
  assert.match(await page.locator('#paper-model-pill').innerText(),/Gemma.*Not loaded/s);
  await page.locator('#paper-model-pill').click();await page.keyboard.press('Escape');
  assert.equal(await page.locator('#paper-model-pill').getAttribute('aria-expanded'),'false');
  assert.equal(await page.locator('#paper-model-pill').evaluate(n=>n===document.activeElement),true);
  await rail(page).getByRole('link',{name:'System',exact:true}).click();
  assert.equal(await page.locator('.paper-header-title').innerText(),'System');
  assert.equal(await page.locator('.paper-model-layout').isVisible(),false);
  assert.equal(await page.locator('#system-readiness').isVisible(),true);
  await page.locator('#theme-toggle').click();
  assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
  await rail(page).getByRole('link',{name:'Chat',exact:true}).click();
  await page.waitForSelector('#paper-context-toggle');
  assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
  assert.equal(await page.locator('.paper-logo img').getAttribute('src'),'/assets/arc-llama-logo.png?v=paper-1');
}

export async function testPaperMobileSheets({page,origin}) {
  await page.setViewportSize({width:390,height:844});await page.goto(origin);await page.waitForSelector('.model-card');
  let box=await rail(page).boundingBox();assert.equal(Math.round(box.width),390);assert.equal(Math.round(box.height),60);
  assert.equal(await page.locator('.paper-inspector').isVisible(),false);
  await page.getByRole('button',{name:'Review model Qwen 3',exact:true}).click();
  assert.equal(await page.locator('.paper-inspector').isVisible(),true);
  box=await page.locator('.paper-inspector').boundingBox();assert.equal(Math.round(box.width),390);
  await page.getByRole('button',{name:'Close model details',exact:true}).click();
  assert.equal(await page.locator('.paper-inspector').isVisible(),false);
  await rail(page).getByRole('link',{name:'Chat',exact:true}).click();await page.waitForSelector('#message-input');
  await page.locator('#history-toggle').click();assert.equal(await page.locator('#history-panel').isVisible(),true);
  await page.getByRole('button',{name:'Close history',exact:true}).click();assert.equal(await page.locator('#history-panel').isVisible(),false);
  await page.locator('#settings-toggle').click();assert.equal(Math.round((await page.locator('#settings-panel').boundingBox()).width),390);
  await page.keyboard.press('Escape');assert.equal(await page.locator('#settings-panel').isVisible(),false);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await page.getByRole('button',{name:'Toggle theme',exact:true}).click();assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
}

export async function testPaperRealClaimsAndContext({page,origin}) {
  await page.route('**/admin/status',route=>route.fulfill(json({...ADMIN_STATUS,models:ADMIN_STATUS.models.map(m=>({...m,file_readiness:{available:m.name==='qwen'},vram_estimate:{fit:false,estimated_mb:50000}}))})));
  await page.goto(origin);await page.waitForSelector('.paper-state-ladder');
  let text=await page.locator('.paper-state-ladder').innerText();assert.match(text,/Downloaded/);assert.match(text,/Estimated to exceed memory/);assert.match(text,/Running/);
  await page.locator('.paper-rail a[aria-label="Discover"]').click();assert.match(await page.locator('#library-title').innerText(),/External/);
  await openChat(page,origin);await page.fill('#message-input','Count this draft');await openContext(page);
  await page.waitForFunction(()=>document.querySelector('#ctx-label-left').textContent.includes('tokens'));
  assert.match(await page.locator('#ctx-label-left').innerText(),/tokens.*reply/);
  assert.equal(await page.locator('#compact-button').isVisible(),true);
  await page.keyboard.press('Escape');assert.equal(await page.locator('#paper-context-pop').isVisible(),false);
}

export async function testPaperModelLoadStop({page,origin}) {
  const calls=[];let loaded=true;
  await page.route('**/admin/status',route=>route.fulfill(json({...ADMIN_STATUS,models:ADMIN_STATUS.models.map(m=>({...m,loaded:m.name==='qwen'&&loaded,file_readiness:{available:true}}))})));
  for(const action of ['load','stop']) await page.route(`**/admin/${action}/qwen`,route=>{calls.push({action,method:route.request().method(),auth:route.request().headers().authorization});loaded=action==='load';return route.fulfill(json({ok:true}));});
  await page.goto(origin);await page.getByRole('button',{name:'Stop model',exact:true}).click();
  await page.getByRole('button',{name:'Load model',exact:true}).click();
  await page.getByRole('button',{name:'Stop model',exact:true}).waitFor();
  assert.deepEqual(calls,[{action:'stop',method:'POST',auth:'Bearer browser-test-token'},{action:'load',method:'POST',auth:'Bearer browser-test-token'}]);
}

export async function testPaperWorkModelPicker({page,origin}) {
  await page.route('**/v1/agent/info', route=>route.fulfill(json({experimental:true,root:'/work/demo',home:'/home/demo',harnesses:[{id:'builtin',label:'Built-in',available:true}]})));
  await page.route('**/admin/library/jobs',route=>route.fulfill(json({jobs:[{id:'download-1',status:'downloading'}]})));
  await openChat(page,origin);
  await page.locator('#mode-switch:not([hidden])').waitFor({state:'attached'});
  await rail(page).getByRole('link',{name:'Work',exact:true}).click();
  await page.waitForSelector('body.work-mode');
  await choosePaperModel(page,'gemma');
  assert.equal(await page.locator('#work-model').inputValue(),'gemma');
  assert.match(await page.locator('#paper-model-pill').innerText(),/Gemma.*Not loaded/s);
  assert.equal(await page.locator('#work-model').isVisible(),false);
  assert.equal(await page.locator('.paper-nav-badge').innerText(),'1');
  await rail(page).getByRole('link',{name:'Chat',exact:true}).click();
  assert.match(await page.locator('#paper-model-pill').innerText(),/Qwen 3.*Running/s);
  // The status poll wins over stale metadata from the initial /v1/models list.
  await page.route('**/admin/status',route=>route.fulfill(json({...ADMIN_STATUS,models:ADMIN_STATUS.models.map(m=>({...m,loaded:false,state:'idle'}))})));
  await page.evaluate(()=>fetchStatus());
  assert.match(await page.locator('#paper-model-pill').innerText(),/Qwen 3.*Not loaded/s);
}

export async function testPaperDiscoverNavigation({page,origin}) {
  await page.goto(origin);await page.waitForSelector('.model-card');
  const discover=rail(page).getByRole('link',{name:'Discover',exact:true});
  const models=rail(page).getByRole('link',{name:'Models',exact:true});
  await discover.click();
  assert.equal(new URL(page.url()).hash,'#discover');
  assert.equal(await discover.getAttribute('aria-current'),'page');
  assert.equal(await page.title(),'Discover: arc-llama');
  assert.equal(await page.locator('.paper-header-title').innerText(),'Discover');
  assert.equal(await page.locator('#library-title').isVisible(),true);
  assert.equal(await page.locator('#model-list').isVisible(),false);
  assert.equal(await page.locator('.paper-inspector').isVisible(),false);
  assert.equal(await page.getByRole('tablist',{name:'Model library'}).count(),0);
  const search=page.locator('#library-query');
  await search.fill('qwen');
  await models.click();
  assert.equal(await page.locator('#model-list').isVisible(),true);
  assert.equal(await page.locator('#library-title').isVisible(),false);
  await page.goBack();
  assert.equal(await discover.getAttribute('aria-current'),'page');
  assert.equal(await search.inputValue(),'qwen');
  await page.goForward();
  assert.equal(await models.getAttribute('aria-current'),'page');
  for(const hash of ['#discover','#library-title']) {
    await page.goto(`${origin}/${hash}`);
    assert.equal(await discover.getAttribute('aria-current'),'page');
    assert.equal(await page.locator('#library-title').isVisible(),true);
    assert.equal(await page.locator('#system-readiness').isVisible(),false);
  }
  for(const path of ['/chat','/work']) {
    await page.goto(`${origin}${path}`);
    assert.equal(await discover.getAttribute('href'),'/#discover');
    await discover.click();
    assert.equal(await discover.getAttribute('aria-current'),'page');
  }
  await page.setViewportSize({width:320,height:740});
  for(const link of await rail(page).locator('.nav-link').all()) {
    const box=await link.boundingBox();
    assert.ok(box && box.x>=0 && box.x+box.width<=320);
  }
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
}

export async function testPaperPickerListVisibility({page,origin}) {
  await page.route('**/v1/agent/info',route=>route.fulfill(json({experimental:true,root:'/work/demo',home:'/home/demo',harnesses:[{id:'builtin',label:'Built-in',available:true}]})));
  const models=Array.from({length:18},(_,i)=>({...MODELS.data[0],id:i?`model-${i}`:'qwen',metadata:{...MODELS.data[0].metadata,display_name:i?`Model ${i+1} with a longer descriptive name`:'Qwen 3',loaded:i===0}}));
  await page.route('**/v1/models',route=>route.fulfill(json({...MODELS,data:models})));
  await page.route('**/admin/status',route=>route.fulfill(json({...ADMIN_STATUS,models:models.map(m=>({...ADMIN_STATUS.models[0],name:m.id,display_name:m.metadata.display_name,loaded:m.metadata.loaded,state:'idle'}))})));
  for(const [path,width,height] of [['/chat',1440,900],['/work',1440,900],['/',1440,900],['/chat',320,640],['/work',844,390]]) {
    await page.setViewportSize({width,height});await page.goto(`${origin}${path}?model=qwen`);
    await page.waitForFunction(()=>document.querySelectorAll('.paper-model-option').length===18);
    await page.locator('#paper-model-pill').click();await page.keyboard.press('Home');
    const menu=page.locator('#paper-model-menu');const options=page.locator('.paper-model-options');
    await page.waitForFunction(()=>!document.querySelector('#paper-model-scroll-hint').hidden);
    assert.equal(await menu.locator('.paper-model-count').innerText(),'18 models');
    assert.match(await page.locator('#paper-model-scroll-hint').innerText(),/more models ↓/);
    const visible=await options.evaluate(list=>{
      const area=list.getBoundingClientRect();
      return [...list.querySelectorAll('button')].filter(item=>{
        const r=item.getBoundingClientRect();const x=r.x+r.width/2,y=r.y+r.height/2;
        return r.top>=area.top && r.bottom<=area.bottom && item.contains(document.elementFromPoint(x,y));
      }).length;
    });
    assert.ok(visible>=(height<450?3:5),`only ${visible} reachable model rows on ${path} at ${width}x${height}`);
    const box=await menu.boundingBox();assert.ok(box.x>=0 && box.x+box.width<=width && box.y+box.height<=height);
    // The footer stays reachable while the list scrolls; keyboard reaches the last model.
    assert.ok(await menu.getByRole('link',{name:'Manage models →'}).evaluate(node=>{const r=node.getBoundingClientRect();return node.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2));}));
    await page.keyboard.press('End');
    await page.waitForFunction(()=>document.querySelector('#paper-model-scroll-hint').textContent.includes('earlier'));
    assert.equal(await options.locator('button').last().evaluate(node=>node===document.activeElement),true);
    await page.keyboard.press('Enter');assert.equal(await menu.isHidden(),true);
    assert.match(await page.locator('#paper-model-pill').innerText(),/Model 18/);
    await page.locator('#paper-model-pill').click();await page.keyboard.press('Escape');
    assert.equal(await menu.isHidden(),true);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  }
  models.splice(2);
  await page.goto(`${origin}/chat?model=qwen`);
  await page.waitForFunction(()=>document.querySelectorAll('.paper-model-option').length===2);
  await page.locator('#paper-model-pill').click();
  assert.equal(await page.locator('.paper-model-count').innerText(),'2 models');
  await page.waitForFunction(()=>document.querySelector('#paper-model-scroll-hint').hidden);
  assert.equal(await page.locator('.paper-model-options').getAttribute('aria-describedby'),null);
}
