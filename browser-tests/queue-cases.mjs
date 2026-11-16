import assert from 'node:assert/strict';
import { openChat, ASSISTANT_REPLY } from './harness.mjs';
const json = body => ({status:200,contentType:'application/json',body:JSON.stringify(body)});

async function fixture(page, origin) {
  let saved = {id:'queue-chat',title:'Queue chat',model:'qwen',messages:[],created_at:1,updated_at:1};
  const other = {id:'other-chat',title:'Other conversation',model:'qwen',messages:[],created_at:1,updated_at:1};
  await page.route('**/v1/chats', route => route.fulfill(json(route.request().method()==='POST'?saved:{data:[saved,other]})));
  await page.route('**/v1/chats/queue-chat', route => {
    if(route.request().method()==='PATCH') saved.messages.push(...route.request().postDataJSON().messages);
    if(route.request().method()==='PUT') saved={...saved,...route.request().postDataJSON()};
    return route.fulfill(json(saved));
  });
  await page.route('**/v1/chats/other-chat', route => route.fulfill(json(other)));
  await page.addInitScript(reply => {
    const fetch = window.fetch.bind(window);
    window.__queuedRequests=[];
    window.fetch=(url,options={})=>{
      if(!String(url).endsWith('/v1/chat/completions')) return fetch(url,options);
      window.__queuedRequests.push(JSON.parse(options.body));
      if(window.__queuedRequests.length>1) return Promise.resolve(new Response(reply,{headers:{'Content-Type':'text/event-stream'}}));
      let ended=false;
      return Promise.resolve(new Response(new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('data: '+JSON.stringify({choices:[{delta:{content:'Partial first answer'}}]})+'\n\n'));
          options.signal.addEventListener('abort',()=>{if(!ended){ended=true;controller.error(new DOMException('Stopped','AbortError'));}},{once:true});
          window.__finishQueuedReply=(fail=false)=>{
            if(ended) return; ended=true;
            if(fail){controller.error(new Error('Synthetic interruption'));return;}
            controller.enqueue(new TextEncoder().encode('data: '+JSON.stringify({choices:[{delta:{},finish_reason:'stop'}]})+'\n\ndata: [DONE]\n\n'));
            controller.close();
          };
        },
        cancel(){ended=true;},
      }),{headers:{'Content-Type':'text/event-stream'}}));
    };
  }, ASSISTANT_REPLY);
  await openChat(page,origin);
  await page.fill('#message-input','First question'); await page.press('#message-input','Enter');
  await page.waitForFunction(()=>window.__queuedRequests.length===1 && document.querySelector('.message.assistant')?.textContent.includes('Partial first answer'));
  assert.equal(await page.locator('#message-input').isEditable(),true);
}
async function queue(page,text='Follow-up question',keyboard=false) {
  await page.fill('#message-input',text);
  if (keyboard) await page.press('#message-input','Enter');
  else await page.getByRole('button',{name:/^(Queue message|Replace queued message)$/}).click();
  await page.waitForFunction(text=>!document.querySelector('#queued-message').hidden && document.querySelector('#queued-message-text').textContent.includes(text),text);
  assert.equal(await page.inputValue('#message-input'),'');
}

export async function testChatQueueAutoSend({page,origin}) {
  await fixture(page,origin); await queue(page,'Old queued text',true); await queue(page,'Replacement follow-up');
  await page.fill('#message-input','Draft for later');
  await page.evaluate(()=>window.__finishQueuedReply());
  await page.waitForFunction(()=>window.__queuedRequests.length===2 && !generating && !sendingMessage);
  const requests=await page.evaluate(()=>window.__queuedRequests);
  assert.equal(requests[1].messages.at(-1).content,'Replacement follow-up');
  assert.equal(requests[1].messages.at(-2).content,'Partial first answer');
  assert.equal(await page.inputValue('#message-input'),'Draft for later');
  assert.equal(await page.locator('#queued-message').isVisible(),false);
}
export async function testChatQueueStopRestores({page,origin}) {
  await fixture(page,origin); await queue(page);
  await page.fill('#message-input','Another draft');
  await page.getByRole('button',{name:'Stop reply',exact:true}).click();
  await page.waitForFunction(()=>!generating && !sendingMessage);
  assert.equal(await page.inputValue('#message-input'),'Follow-up question\n\nAnother draft');
  assert.equal(await page.evaluate(()=>window.__queuedRequests.length),1);
  assert.equal(await page.locator('#queued-message').isVisible(),false);
}
export async function testChatQueueSwitchRestores({page,origin}) {
  await fixture(page,origin); await queue(page);
  await page.click('#history-toggle'); await page.locator('.h-card-title').filter({hasText:'Other conversation'}).click();
  // Existing protection requires Stop before switching; attempting a change cancels the queue.
  assert.equal(await page.inputValue('#message-input'),'Follow-up question');
  await page.getByRole('button',{name:'Close history',exact:true}).click();
  await page.getByRole('button',{name:'Stop reply',exact:true}).click();
  await page.waitForFunction(()=>!generating && !sendingMessage);
  await page.click('#history-toggle');
  await page.locator('.h-card-title').filter({hasText:'Other conversation'}).click();
  await page.waitForFunction(()=>currentChatId==='other-chat');
  assert.equal(await page.inputValue('#message-input'),'Follow-up question');
  assert.equal(await page.evaluate(()=>window.__queuedRequests.length),1);
}
export async function testChatQueueRemove({page,origin}) {
  await fixture(page,origin); await queue(page);
  await page.getByRole('button',{name:'Remove queued message'}).click();
  await page.evaluate(()=>window.__finishQueuedReply());
  await page.waitForFunction(()=>!generating && !sendingMessage);
  assert.equal(await page.evaluate(()=>window.__queuedRequests.length),1);
  assert.equal(await page.inputValue('#message-input'),'');
  assert.equal(await page.locator('#queued-message').isVisible(),false);
}
export async function testChatQueueErrorRestores({page,origin}) {
  await fixture(page,origin); await queue(page);
  await page.evaluate(()=>window.__finishQueuedReply(true));
  await page.waitForFunction(()=>!generating && !sendingMessage);
  assert.equal(await page.inputValue('#message-input'),'Follow-up question');
  assert.equal(await page.evaluate(()=>window.__queuedRequests.length),1);
}
