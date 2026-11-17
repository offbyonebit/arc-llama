import assert from 'node:assert/strict';
import { openChat } from './harness.mjs';
const localA={voiceURI:'local-a',name:'Local A',lang:'en-US',localService:true};
const localB={voiceURI:'local-b',name:'Local B',lang:'en-GB',localService:true};
const network={voiceURI:'network',name:'Network voice',lang:'en-US',localService:false};

async function setup(page,origin,{voices=[network,localA,localB],stored='network',storageFails=false}={}) {
  await page.addInitScript(({voices,stored,storageFails})=>{
    const listeners=[];
    window.__speechCalls=[];window.__speechCancels=0;
    let currentVoices=voices;
    const synth={
      getVoices:()=>currentVoices,
      addEventListener:(event,fn)=>{if(event==='voiceschanged')listeners.push(fn);},
      speak:utterance=>window.__speechCalls.push({text:utterance.text,voice:utterance.voice.voiceURI,local:utterance.voice.localService}),
      cancel:()=>{window.__speechCancels++;},
    };
    Object.defineProperty(window,'speechSynthesis',{configurable:true,value:synth});
    Object.defineProperty(window,'SpeechSynthesisUtterance',{configurable:true,value:class {constructor(text){this.text=text;}}});
    window.__setSpeechVoices=next=>{currentVoices=next;for(const fn of listeners)fn();};
    if(stored&&!localStorage.getItem('arc-llama-local-voice'))localStorage.setItem('arc-llama-local-voice',stored);
    if(storageFails){
      const get=Storage.prototype.getItem,set=Storage.prototype.setItem;
      Storage.prototype.getItem=function(key){if(key==='arc-llama-local-voice')throw new Error('storage blocked');return get.call(this,key);};
      Storage.prototype.setItem=function(key,value){if(key==='arc-llama-local-voice')throw new Error('storage blocked');return set.call(this,key,value);};
    }
  },{voices,stored,storageFails});
  await page.route('**/v1/chat/completions',route=>route.fulfill({status:200,contentType:'text/event-stream',body:
    'data: '+JSON.stringify({choices:[{delta:{content:'<think>private reasoning</think>Public reply.\n\n```python\nsecret_code()\n```\n\nFinal sentence.'}}]})+'\n\n'+
    'data: '+JSON.stringify({choices:[{delta:{},finish_reason:'stop'}]})+'\n\ndata: [DONE]\n\n'}));
  await openChat(page,origin);
}
async function reply(page,text='Read this reply') {
  await page.fill('#message-input',text);await page.press('#message-input','Enter');
  await page.waitForFunction(()=>!generating && !sendingMessage && document.querySelector('.message.assistant .content')?.textContent.includes('Public reply'));
}

export async function testSpeechLocalOnlyAndToggle({page,origin}) {
  await setup(page,origin);await reply(page);
  const button=page.getByRole('button',{name:'Read aloud',exact:true});await button.waitFor();
  await button.click();
  const [call]=await page.evaluate(()=>window.__speechCalls);
  assert.equal(call.local,true);assert.equal(call.voice,'local-a');
  assert.match(call.text,/Public reply/);assert.match(call.text,/code block/);assert.match(call.text,/Final sentence/);
  assert.doesNotMatch(call.text,/private reasoning|secret_code/);
  await page.getByRole('button',{name:'Stop reading',exact:true}).click();
  assert.equal(await page.evaluate(()=>window.__speechCancels),1);
  assert.equal(await page.evaluate(()=>window.__speechCalls.length),1);
}
export async function testSpeechNoLocalVoice({page,origin}) {
  await setup(page,origin,{voices:[network]});await reply(page);
  assert.equal(await page.getByRole('button',{name:'Read aloud',exact:true}).count(),0);
  await page.click('#settings-toggle');assert.equal(await page.locator('#speech-settings').isVisible(),false);
  assert.equal(await page.evaluate(()=>window.__speechCalls.length),0);
}
export async function testSpeechVoicePickerAndReplacement({page,origin}) {
  await setup(page,origin);await reply(page,'First');await reply(page,'Second');
  await page.click('#settings-toggle');
  const picker=page.getByLabel('Read-aloud voice');
  assert.deepEqual(await picker.locator('option').evaluateAll(nodes=>nodes.map(n=>n.value)),['local-a','local-b']);
  await picker.selectOption('local-b');
  assert.equal(await page.evaluate(()=>localStorage.getItem('arc-llama-local-voice')),'local-b');
  await page.keyboard.press('Escape');
  const buttons=page.getByRole('button',{name:'Read aloud',exact:true});
  await buttons.nth(0).click();await page.getByRole('button',{name:'Read aloud',exact:true}).click();
  const calls=await page.evaluate(()=>window.__speechCalls);
  assert.equal(calls.length,2);assert.ok(calls.every(c=>c.local && c.voice==='local-b'));
  assert.equal(await page.evaluate(()=>window.__speechCancels),1);
  await page.reload();await page.waitForSelector('#model-select option',{state:'attached'});
  await page.click('#settings-toggle');assert.equal(await page.inputValue('#speech-voice'),'local-b');
}
export async function testSpeechVoicesChangingNeverFallback({page,origin}) {
  await setup(page,origin);await reply(page);await page.getByRole('button',{name:'Read aloud',exact:true}).click();
  await page.evaluate(voice=>window.__setSpeechVoices([voice]),network);
  assert.equal(await page.evaluate(()=>window.__speechCancels),1);
  assert.equal(await page.locator('.read-aloud').count(),0);
  assert.equal(await page.evaluate(()=>window.__speechCalls.length),1);
}
export async function testSpeechBlockedPreferenceStorage({page,origin}) {
  await setup(page,origin,{stored:null,storageFails:true});await reply(page);
  await page.click('#settings-toggle');await page.getByLabel('Read-aloud voice').selectOption('local-b');
  await page.keyboard.press('Escape');await page.getByRole('button',{name:'Read aloud',exact:true}).click();
  assert.equal(await page.evaluate(()=>window.__speechCalls[0].voice),'local-b');
}
