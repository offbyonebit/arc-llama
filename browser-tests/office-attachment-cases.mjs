import assert from 'node:assert/strict';
import { openChat, ASSISTANT_REPLY } from './harness.mjs';
export async function testOfficeTextAttachment({page,origin}) {
  const uploads=[],messages=[];
  await page.route('**/admin/parse-document',route=>{
    uploads.push({auth:route.request().headers().authorization,body:route.request().postData()});
    return route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({filename:'brief.docx',text:'Controlled Word attachment text.'})});
  });
  await page.route('**/v1/chat/completions',route=>{
    messages.push(route.request().postDataJSON());
    return route.fulfill({status:200,contentType:'text/event-stream',body:ASSISTANT_REPLY});
  });
  await openChat(page,origin);
  assert.match(await page.locator('#pdf-input').getAttribute('accept'),/\.docx/);
  await page.locator('#pdf-input').setInputFiles({name:'brief.docx',mimeType:'application/vnd.openxmlformats-officedocument.wordprocessingml.document',buffer:Buffer.from('controlled upload handled by mocked reader')});
  await page.waitForFunction(()=>document.querySelector('.attachment-chip:not(.processing):not(.error) .filename')?.textContent==='brief.docx');
  assert.equal(uploads.length,1);assert.equal(uploads[0].auth,'Bearer browser-test-token');
  assert.match(uploads[0].body,/brief.docx/);
  await page.fill('#message-input','Summarize the file');await page.press('#message-input','Enter');
  await page.waitForFunction(()=>!generating && !sendingMessage && document.querySelector('.message.assistant')?.textContent.includes('Hello'));
  assert.match(messages[0].messages.at(-1).content,/\[Attachment: brief.docx\]\nControlled Word attachment text\./);
}
