import assert from 'node:assert/strict';
import { ADMIN_STATUS } from './harness.mjs';
export async function testRuntimeCorrectnessWarning({page,origin}) {
  const warning={status:'warning',build:'b100',detail:'Expected a correct short answer'};
  await page.route('**/admin/status',route=>route.fulfill({json:{...ADMIN_STATUS,models:ADMIN_STATUS.models.map(m=>({...m,loaded:true,correctness_check:m.name==='qwen'?warning:null}))}}));
  await page.goto(origin);
  const card=page.locator('.model-card[data-model-name="qwen"]');
  await card.waitFor();
  assert.match(await card.innerText(),/self-check warning.*b100/s);
  assert.match(await card.innerText(),/remains running.*runtime rollback/s);
  assert.equal(await card.getByRole('link',{name:'System details'}).getAttribute('href'),'/#system');
  assert.equal(await page.locator('.model-card[data-model-name="gemma"]').getByText(/self-check warning/).count(),0);
}
