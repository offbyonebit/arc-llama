import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { openChat } from './harness.mjs';

const summaries = [
  { id: 'a', title: 'Planning', folder: '', created_at: 1700000000, updated_at: 1700000002 },
  { id: 'b', title: 'Project notes', folder: '', created_at: 1700000000, updated_at: 1700000001 },
];
const rawReply = '**Hello** café\n\n```python\nprint("hello")\n```';
const json = body => ({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });

async function history(page, origin) {
  await page.route('**/v1/chats', route => route.fulfill(json({ data: summaries })));
  await page.route('**/v1/chats/a', route => route.fulfill(json({ ...summaries[0], messages: [
    { role: 'user', content: 'Question?' }, { role: 'assistant', content: rawReply },
    { role: 'user', content: 'Again?' }, { role: 'assistant', content: 'Second **reply**' },
  ] })));
  await openChat(page, origin);
  await page.click('#history-toggle');
  await page.waitForFunction(() => document.querySelectorAll('#h-list .h-card').length === 2);
}

export async function testChatBasicsSearch({ page, origin }) {
  const queries = [];
  let releaseLate, sawLate;
  const gate = new Promise(resolve => { releaseLate = resolve; });
  const late = new Promise(resolve => { sawLate = resolve; });
  await page.route('**/v1/chats/search', async route => {
    const query = route.request().postDataJSON().query;
    queries.push(query);
    if (query === 'late') { sawLate(); await gate; }
    await route.fulfill(json({ data: [{ chat: summaries[1], matching_message_indices: [0] }] }));
  });
  await history(page, origin);
  await page.locator('#h-search').pressSequentially('needle');
  await page.waitForFunction(() => document.querySelectorAll('#h-list .h-card').length === 1);
  assert.deepEqual(await page.locator('.h-card-title').allTextContents(), ['Project notes']);
  assert.deepEqual(queries, ['needle']);
  await page.fill('#h-search', 'late');
  await late;
  await page.fill('#h-search', '');
  const completed = page.waitForResponse(response => response.url().endsWith('/v1/chats/search'));
  releaseLate();
  await (await completed).finished();
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  assert.equal(await page.locator('#h-list .h-card').count(), 2);
}

export async function testChatBasicsCopy({ page, origin }) {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
      writeText: async text => { window.copiedReply = text; },
    } });
  });
  await history(page, origin);
  await page.locator('.h-card[data-id="a"] .h-card-title').click();
  const buttons = page.getByRole('button', { name: 'Copy whole reply' });
  await buttons.first().waitFor();
  assert.equal(await buttons.count(), 2);
  await buttons.first().focus();
  await page.keyboard.press('Enter');
  await page.waitForFunction(raw => window.copiedReply === raw, rawReply);
  await buttons.last().click();
  await page.waitForFunction(() => window.copiedReply === 'Second **reply**');
}

export async function testChatBasicsMarkdownExport({ page, origin }) {
  const markdown = '# Planning\n\n## Assistant\n\n' + rawReply + '\n';
  let exported;
  await page.route('**/v1/chats/export?*', route => {
    exported = new URL(route.request().url());
    return route.fulfill({ status: 200, contentType: 'text/markdown', body: markdown });
  });
  await history(page, origin);
  const downloading = page.waitForEvent('download');
  await page.locator('.h-card[data-id="a"]').getByRole('button', { name: 'Export chat as Markdown' }).click();
  const download = await downloading;
  assert.equal(exported.searchParams.get('chat_id'), 'a');
  assert.equal(exported.searchParams.get('format'), 'markdown');
  assert.equal(download.suggestedFilename(), 'arc-llama-chat-a.md');
  assert.equal(await readFile(await download.path(), 'utf8'), markdown);
}
