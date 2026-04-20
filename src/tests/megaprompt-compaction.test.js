import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';

// Ensure feature flags are picked from environment so util.getConfigValue uses env and not config.yaml
process.env.SILLYTAVERN_CLAUDE_MEGAPROMPT_ENABLED = 'true';
process.env.SILLYTAVERN_CLAUDE_MEGAPROMPT_TURNMULTIPLE = '4'; // small multiple for tests
process.env.SILLYTAVERN_CLAUDE_MEGAPROMPT_MINLIVETAILTURNS = '2';
process.env.SILLYTAVERN_CLAUDE_EXTENDEDTTL = 'false';

// Point config reader at the default config to satisfy getConfigValue's file access
import { setConfigFilePath } from '../util.js';
setConfigFilePath(path.resolve(process.cwd(), 'default/config.yaml'));

// Import after setting env + config path so module init reads the right sources
const {
  applyMegapromptCompaction,
  postProcessPrompt,
  PROMPT_PROCESSING_TYPE,
  convertClaudeMessages,
  cachingAtDepthForOpenRouterClaude,
  cachingSystemPromptForOpenRouter,
} = await import('../prompt-converters.js');

const msg = (role, content) => ({ role, content });
const U = (t) => msg('user', t);
const A = (t) => msg('assistant', t);
const SYS = (t) => msg('system', t);
const TOOL = (name, payload = '{}') => ({ role: 'tool', content: [{ type: 'text', text: `TOOL(${name}): ${payload}` }] });

const texts = (messages) => messages.map(m => typeof m.content === 'string' ? m.content : (Array.isArray(m.content) ? (m.content.find(p => p?.type === 'text')?.text ?? '') : (m.content?.text ?? '')));

// Helper to find sealed text
const getSealedText = (messages) => {
  const m0 = messages[0];
  if (!m0) return '';
  const txt = typeof m0.content === 'string' ? m0.content : (m0.content?.[0]?.text ?? '');
  return txt;
};

beforeEach(() => {
  // No global state to reset for compaction
});

// 1) No-op when not enough archival turns
test('[megaprompt] short chat does not compact', () => {
  // UA turns = 3, base tail = max(2*cachingAtDepth, 2) -> with depth 1 => 2, archival = 1 < multiple(4)
  const msgs = [U('1'), A('2'), U('3')];
  const out = applyMegapromptCompaction(msgs, /*cachingAtDepth*/ 1, { enabled: true, turnMultiple: 4, minLiveTailTurns: 2, ttl: '5m' });
  assert.equal(out.length, 3);
  assert.equal(texts(out).join(','), texts(msgs).join(','));
});

// 2) Seal up to last completed multiple and leave tail intact
test('[megaprompt] seals first 4 UA turns; tail of 2 UA turns remains', () => {
  // UA: 6, depth=1 => baseTail=2, archival=4 -> lastMultiple=4
  const msgs = [U('1'), A('2'), U('3'), A('4'), U('5'), A('6')];
  const out = applyMegapromptCompaction(msgs, 1, { enabled: true, turnMultiple: 4, minLiveTailTurns: 2, ttl: '5m' });

  // Expect [sealed] + [U5, A6]
  assert.equal(out.length, 3);
  assert.equal(out[0].role, 'user');
  const sealed = getSealedText(out);
  // assert.match(sealed, /Earlier transcript \(sealed; turns 1–4\)/);
  assert.ok(sealed.includes('1'));
  assert.ok(sealed.includes('2'));
  assert.ok(sealed.includes('3'));
  assert.ok(sealed.includes('4'));
  assert.equal(texts(out.slice(1)).join(','), ['5','6'].join(','));
});

// 3) Sealed text remains identical across repeated calls (no multiple crossed)
test('[megaprompt] sealed bytes stable across identical calls', () => {
  const msgs = [U('1'), A('2'), U('3'), A('4'), U('5'), A('6')]; // same as previous test
  const out1 = applyMegapromptCompaction(msgs, 1, { enabled: true, turnMultiple: 4, minLiveTailTurns: 2, ttl: '5m' });
  const out2 = applyMegapromptCompaction(msgs, 1, { enabled: true, turnMultiple: 4, minLiveTailTurns: 2, ttl: '5m' });
  assert.equal(getSealedText(out1), getSealedText(out2));
});

// 4) Rebuild only when crossing the next multiple
// With turnMultiple=4 and baseTail=2, archivalTurns >= 8 requires UA >= 10
// So only when we go from UA=6 to UA=10 we expect the sealed chunk to grow from 4 -> 8
test('[megaprompt] sealed grows when archival crosses the next multiple', () => {
  const msgs6 = [U('1'), A('2'), U('3'), A('4'), U('5'), A('6')];
  const out6 = applyMegapromptCompaction(msgs6, 1, { enabled: true, turnMultiple: 4, minLiveTailTurns: 2, ttl: '5m' });
  const sealed6 = getSealedText(out6);

  const msgs10 = [U('1'), A('2'), U('3'), A('4'), U('5'), A('6'), U('7'), A('8'), U('9'), A('10')];
  const out10 = applyMegapromptCompaction(msgs10, 1, { enabled: true, turnMultiple: 4, minLiveTailTurns: 2, ttl: '5m' });
  const sealed10 = getSealedText(out10);

  // Should grow to include the next completed multiple of UA turns.
  assert.notEqual(sealed6, sealed10);
  assert.ok(sealed10.includes('1'));
  assert.ok(sealed10.includes('2'));
  assert.ok(sealed10.includes('3'));
  assert.ok(sealed10.includes('4'));
  assert.ok(sealed10.includes('5'));
  assert.ok(sealed10.includes('6'));
  assert.ok(sealed10.includes('7'));
  assert.ok(sealed10.includes('8'));
  assert.ok(!sealed10.includes('9'));
  assert.ok(!sealed10.includes('10'));
});

// 5) Tools excluded from sealed text, preserved in tail
test('[megaprompt] excludes tools from sealed, preserves tools in tail', () => {
  // Place a tool message in the tail region
  const msgs = [U('1'), A('2'), U('3'), A('4'), U('5'), TOOL('search', '{"q":"abc"}'), A('6')];
  // UA count = 6, depth=1 => baseTail=2, archival=4 -> lastMultiple=4, tail starts at UA index of U5
  const out = applyMegapromptCompaction(msgs, 1, { enabled: true, turnMultiple: 4, minLiveTailTurns: 2, ttl: '5m' });

  const sealed = getSealedText(out);
  // Ensure the tool marker isn't pulled into the sealed text
  assert.ok(!sealed.includes('TOOL('));

  // Tail should include the tool message and A6 in order
  const tail = out.slice(1);
  const tailText = texts(tail);
  assert.equal(tail.length, 3);
  assert.ok(tailText[0].includes('5')); // U5
  assert.ok(tailText[1].includes('TOOL(search)'));
  assert.ok(tailText[2].includes('6'));
});


// 6) Sealed remains stable when chat grows but does not cross the next multiple
// With turnMultiple=4 and baseTail=2, going from UA=6 -> UA=7 leaves lastMultiple=4, so sealed is unchanged
test('[megaprompt] sealed stable when prompt grows within same multiple', () => {
  const msgs6 = [U('1'), A('2'), U('3'), A('4'), U('5'), A('6')];
  const out6 = applyMegapromptCompaction(msgs6, 1, { enabled: true, turnMultiple: 4, minLiveTailTurns: 2, ttl: '5m' });
  const sealed6 = getSealedText(out6);

  const msgs7 = [U('1'), A('2'), U('3'), A('4'), U('5'), A('6'), U('7')];
  const out7 = applyMegapromptCompaction(msgs7, 1, { enabled: true, turnMultiple: 4, minLiveTailTurns: 2, ttl: '5m' });
  const sealed7 = getSealedText(out7);

  assert.equal(sealed6, sealed7);
});


// 7) After MERGE post-processing and Claude conversion, sealed still has cache_control on last text part
test('[megaprompt] sealed retains cache_control on the LAST content part after MERGE + Claude conversion', () => {
  const msgs = [U('1'), A('2'), U('3'), A('4'), U('5'), A('6')];
  const compacted = applyMegapromptCompaction(msgs, 1, { enabled: true, turnMultiple: 4, minLiveTailTurns: 2, ttl: '5m' });

  // Simulate optional custom post-processing the server may apply
  const names = { charName: '', userName: '', groupNames: [], startsWithGroupName: () => false };
  const merged = postProcessPrompt(compacted, PROMPT_PROCESSING_TYPE.MERGE, names);

  const converted = convertClaudeMessages([...merged], /*prefill*/ '', /*useSysPrompt*/ false, /*useTools*/ false, names);
  const first = converted.messages?.[0];
  assert.equal(first?.role, 'user');
  const content = Array.isArray(first?.content) ? first.content : [];
  const lastText = [...content].reverse().find((c) => c?.type === 'text') ?? content.at(-1);
  assert.ok(lastText?.cache_control, 'expected cache_control on last content part');
  assert.equal(lastText.cache_control.type, 'ephemeral');
  assert.equal(lastText.cache_control.ttl, '5m');
});

// 8) MERGE should preserve existing cache_control-bearing text parts (important for OpenRouter path)
test('[megaprompt][merge] preserves sealed cache_control metadata before provider conversion', () => {
  const msgs = [U('1'), A('2'), U('3'), A('4'), U('5'), A('6')];
  const compacted = applyMegapromptCompaction(msgs, 1, { enabled: true, turnMultiple: 4, minLiveTailTurns: 2, ttl: '5m' });

  const names = { charName: '', userName: '', groupNames: [], startsWithGroupName: () => false };
  const merged = postProcessPrompt(compacted, PROMPT_PROCESSING_TYPE.MERGE, names);

  const sealedCarrier = merged.find((m) => m?._megapromptSealed);
  assert.ok(sealedCarrier, 'expected merged prompt to retain _megapromptSealed marker');
  assert.ok(Array.isArray(sealedCarrier.content), 'expected merged sealed content to remain an array');

  const cachePart = sealedCarrier.content.find((c) => c?.cache_control);
  assert.ok(cachePart, 'expected cache_control to survive MERGE');
  assert.equal(cachePart.cache_control.type, 'ephemeral');
  assert.equal(cachePart.cache_control.ttl, '5m');
});


// 9) OpenRouter path: after MERGE, depth=0 retains at least one cache block.
// Depth=1 on a short merged prompt can legitimately strip the sealed block and place none.
test('[megaprompt][openrouter] depth=0 retains at least one cache block after MERGE', () => {
  const richText = (prefix, count) => Array.from({ length: count }, (_, i) => ({ type: 'text', text: `${prefix}-${i + 1}` }));
  const msgs = [
    { role: 'user', content: richText('u1', 10) },
    { role: 'assistant', content: richText('a2', 10) },
    { role: 'user', content: richText('u3', 10) },
    { role: 'assistant', content: richText('a4', 10) },
    { role: 'user', content: richText('u5', 10) },
    { role: 'assistant', content: richText('a6', 10) },
  ];
  const compacted = applyMegapromptCompaction(msgs, 1, { enabled: true, turnMultiple: 4, minLiveTailTurns: 2, ttl: '5m' });

  // Apply MERGE like the server may
  const names = { charName: '', userName: '', groupNames: [], startsWithGroupName: () => false };
  const merged = postProcessPrompt(compacted, PROMPT_PROCESSING_TYPE.MERGE, names);

  // Simulate OpenRouter anchoring
  cachingAtDepthForOpenRouterClaude(merged, /*depth*/ 0, /*ttl*/ '5m');

  // Expect at least one cache_control anchor somewhere in the messages.
  const anchorCount = merged.reduce((acc, m) => {
    const content = Array.isArray(m?.content)
      ? m.content
      : (typeof m?.content === 'string' ? [{ type: 'text', text: m.content }] : []);
    return acc + content.filter((c) => c?.type === 'text' && c?.cache_control).length;
  }, 0);
  assert.ok(anchorCount >= 1, 'expected at least one cache_control anchor after OpenRouter anchoring');
});

test('[openrouter][system-cache] upgrades existing message-level cache_control with ttl', () => {
  const msgs = [
    { role: 'system', content: [{ type: 'text', text: 'SYS' }], cache_control: { type: 'ephemeral' } },
    U('hello'),
  ];

  cachingSystemPromptForOpenRouter(msgs, '5m');
  assert.equal(msgs[0].cache_control?.type, 'ephemeral');
  assert.equal(msgs[0].cache_control?.ttl, '5m');
});

test('[openrouter][system-cache] upgrades existing content-level cache_control with ttl', () => {
  const msgs = [
    { role: 'system', content: [{ type: 'text', text: 'SYS', cache_control: { type: 'ephemeral' } }] },
    U('hello'),
  ];

  cachingSystemPromptForOpenRouter(msgs, '5m');
  const part = msgs[0].content[0];
  assert.equal(part.cache_control?.type, 'ephemeral');
  assert.equal(part.cache_control?.ttl, '5m');
});


// 10) Tail absorbs leftover beyond multiple to avoid any middle gap
// UA: 8, depth=1 => baseTail=2, archival=6, lastMultiple=4, leftover=2 => effectiveTail=4 => [U5,A6,U7,A8]
test('[megaprompt] tail absorbs leftover beyond multiple (no middle gap)', () => {
  const msgs8 = [U('1'), A('2'), U('3'), A('4'), U('5'), A('6'), U('7'), A('8')];
  const out = applyMegapromptCompaction(msgs8, 1, { enabled: true, turnMultiple: 4, minLiveTailTurns: 2, ttl: '5m' });
  const tailTexts = texts(out.slice(1));
  assert.deepEqual(tailTexts, ['5', '6', '7', '8']);
});
