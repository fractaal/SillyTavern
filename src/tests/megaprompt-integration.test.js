import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';

// Enable features via env so util.getConfigValue reads env instead of config.yaml
process.env.SILLYTAVERN_CLAUDE_FIRSTANCHORCACHING_ENABLED = 'true';
process.env.SILLYTAVERN_CLAUDE_FIRSTANCHORCACHING_TTLSECONDS = '3600';
process.env.SILLYTAVERN_CLAUDE_FIRSTANCHORCACHING_ALLOWREWIND = 'true';

process.env.SILLYTAVERN_CLAUDE_MEGAPROMPT_ENABLED = 'true';
process.env.SILLYTAVERN_CLAUDE_MEGAPROMPT_TURNMULTIPLE = '4'; // small multiple for tests
process.env.SILLYTAVERN_CLAUDE_MEGAPROMPT_MINLIVETAILTURNS = '2';
process.env.SILLYTAVERN_CLAUDE_EXTENDEDTTL = 'false';

// Ensure util.getConfigValue can resolve config.yaml
import { setConfigFilePath } from '../util.js';
setConfigFilePath(path.resolve(process.cwd(), 'default/config.yaml'));

// Import after env + config path
const {
  applyMegapromptCompaction,
} = await import('../prompt-converters.js');

const {
  applyFirstAnchorReconstruction,
  FIRST_ANCHOR_STORE,
} = await import('../first-anchor-cache.js');

const msg = (role, content) => ({ role, content });
const U = (t) => msg('user', t);
const A = (t) => msg('assistant', t);
const SYS = (t) => msg('system', t);

const getText = (m) => typeof m.content === 'string'
  ? m.content
  : (Array.isArray(m.content) ? (m.content.find(p => p?.type === 'text')?.text ?? '') : (m.content?.text ?? ''));

const getSealedText = (messages) => {
  const m0 = messages.find(m => m.role !== 'system');
  if (!m0) return '';
  return Array.isArray(m0.content) ? (m0.content?.[0]?.text ?? '') : (m0.content ?? '');
};

beforeEach(() => {
  // Reset FirstAnchor global store between tests to avoid cross-test bleed
  FIRST_ANCHOR_STORE.length = 0;
});

// Mimic the server pipeline: FirstAnchor -> Megaprompt (preserving leading systems)
const runPipeline = (fullMessages, { depth = 1, turnMultiple = 4, minTail = 2, ttl = '5m' } = {}) => {
  const req = { body: { model: 'anthropic/claude-sonnet-4.5', messages: fullMessages.map(m => ({ ...m })) } };

  // FirstAnchor reconstruction (in-place on req.body.messages)
  applyFirstAnchorReconstruction(req);

  // Split leading systems
  let leadingSystemCount = 0;
  for (let i = 0; i < req.body.messages.length; i++) {
    const m = req.body.messages[i];
    if (m?.role === 'system') {
      leadingSystemCount++;
      continue;
    }
    break;
  }
  const leadingSystems = req.body.messages.slice(0, leadingSystemCount);
  const remainder = req.body.messages.slice(leadingSystemCount);

  // Megaprompt compaction on remainder
  const compacted = applyMegapromptCompaction(remainder, depth, {
    enabled: true,
    turnMultiple,
    minLiveTailTurns: minTail,
    ttl,
  });

  return [...leadingSystems, ...compacted];
};

// Ensures no UA turns are lost across consecutive runs near the live-tail boundary with a trailing system
// Repro: messages get "eaten" near the edge if boundary math or reconstruction drops a UA turn.
// Expectation: tail2 starts with tail1, and equals tail1 + newly appended UA turns.
test('[integration][first-anchor+megaprompt] preserves tail continuity across consecutive runs with trailing system', () => {
  const sysLead = SYS('SYS');
  const sysTrail = SYS('<CRITICAL_REMINDERS_AND_DIRECTIVES> ...');

  // UA: 6 (will seal 4, keep tail 2) + trailing system
  const base = [sysLead, U('1'), A('2'), U('3'), A('4'), U('5'), A('6'), sysTrail];
  const out1 = runPipeline(base);

  // Extract UA tail after sealed (skip leading systems and the sealed message itself)
  const nonSys = out1.filter(m => m.role !== 'system');
  const tail1 = nonSys.slice(1).map(getText); // after sealed
  assert.deepEqual(tail1, ['5', '6']);

  // Append two more UA turns just after A6, keep trailing system
  const extended = [sysLead, U('1'), A('2'), U('3'), A('4'), U('5'), A('6'), U('7'), A('8'), sysTrail];
  const out2 = runPipeline(extended);
  const nonSys2 = out2.filter(m => m.role !== 'system');
  const tail2 = nonSys2.slice(1).map(getText);

  // Tail continuity: previously live tail remains intact and new turns are appended
  assert.ok(tail2.length >= tail1.length, 'tail2 must be at least as long as tail1');
  assert.deepEqual(tail2.slice(0, tail1.length), tail1, 'tail2 should start with tail1 (no UA turns lost)');
  assert.deepEqual(tail2, ['5', '6', '7', '8']);
});

// Guard: within the same multiple (no crossing), the sealed chunk should remain stable and all UA turns are present
// Combine sealed UA texts + explicit UA tail, and ensure they cover all UA texts from input
test('[integration] union(sealed UA + tail UA) == original UA (no loss, no dup) within same multiple', () => {
  const sysLead = SYS('SYS');
  const sysTrail = SYS('<CRITICAL_REMINDERS_AND_DIRECTIVES> ...');

  const full = [sysLead, U('1'), A('2'), U('3'), A('4'), U('5'), A('6'), U('7'), A('8'), sysTrail];
  const out = runPipeline(full);

  const uaOriginal = full.filter(m => m.role === 'user' || m.role === 'assistant').map(getText);

  const sealed = getSealedText(out);
  // Parse by test fixture shape (each message text was a simple token like '1', '2', ...)
  const sealedParts = sealed.split(/\n\n+/).map(s => s.trim()).filter(Boolean);

  const uaTail = out.filter(m => m.role !== 'system').slice(1).map(getText);
  const uaUnion = [...sealedParts, ...uaTail];

  assert.equal(new Set(uaUnion).size, uaOriginal.length, 'no duplicates expected');
  assert.deepEqual(uaUnion, uaOriginal, 'sealed+tail should cover all UA messages in order');
});

