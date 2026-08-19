import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';

import { setConfigFilePath } from '../util.js';
setConfigFilePath(path.resolve(process.cwd(), 'default/config.yaml'));

const {
    cachingAtDepthForOpenRouterGemini,
    cachingAtDepthForOpenRouterClaude,
} = await import('../prompt-converters.js');

const U = (t) => ({ role: 'user', content: t });
const A = (t) => ({ role: 'assistant', content: t });
const SYS = (t) => ({ role: 'system', content: t });

/** Builds a system prompt plus `turns` complete user/assistant exchanges, then a trailing user turn. */
function buildChat(turns) {
    const messages = [SYS('character card')];
    for (let i = 1; i <= turns; i++) {
        messages.push(U(`user ${i}`));
        messages.push(A(`assistant ${i}`));
    }
    messages.push(U(`user ${turns + 1}`));
    return messages;
}

/** Returns the indexes of every message carrying a cache_control breakpoint. */
function anchorIndexes(messages) {
    const found = [];
    messages.forEach((msg, i) => {
        const parts = Array.isArray(msg.content) ? msg.content : [];
        if (parts.some(p => p?.cache_control)) {
            found.push(i);
        }
    });
    return found;
}

test('places exactly one anchor -- OpenRouter honours only the last breakpoint for Gemini', () => {
    const messages = buildChat(12);
    cachingAtDepthForOpenRouterGemini(messages, 2, 4);
    assert.equal(anchorIndexes(messages).length, 1);
});

test('keeps the configured number of recent user turns outside the cached block', () => {
    const depth = 2;
    const messages = buildChat(12);
    cachingAtDepthForOpenRouterGemini(messages, depth, 1);

    const [anchor] = anchorIndexes(messages);
    const userIndexes = messages.flatMap((m, i) => (m.role === 'user' ? [i] : []));
    const anchorPosition = userIndexes.indexOf(anchor);

    assert.notEqual(anchorPosition, -1, 'anchor should land on a user message');
    // With step 1 the anchor sits exactly `depth` user turns back from the newest.
    assert.equal(userIndexes.length - 1 - anchorPosition, depth);
});

test('anchor holds still between jumps so it does not pay a cache write every turn', () => {
    const step = 4;
    const positions = [];

    // Walk a conversation forward one turn at a time and record where the anchor lands.
    for (let turns = 6; turns <= 17; turns++) {
        const messages = buildChat(turns);
        cachingAtDepthForOpenRouterGemini(messages, 2, step);
        const [anchor] = anchorIndexes(messages);
        positions.push(anchor);
    }

    const distinct = [...new Set(positions)];
    // Twelve consecutive turns must not produce twelve different anchors; with step 4
    // the anchor should move roughly every 4 turns.
    assert.ok(distinct.length < positions.length, 'anchor moved on every turn');
    assert.ok(distinct.length <= Math.ceil(positions.length / step) + 1,
        `anchor moved too often: ${distinct.length} distinct positions over ${positions.length} turns`);

    // And the anchor must only ever move forward, never backwards.
    for (let i = 1; i < positions.length; i++) {
        assert.ok(positions[i] >= positions[i - 1], 'anchor moved backwards');
    }
});

test('falls back to the system prompt when the chat is too short to quantise an anchor', () => {
    const messages = [SYS('character card'), U('hello')];
    cachingAtDepthForOpenRouterGemini(messages, 2, 4);

    const anchors = anchorIndexes(messages);
    assert.deepEqual(anchors, [0], 'short chats should still cache the system prompt');
});

test('does nothing when depth anchoring is disabled', () => {
    const messages = buildChat(12);
    cachingAtDepthForOpenRouterGemini(messages, -1, 4);
    assert.equal(anchorIndexes(messages).length, 0);
});

test('converts string content to a text part without losing the original text', () => {
    const messages = buildChat(12);
    cachingAtDepthForOpenRouterGemini(messages, 2, 4);

    const [anchor] = anchorIndexes(messages);
    const parts = messages[anchor].content;
    assert.ok(Array.isArray(parts));
    assert.equal(parts[0].type, 'text');
    assert.ok(parts[0].text.length > 0);
    assert.deepEqual(parts[0].cache_control, { type: 'ephemeral' });
});

test('leaves messages untouched when the array is empty or not an array', () => {
    assert.doesNotThrow(() => cachingAtDepthForOpenRouterGemini([], 2, 4));
    // @ts-expect-error deliberately wrong type
    assert.doesNotThrow(() => cachingAtDepthForOpenRouterGemini(null, 2, 4));
});

test('Claude anchoring is unaffected -- it still places its own multi-anchor layout', () => {
    // Regression guard: the Gemini work must not change the Claude path, which relies on
    // several breakpoints and on TTL being carried through.
    const messages = buildChat(30);
    cachingAtDepthForOpenRouterClaude(messages, 2, '1h');

    const anchors = anchorIndexes(messages);
    assert.ok(anchors.length > 1, 'Claude should still place more than one anchor');

    for (const i of anchors) {
        const part = messages[i].content.find(p => p?.cache_control);
        assert.equal(part.cache_control.ttl, '1h', 'Claude anchors must keep their TTL');
    }
});
