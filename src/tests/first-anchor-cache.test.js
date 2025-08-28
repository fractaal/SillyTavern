import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

// Ensure feature flags are picked from environment so util.getConfigValue doesn't read config.yaml
process.env.SILLYTAVERN_CLAUDE_FIRSTANCHORCACHING_ENABLED = 'true';
process.env.SILLYTAVERN_CLAUDE_FIRSTANCHORCACHING_TTLSECONDS = '3600';
process.env.SILLYTAVERN_CLAUDE_FIRSTANCHORCACHING_ALLOWREWIND = 'true';

import { applyFirstAnchorReconstruction, FIRST_ANCHOR_STORE } from '../first-anchor-cache.js';

const msg = (role, text) => ({ role, content: text });
const sys = (text) => msg('system', text);

const $ = {
    a: (text) => msg('assistant', text),
    u: (text) => msg('user', text)
};

const buildMessages = (sysText, messageBuilders) => [
    sys(sysText),
    ...messageBuilders.map(builder => builder)
];

const texts = (messages) => messages.map(m => (typeof m.content === 'string' ? m.content : (m.content?.[0]?.text ?? '')));

beforeEach(() => {
    // Reset the in-memory store between tests
    FIRST_ANCHOR_STORE.length = 0;
});

// [SYS,1,2,3,4] -> [SYS,2,3,4,5] = [SYS,1,2,3,4,5]
test('[extend] [SYS,1,2,3,4] -> [SYS,2,3,4,5] = [SYS,1,2,3,4,5]', () => {
    const sysText = 'SYS';
    const req1 = { body: { messages: buildMessages(sysText, [$.u('1'), $.a('2'), $.u('3'), $.a('4')]) } };
    applyFirstAnchorReconstruction(req1); // seed only, no rewrite expected
    assert.equal(texts(req1.body.messages).join(','), ['SYS', '1', '2', '3', '4'].join(','));

    const req2 = { body: { messages: buildMessages(sysText, [$.a('2'), $.u('3'), $.a('4'), $.u('5')]) } };
    applyFirstAnchorReconstruction(req2);
    assert.equal(texts(req2.body.messages).join(','), ['SYS', '1', '2', '3', '4', '5'].join(','));
});

// [SYS,1,2,3,4] -> [SYS,3,4,5,6] = [SYS,1,2,3,4,5,6]
test('[extend] [SYS,1,2,3,4] -> [SYS,3,4,5,6] = [SYS,1,2,3,4,5,6]', () => {
    const sysText = 'SYS';
    const req1 = { body: { messages: buildMessages(sysText, [$.u('1'), $.a('2'), $.u('3'), $.a('4')]) } };
    applyFirstAnchorReconstruction(req1); // seed

    const req2 = { body: { messages: buildMessages(sysText, [$.u('3'), $.a('4'), $.u('5'), $.a('6')]) } };
    applyFirstAnchorReconstruction(req2);
    assert.equal(texts(req2.body.messages).join(','), ['SYS', '1', '2', '3', '4', '5', '6'].join(','));
});

// [SYS,1,2,3,4,5,6] -> [SYS,1,2,3] = [SYS,1,2,3]
test('[rewind] [SYS,1,2,3,4,5,6] -> [SYS,1,2,3] = [SYS,1,2,3]', () => {
    const sysText = 'SYS';
    const req1 = { body: { messages: buildMessages(sysText, [$.u('1'), $.a('2'), $.u('3'), $.a('4'), $.u('5'), $.a('6')]) } };
    applyFirstAnchorReconstruction(req1); // seed

    const req2 = { body: { messages: buildMessages(sysText, [$.u('1'), $.a('2'), $.u('3')]) } };
    applyFirstAnchorReconstruction(req2);
    assert.equal(texts(req2.body.messages).join(','), ['SYS', '1', '2', '3'].join(','));
});

// [SYS,1,2,3,4,5,6] -> [SYS,1,2,3E] = [SYS,1,2,3E]
test('[rewind+edit] [SYS,1,2,3,4,5,6] -> [SYS,1,2,3E] = [SYS,1,2,3E]', () => {
    const sysText = 'SYS';
    const req1 = { body: { messages: buildMessages(sysText, [$.u('1'), $.a('2'), $.u('3'), $.a('4'), $.u('5'), $.a('6')]) } };
    applyFirstAnchorReconstruction(req1); // seed

    const req2 = { body: { messages: buildMessages(sysText, [$.u('1'), $.a('2'), $.u('3E')]) } };
    applyFirstAnchorReconstruction(req2);
    assert.equal(texts(req2.body.messages).join(','), ['SYS', '1', '2', '3E'].join(','));
});

test('[rewind-2] [SYS,1,2,3,4,5,6] -> [SYS,2,3,4] = [SYS,1,2,3,4]', () => {
    const sysText = 'SYS';
    const req1 = { body: { messages: buildMessages(sysText, [$.u('1'), $.a('2'), $.u('3'), $.a('4'), $.u('5'), $.a('6')]) } };
    applyFirstAnchorReconstruction(req1); // seed

    const req2 = { body: { messages: buildMessages(sysText, [$.a('2'), $.u('3'), $.a('4')]) } };
    applyFirstAnchorReconstruction(req2);
    assert.equal(texts(req2.body.messages).join(','), ['SYS', '1', '2', '3', '4'].join(','));
});

test('[rewind+edit-2] [SYS,1,2,3,4,5,6] -> [SYS,2,3,4E] = [SYS,1,2,3,4E]', () => {
    const sysText = 'SYS';
    const req1 = { body: { messages: buildMessages(sysText, [$.u('1'), $.a('2'), $.u('3'), $.a('4'), $.u('5'), $.a('6')]) } };
    applyFirstAnchorReconstruction(req1); // seed

    const req2 = { body: { messages: buildMessages(sysText, [$.a('2'), $.u('3'), $.a('4E')]) } };
    applyFirstAnchorReconstruction(req2);
    assert.equal(texts(req2.body.messages).join(','), ['SYS', '1', '2', '3', '4E'].join(','));
});

test('[not-match]', () => {
    const sysText = 'SYS';

    const req1 = { body: { messages: buildMessages(sysText, [$.u('This'), $.a('is'), $.u('a'), $.a('conversation')]) } };

    applyFirstAnchorReconstruction(req1); // seed

    const req2 = { body: { messages: buildMessages(sysText, [$.u('Completely different'), $.a('conversation'), $.u('now'), $.a('lol')]) } };

    applyFirstAnchorReconstruction(req2);
    assert.equal(texts(req2.body.messages).join(','), ['SYS', 'Completely different', 'conversation', 'now', 'lol'].join(','));
});
