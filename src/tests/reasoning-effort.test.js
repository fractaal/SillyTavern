import { test } from 'node:test';
import assert from 'node:assert/strict';

import { reasoning_effort_types, resolveReasoningEffort, resolveVerbosity, verbosity_levels } from '../../public/scripts/reasoning-effort.js';

test('[reasoning-effort] openrouter preserves max literally', () => {
    const effort = resolveReasoningEffort({
        chatCompletionSource: 'openrouter',
        model: 'anthropic/claude-opus-4.7',
        selectedEffort: reasoning_effort_types.max,
        showThoughts: true,
    });

    assert.equal(effort, 'max');
});

test('[reasoning-effort] openrouter preserves xhigh literally', () => {
    const effort = resolveReasoningEffort({
        chatCompletionSource: 'openrouter',
        model: 'anthropic/claude-opus-4.7',
        selectedEffort: reasoning_effort_types.xhigh,
        showThoughts: true,
    });

    assert.equal(effort, 'xhigh');
});

test('[reasoning-effort] non-openrouter xhigh falls back to high', () => {
    const effort = resolveReasoningEffort({
        chatCompletionSource: 'openai',
        model: 'gpt-5',
        selectedEffort: reasoning_effort_types.xhigh,
        showThoughts: true,
    });

    assert.equal(effort, 'high');
});

test('[reasoning-effort] openrouter minimum without thoughts disables reasoning', () => {
    const effort = resolveReasoningEffort({
        chatCompletionSource: 'openrouter',
        model: 'anthropic/claude-opus-4.7',
        selectedEffort: reasoning_effort_types.min,
        showThoughts: false,
    });

    assert.equal(effort, 'none');
});

test('[verbosity] openrouter preserves xhigh literally', () => {
    const verbosity = resolveVerbosity({
        chatCompletionSource: 'openrouter',
        selectedVerbosity: verbosity_levels.xhigh,
    });

    assert.equal(verbosity, 'xhigh');
});

test('[verbosity] claude preserves max literally', () => {
    const verbosity = resolveVerbosity({
        chatCompletionSource: 'claude',
        selectedVerbosity: verbosity_levels.max,
    });

    assert.equal(verbosity, 'max');
});

test('[verbosity] non-claude providers degrade xhigh to high', () => {
    const verbosity = resolveVerbosity({
        chatCompletionSource: 'openai',
        selectedVerbosity: verbosity_levels.xhigh,
    });

    assert.equal(verbosity, 'high');
});

test('[verbosity] auto omits verbosity', () => {
    const verbosity = resolveVerbosity({
        chatCompletionSource: 'openrouter',
        selectedVerbosity: verbosity_levels.auto,
    });

    assert.equal(verbosity, undefined);
});
