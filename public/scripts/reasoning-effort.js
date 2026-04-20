export const reasoning_effort_types = {
    auto: 'auto',
    low: 'low',
    medium: 'medium',
    high: 'high',
    min: 'min',
    max: 'max',
    xhigh: 'xhigh',
};

export const verbosity_levels = {
    auto: 'auto',
    low: 'low',
    medium: 'medium',
    high: 'high',
    xhigh: 'xhigh',
    max: 'max',
};

export function resolveReasoningEffort({
    chatCompletionSource,
    model,
    selectedEffort,
    showThoughts,
}) {
    switch (selectedEffort) {
        case reasoning_effort_types.auto:
            return undefined;
        case reasoning_effort_types.min:
            if (chatCompletionSource === 'openrouter' && !showThoughts) {
                return 'none';
            }

            return ['openai', 'azure_openai'].includes(chatCompletionSource) && /^gpt-5/.test(model)
                ? reasoning_effort_types.min
                : reasoning_effort_types.low;
        case reasoning_effort_types.max:
            return chatCompletionSource === 'openrouter'
                ? reasoning_effort_types.max
                : reasoning_effort_types.high;
        case reasoning_effort_types.xhigh:
            return chatCompletionSource === 'openrouter'
                ? reasoning_effort_types.xhigh
                : reasoning_effort_types.high;
        default:
            return selectedEffort;
    }
}

export function resolveVerbosity({
    chatCompletionSource,
    selectedVerbosity,
}) {
    if (selectedVerbosity === verbosity_levels.auto) {
        return undefined;
    }

    if ([verbosity_levels.max, verbosity_levels.xhigh].includes(selectedVerbosity)) {
        return ['openrouter', 'claude'].includes(chatCompletionSource)
            ? selectedVerbosity
            : verbosity_levels.high;
    }

    return selectedVerbosity;
}
