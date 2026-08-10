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
    if (chatCompletionSource === 'deepseek') {
        switch (selectedEffort) {
            case reasoning_effort_types.auto:
                return undefined;
            case reasoning_effort_types.max:
                return reasoning_effort_types.max;
            default:
                return reasoning_effort_types.high;
        }
    }

    if (chatCompletionSource === 'custom' && /^koboldcpp\/(.+)$/.test(model)) {
        switch (selectedEffort) {
            case reasoning_effort_types.auto:
                return undefined;
            case reasoning_effort_types.min:
                return 'minimal';
            case reasoning_effort_types.low:
                return 'low';
            case reasoning_effort_types.medium:
                return 'medium';
            case reasoning_effort_types.high:
                return 'high';
            case reasoning_effort_types.max:
                return 'xhigh';
            default:
                return selectedEffort;
        }
    }

    switch (selectedEffort) {
        case reasoning_effort_types.auto:
            return undefined;
        case reasoning_effort_types.min:
            if (chatCompletionSource === 'openrouter' && !showThoughts) {
                return 'none';
            }

            if (['openai', 'azure_openai'].includes(chatCompletionSource)) {
                if (/^gpt-5\.(4|5|6)/.test(model)) {
                    return 'none';
                }
                if (/^gpt-5/.test(model)) {
                    return reasoning_effort_types.min;
                }
            }

            return reasoning_effort_types.low;
        case reasoning_effort_types.max:
            if (chatCompletionSource === 'openrouter') {
                return reasoning_effort_types.max;
            }

            if (['openai', 'azure_openai'].includes(chatCompletionSource) && /^gpt-5\.6/.test(model)) {
                // GPT-5.6 reserves "max" effort for the Responses API.
                return 'xhigh';
            }

            return reasoning_effort_types.high;
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
