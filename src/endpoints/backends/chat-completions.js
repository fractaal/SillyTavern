import process from 'node:process';
import util from 'node:util';
import express from 'express';
import fetch from 'node-fetch';

import {
    AIMLAPI_HEADERS,
    CHAT_COMPLETION_SOURCES,
    GEMINI_SAFETY,
    OPENROUTER_HEADERS,
} from '../../constants.js';
import {
    forwardFetchResponse,
    getConfigValue,
    tryParse,
    uuidv4,
    mergeObjectWithYaml,
    excludeKeysByYaml,
    color,
    trimTrailingSlash,
    flattenSchema,
} from '../../util.js';
import {
    convertClaudeMessages,
    convertGooglePrompt,
    convertTextCompletionPrompt,
    convertCohereMessages,
    convertMistralMessages,
    convertAI21Messages,
    convertXAIMessages,
    cachingAtDepthForOpenRouterClaude,
    cachingAtDepthForClaude,
    getPromptNames,
    calculateClaudeBudgetTokens,
    calculateGoogleBudgetTokens,
    postProcessPrompt,
    PROMPT_PROCESSING_TYPE,
    addAssistantPrefix,
} from '../../prompt-converters.js';

import { readSecret, SECRET_KEYS } from '../secrets.js';
import {
    getTokenizerModel,
    getSentencepiceTokenizer,
    getTiktokenTokenizer,
    sentencepieceTokenizers,
    TEXT_COMPLETION_MODELS,
    webTokenizers,
    getWebTokenizer,
} from '../tokenizers.js';
import { getVertexAIAuth, getProjectIdFromServiceAccount } from '../google.js';

const API_OPENAI = 'https://api.openai.com/v1';
const API_CLAUDE = 'https://api.anthropic.com/v1';
const API_MISTRAL = 'https://api.mistral.ai/v1';
const API_COHERE_V1 = 'https://api.cohere.ai/v1';
const API_COHERE_V2 = 'https://api.cohere.ai/v2';
const API_PERPLEXITY = 'https://api.perplexity.ai';
const API_GROQ = 'https://api.groq.com/openai/v1';
const API_MAKERSUITE = 'https://generativelanguage.googleapis.com';

// --- First-Anchor Context Reconstruction (global matching) ---
// Transparent, optional pre-processing to preserve cache hits under limited context.
// Controlled by config: claude.firstAnchorCaching.enabled / ttlSeconds
const FIRST_ANCHOR_STORE = [];

const removeEphemeralFields = (obj) => {
    if (!obj || typeof obj !== 'object') return obj;
    // Shallow copy then strip known ephemeral keys
    const copy = Array.isArray(obj) ? obj.map(removeEphemeralFields) : { ...obj };
    if (!Array.isArray(copy)) {
        delete copy.cache_control;
        delete copy.id; // ignore any transient ids
    }
    if (Array.isArray(copy)) return copy;
    for (const k of Object.keys(copy)) {
        const v = copy[k];
        if (v && typeof v === 'object') copy[k] = removeEphemeralFields(v);
    }
    return copy;
};

const normalizeContentArray = (content) => {
    if (typeof content === 'string') {
        return [{ type: 'text', text: content }];
    }
    if (Array.isArray(content)) {
        return content.map((part) => {
            if (typeof part === 'string') return { type: 'text', text: part };
            return removeEphemeralFields(part);
        });
    }
    return [];
};

const fingerprintMessage = (msg) => {
    const norm = {
        role: msg.role,
        content: normalizeContentArray(msg.content),
    };
    return JSON.stringify(norm);
};

const getTextPreview = (msg, max = 80) => {
    try {
        let text = '';
        if (typeof msg?.content === 'string') {
            text = msg.content;
        } else if (Array.isArray(msg?.content)) {
            const textPart = msg.content.find(p => p && p.type === 'text');
            text = textPart?.text ?? '';
        }
        return String(text).replace(/\s+/g, ' ').trim().slice(0, max);
    } catch {
        return '';
    }
};

const msgPreview = (msg) => `${msg?.role ?? '?'}: "${getTextPreview(msg)}"`;

const purgeExpiredFirstAnchor = () => {
    const now = Date.now();
    let purged = 0;
    for (let i = FIRST_ANCHOR_STORE.length - 1; i >= 0; i--) {
        if (FIRST_ANCHOR_STORE[i].expireAt <= now) { FIRST_ANCHOR_STORE.splice(i, 1); purged++; }
    }
    return purged;
};

const longestRightEndedOverlap = (a, b) => {
    const maxK = Math.min(a.length, b.length);
    for (let k = maxK; k > 0; k--) {
        let match = true;
        for (let i = 0; i < k; i++) {
            if (a[a.length - k + i] !== b[i]) { match = false; break; }
        }
        if (match) return k;
    }
    return 0;
};

const findLongestPrefixMatch = (big, small) => {
    // Returns { startIndex, length } where small[0..length-1] matches big[start..start+length-1]
    if (!Array.isArray(big) || !Array.isArray(small) || small.length === 0) return { startIndex: -1, length: 0 };
    let best = { startIndex: -1, length: 0 };
    for (let s = 0; s < big.length; s++) {
        let len = 0;
        while (s + len < big.length && len < small.length && big[s + len] === small[len]) {
            len++;
        }
        if (len > best.length) best = { startIndex: s, length: len };
        if (best.length === small.length) break; // perfect
    }
    return best;
};

const indexOfSubsequence = (big, small) => {
    if (!Array.isArray(big) || !Array.isArray(small) || small.length === 0 || small.length > big.length) return -1;
    outer: for (let i = 0; i <= big.length - small.length; i++) {
        for (let j = 0; j < small.length; j++) {
            if (big[i + j] !== small[j]) continue outer;
        }
        return i;
    }
    return -1;
};


function applyFirstAnchorReconstruction(request) {
    try {
        const enabled = getConfigValue('claude.firstAnchorCaching.enabled', false, 'boolean');
        if (!enabled) return;
        if (!Array.isArray(request.body?.messages) || request.body.messages.length === 0) return;

        // Split leading system block
        const sys = [];
        let i = 0;
        while (i < request.body.messages.length && request.body.messages[i]?.role === 'system') {
            sys.push(request.body.messages[i]);
            i++;
        }

        console.log("\n\n");

        // Trim trailing floating system messages (common for reminders)
        const remainder = request.body.messages.slice(i);
        if (remainder.length === 0) return; // nothing to reconstruct
        let endNonSys = remainder.length - 1;
        while (endNonSys >= 0 && remainder[endNonSys]?.role === 'system') endNonSys--;
        const trailingSys = endNonSys < remainder.length - 1 ? remainder.slice(endNonSys + 1) : [];
        const windowMsgs = remainder.slice(0, endNonSys + 1);
        if (trailingSys.length > 0) {
            const tailPrev = msgPreview(trailingSys.at(-1));
            const windowTailNow = windowMsgs.length ? `${msgPreview(windowMsgs.at(-1))}` : '(empty window)';
            console.log('[FirstAnchor] Trimmed trailing system messages:', trailingSys.length, '| last trailing sys:', tailPrev, '| window tail is now at:', windowTailNow);
        }
        if (windowMsgs.length === 0) return; // nothing to reconstruct after trimming

        // Build fingerprints for current window
        const windowFP = windowMsgs.map(fingerprintMessage);

        // TTL
        const ttlSec = getConfigValue('claude.firstAnchorCaching.ttlSeconds', 3600, 'number');
        const ttlMs = Math.max(0, Number.isFinite(ttlSec) ? ttlSec * 1000 : 3600000);

        purgeExpiredFirstAnchor();

        // Find best match globally (max right-ended overlap)
        let best = null;
        let bestK = 0;
        for (const entry of FIRST_ANCHOR_STORE) {
            const k = longestRightEndedOverlap(entry.fps, windowFP);
            if (k > bestK) { bestK = k; best = entry; }
        }

        if (best && bestK > 0) {
            // Extend lynchpin with new right tail
            const tailMsgs = windowMsgs.slice(bestK);
            const tailFPs = windowFP.slice(bestK);
            if (tailMsgs.length > 0) {
                // Logging: narrate tail movement for clarity
                const prevTail = best.msgs.length ? `${msgPreview(best.msgs.at(-1))}` : '(empty)';
                const incomingHead = `${msgPreview(tailMsgs[0])}`;
                console.log('[FirstAnchor] Lynchpin match. Overlap=', bestK);
                console.log('[FirstAnchor] PrevTail:', prevTail);
                console.log('[FirstAnchor] ExtendStart:', incomingHead);

                best.msgs.push(...tailMsgs);
                best.fps.push(...tailFPs);

                const newTail = `${msgPreview(best.msgs.at(-1))}`;
                console.log('[FirstAnchor] Fast-forward -> NewTail:', newTail, '| ExtendCount=', tailMsgs.length, '| LynchpinLen=', best.msgs.length);
            } else {
                console.log('[FirstAnchor] Lynchpin hit with zero extension (perfect match).');
            }
            best.expireAt = Date.now() + ttlMs; // refresh TTL

            // Rebuild full context: system + lynchpin
            request.body.messages = [...sys, ...best.msgs, ...trailingSys];
            const sysPrev = sys.length ? `${msgPreview(sys.at(-1))}` : '(no system)';
            console.log('[FirstAnchor] Reconstructed context applied. System last:', sysPrev, '| Lynchpin len:', best.msgs.length);
            return; // all or nothing
        }

        // No right-ended overlap: try rewind if enabled and we can find a contiguous subsequence match
        const allowRewind = getConfigValue('claude.firstAnchorCaching.allowRewind', true, 'boolean');
        if (allowRewind && best?.fps?.length) {
            const s = indexOfSubsequence(best.fps, windowFP);
            if (s !== -1) {
                const e = s + windowFP.length - 1;
                const prevTail = best.msgs.length ? `${msgPreview(best.msgs.at(-1))}` : '(empty)';
                const newTail = `${msgPreview(best.msgs[e])}`;
                const cutCount = Math.max(0, best.msgs.length - (e + 1));
                const prevIndex = best.msgs.length - 1;
                const rewindToIdx = e;
                const jumpBack = Math.max(0, prevIndex - rewindToIdx);
                console.log('[FirstAnchor] Rewind detected. Jumping back', jumpBack, 'steps to:', newTail);
                console.log(`[FirstAnchor] PrevTail (index ${prevIndex}):`, prevTail);
                console.log(`[FirstAnchor] PrevTail (rewind to ${prevIndex - jumpBack}):`, newTail);
                console.log('[FirstAnchor] CutCount=', cutCount, '| LynchpinLen=', e + 1);

                best.msgs = best.msgs.slice(0, e + 1);
                best.fps = best.fps.slice(0, e + 1);
                best.expireAt = Date.now() + ttlMs;

                request.body.messages = [...sys, ...best.msgs, ...trailingSys];
                console.log('[FirstAnchor] Reconstructed context applied after rewind. Tail is now at:', newTail);
                return;
            }
        }

        // Rewind + edit: allow prefix match of the window against the lynchpin when only the tail differs
        if (allowRewind && best?.fps?.length) {
            const windowLen = windowFP.length;
            if (windowLen >= 2) {
                const pref = findLongestPrefixMatch(best.fps, windowFP);
                const appendedCount = windowLen - pref.length;
                if (pref.startIndex !== -1 && pref.length >= windowLen - 1 && appendedCount > 0) {
                    const cutTo = pref.startIndex + pref.length - 1;
                    const prevTail = best.msgs.length ? `${msgPreview(best.msgs.at(-1))}` : '(empty)';
                    const newTailMsgObj = windowMsgs[windowLen - 1];
                    const newTail = `${msgPreview(newTailMsgObj)}`;
                    const cutCount2 = Math.max(0, best.msgs.length - (cutTo + 1));
                    console.log('[FirstAnchor] Rewind + edit detected. Prefix match=', pref.length);
                    console.log(`[FirstAnchor] PrevTail (index ${best.msgs.length - 1}):`, prevTail);
                    console.log(`[FirstAnchor] Cut to index ${cutTo}, append edited messages:`, appendedCount);

                    best.msgs = best.msgs.slice(0, cutTo + 1).concat(windowMsgs.slice(pref.length));
                    best.fps = best.fps.slice(0, cutTo + 1).concat(windowFP.slice(pref.length));
                    best.expireAt = Date.now() + ttlMs;

                    request.body.messages = [...sys, ...best.msgs, ...trailingSys];
                    console.log('[FirstAnchor] Reconstructed context applied after rewind+edit. NewTail:', newTail, '| LynchpinLen=', best.msgs.length);
                    return;
                }
            }
        }

        // No match found -> seed a new lynchpin from current window, but do not modify request
        const latestTail = windowMsgs.length ? msgPreview(windowMsgs.at(-1)) : '(empty window)';
        console.log('[FirstAnchor] Tail(latest):', latestTail);
        console.log('[FirstAnchor] No lynchpin. Seeded. WindowLen=', windowMsgs.length);
        FIRST_ANCHOR_STORE.push({ msgs: windowMsgs.slice(), fps: windowFP.slice(), expireAt: Date.now() + ttlMs });
    } catch (e) {
        console.warn('FirstAnchor reconstruction failed:', e);
        // Fail open: do nothing
    }
}
// --- End First-Anchor Context Reconstruction ---

const API_VERTEX_AI = 'https://us-central1-aiplatform.googleapis.com';
const API_AI21 = 'https://api.ai21.com/studio/v1';
const API_NANOGPT = 'https://nano-gpt.com/api/v1';
const API_DEEPSEEK = 'https://api.deepseek.com/beta';
const API_XAI = 'https://api.x.ai/v1';
const API_AIMLAPI = 'https://api.aimlapi.com/v1';
const API_POLLINATIONS = 'https://text.pollinations.ai/openai';
const API_MOONSHOT = 'https://api.moonshot.ai/v1';
const API_FIREWORKS = 'https://api.fireworks.ai/inference/v1';
const API_COMETAPI = 'https://api.cometapi.com/v1';

/**
 * Gets OpenRouter transforms based on the request.
 * @param {import('express').Request} request Express request
 * @returns {string[] | undefined} OpenRouter transforms
 */
function getOpenRouterTransforms(request) {
    switch (request.body.middleout) {
        case 'on':
            return ['middle-out'];
        case 'off':
            return [];
        case 'auto':
            return undefined;
    }
}

/**
 * Gets OpenRouter plugins based on the request.
 * @param {import('express').Request} request
 * @returns {any[]} OpenRouter plugins
 */
function getOpenRouterPlugins(request) {
    const plugins = [];

    if (request.body.enable_web_search) {
        plugins.push({ 'id': 'web' });
    }

    return plugins;
}

/**
 * Hacky way to use JSON schema only if json_object format is supported.
 * @param {object} bodyParams Additional body parameters
 * @param {object[]} messages Array of messages
 * @param {object} jsonSchema JSON schema object
 */
function setJsonObjectFormat(bodyParams, messages, jsonSchema) {
    bodyParams['response_format'] = {
        type: 'json_object',
    };
    const message = {
        role: 'user',
        content: `JSON schema for the response:\n${JSON.stringify(jsonSchema.value, null, 4)}`,
    };
    messages.push(message);
}

/**
 * Sends a request to Claude API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendClaudeRequest(request, response) {
    const apiUrl = new URL(request.body.reverse_proxy || API_CLAUDE).toString();
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.CLAUDE);
    const divider = '-'.repeat(process.stdout.columns);
    const enableSystemPromptCache = getConfigValue('claude.enableSystemPromptCache', false, 'boolean');
    let cachingAtDepth = getConfigValue('claude.cachingAtDepth', -1, 'number');
    // Disabled if not an integer or negative
    if (!Number.isInteger(cachingAtDepth) || cachingAtDepth < 0) {
        cachingAtDepth = -1;
    }

    if (!apiKey) {
        console.warn(color.red(`Claude API key is missing.\n${divider}`));
        return response.status(400).send({ error: true });
    }

    try {
        const controller = new AbortController();
        request.socket.removeAllListeners('close');
        request.socket.on('close', function () {
            controller.abort();
        });
        const additionalHeaders = {};
        const betaHeaders = ['output-128k-2025-02-19'];
        const useTools = Array.isArray(request.body.tools) && request.body.tools.length > 0;
        const useSystemPrompt = Boolean(request.body.claude_use_sysprompt);
        const convertedPrompt = convertClaudeMessages(request.body.messages, request.body.assistant_prefill, useSystemPrompt, useTools, getPromptNames(request));
        const useThinking = /^claude-(3-7|opus-4|sonnet-4)/.test(request.body.model);
        const useWebSearch = /^claude-(3-5|3-7|opus-4|sonnet-4)/.test(request.body.model) && Boolean(request.body.enable_web_search);
        const isOpus41 = /^claude-opus-4-1/.test(request.body.model);
        const cacheTTL = getConfigValue('claude.extendedTTL', false, 'boolean') ? '1h' : '5m';
        let fixThinkingPrefill = false;
        // Add custom stop sequences
        const stopSequences = [];
        if (Array.isArray(request.body.stop)) {
            stopSequences.push(...request.body.stop);
        }

        const requestBody = {
            /** @type {any} */ system: [],
            messages: convertedPrompt.messages,
            model: request.body.model,
            max_tokens: request.body.max_tokens,
            stop_sequences: stopSequences,
            temperature: request.body.temperature,
            top_p: request.body.top_p,
            top_k: request.body.top_k,
            stream: request.body.stream,
        };
        if (useSystemPrompt) {
            if (enableSystemPromptCache && Array.isArray(convertedPrompt.systemPrompt) && convertedPrompt.systemPrompt.length) {
                convertedPrompt.systemPrompt[convertedPrompt.systemPrompt.length - 1]['cache_control'] = { type: 'ephemeral', ttl: cacheTTL };
            }

            requestBody.system = convertedPrompt.systemPrompt;
        } else {
            delete requestBody.system;
        }
        if (useTools) {
            betaHeaders.push('tools-2024-05-16');
            requestBody.tool_choice = { type: request.body.tool_choice };
            requestBody.tools = request.body.tools
                .filter(tool => tool.type === 'function')
                .map(tool => tool.function)
                .map(fn => ({ name: fn.name, description: fn.description, input_schema: flattenSchema(fn.parameters, request.body.chat_completion_source) }));

            if (enableSystemPromptCache && requestBody.tools.length) {
                requestBody.tools[requestBody.tools.length - 1]['cache_control'] = { type: 'ephemeral', ttl: cacheTTL };
            }
        }

        // Structured output is a forced tool
        if (request.body.json_schema) {
            const jsonTool = {
                name: request.body.json_schema.name,
                description: request.body.json_schema.description || 'Well-formed JSON object',
                input_schema: request.body.json_schema.value,
            };
            requestBody.tools = [...(requestBody.tools || []), jsonTool];
            requestBody.tool_choice = { type: 'tool', name: request.body.json_schema.name };
        }

        if (useWebSearch) {
            const webSearchTool = [{
                'type': 'web_search_20250305',
                'name': 'web_search',
            }];
            requestBody.tools = [...webSearchTool, ...(requestBody.tools || [])];
        }

        if (cachingAtDepth !== -1) {
            cachingAtDepthForClaude(convertedPrompt.messages, cachingAtDepth, cacheTTL);
        }

        if (enableSystemPromptCache || cachingAtDepth !== -1) {
            betaHeaders.push('prompt-caching-2024-07-31');
            betaHeaders.push('extended-cache-ttl-2025-04-11');
        }

        if (isOpus41){
            if (requestBody.top_p < 1) {
                delete requestBody.temperature;
            } else {
                delete requestBody.top_p;
            }
        }

        const reasoningEffort = request.body.reasoning_effort;
        const budgetTokens = calculateClaudeBudgetTokens(requestBody.max_tokens, reasoningEffort, requestBody.stream);

        if (useThinking && Number.isInteger(budgetTokens)) {
            // No prefill when thinking
            fixThinkingPrefill = true;
            const minThinkTokens = 1024;
            if (requestBody.max_tokens <= minThinkTokens) {
                const newValue = requestBody.max_tokens + minThinkTokens;
                console.warn(color.yellow(`Claude thinking requires a minimum of ${minThinkTokens} response tokens.`));
                console.info(color.blue(`Increasing response length to ${newValue}.`));
                requestBody.max_tokens = newValue;
            }
            requestBody.thinking = {
                type: 'enabled',
                budget_tokens: budgetTokens,
            };

            // NO I CAN'T SILENTLY IGNORE THE TEMPERATURE.
            delete requestBody.temperature;
            delete requestBody.top_p;
            delete requestBody.top_k;
        }

        if (fixThinkingPrefill && convertedPrompt.messages.length && convertedPrompt.messages[convertedPrompt.messages.length - 1].role === 'assistant') {
            convertedPrompt.messages[convertedPrompt.messages.length - 1].role = 'user';
        }

        if (betaHeaders.length) {
            additionalHeaders['anthropic-beta'] = betaHeaders.join(',');
        }

        console.debug('Claude request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/messages', {
            method: 'POST',
            signal: controller.signal,
            body: JSON.stringify(requestBody),
            headers: {
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01',
                'x-api-key': apiKey,
                ...additionalHeaders,
            },
        });

        if (request.body.stream) {
            // Pipe remote SSE stream to Express response
            forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const generateResponseText = await generateResponse.text();
                console.warn(color.red(`Claude API returned error: ${generateResponse.status} ${generateResponse.statusText}\n${generateResponseText}\n${divider}`));
                return response.status(500).send({ error: true });
            }

            /** @type {any} */
            const generateResponseJson = await generateResponse.json();
            const responseText = generateResponseJson?.content?.[0]?.text || '';
            console.debug('Claude response:', generateResponseJson);

            // Wrap it back to OAI format + save the original content
            const reply = { choices: [{ 'message': { 'content': responseText } }], content: generateResponseJson.content };
            return response.send(reply);
        }
    } catch (error) {
        console.error(color.red(`Error communicating with Claude: ${error}\n${divider}`));
        if (!response.headersSent) {
            return response.status(500).send({ error: true });
        }
    }
}

/**
 * Sends a request to Google AI API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendMakerSuiteRequest(request, response) {
    const useVertexAi = request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.VERTEXAI;
    const apiName = useVertexAi ? 'Google Vertex AI' : 'Google AI Studio';
    let apiUrl;
    let apiKey;

    let authHeader;
    let authType;

    if (useVertexAi) {
        apiUrl = new URL(request.body.reverse_proxy || API_VERTEX_AI);

        try {
            const auth = await getVertexAIAuth(request);
            authHeader = auth.authHeader;
            authType = auth.authType;
            console.debug(`Using Vertex AI authentication type: ${authType}`);
        } catch (error) {
            console.warn(`${apiName} authentication failed: ${error.message}`);
            return response.status(400).send({ error: true, message: error.message });
        }
    } else {
        apiUrl = new URL(request.body.reverse_proxy || API_MAKERSUITE);
        apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MAKERSUITE);

        if (!request.body.reverse_proxy && !apiKey) {
            console.warn(`${apiName} API key is missing.`);
            return response.status(400).send({ error: true });
        }

        authHeader = `Bearer ${apiKey}`;
        authType = 'api_key';
    }

    const model = String(request.body.model);
    const stream = Boolean(request.body.stream);
    const enableWebSearch = Boolean(request.body.enable_web_search);
    const requestImages = Boolean(request.body.request_images);
    const reasoningEffort = String(request.body.reasoning_effort);
    const includeReasoning = Boolean(request.body.include_reasoning);
    const isGemma = model.includes('gemma');
    const isLearnLM = model.includes('learnlm');

    const responseMimeType = request.body.responseMimeType ?? (request.body.json_schema ? 'application/json' : undefined);
    const responseSchema = request.body.responseSchema ?? (request.body.json_schema ? request.body.json_schema.value : undefined);

    const generationConfig = {
        stopSequences: request.body.stop,
        candidateCount: 1,
        maxOutputTokens: request.body.max_tokens,
        temperature: request.body.temperature,
        topP: request.body.top_p,
        topK: request.body.top_k || undefined,
        responseMimeType: responseMimeType,
        responseSchema: responseSchema,
        seed: request.body.seed,
    };

    function getGeminiBody() {
        // #region UGLY MODEL LISTS AREA
        const imageGenerationModels = [
            'gemini-2.0-flash-exp',
            'gemini-2.0-flash-exp-image-generation',
            'gemini-2.0-flash-preview-image-generation',
            'gemini-2.5-flash-image-preview',
        ];

        // These models do not support setting the threshold to OFF at all.
        const blockNoneModels = [
            'gemini-1.5-pro-001',
            'gemini-1.5-flash-001',
            'gemini-1.5-flash-8b-exp-0827',
            'gemini-1.5-flash-8b-exp-0924',
        ];

        const isThinkingConfigModel = m => /^gemini-2.5-(flash|pro)/.test(m) && !/-image-preview$/.test(m);

        const noSearchModels = [
            'gemini-2.0-flash-lite',
            'gemini-2.0-flash-lite-001',
            'gemini-2.0-flash-lite-preview-02-05',
            'gemini-1.5-flash-8b-exp-0924',
            'gemini-1.5-flash-8b-exp-0827',
        ];
        // #endregion

        if (!Array.isArray(generationConfig.stopSequences) || !generationConfig.stopSequences.length) {
            delete generationConfig.stopSequences;
        }

        const enableImageModality = requestImages && imageGenerationModels.includes(model);
        if (enableImageModality) {
            generationConfig.responseModalities = ['text', 'image'];
        }

        const useSystemPrompt = !enableImageModality && !isGemma && request.body.use_makersuite_sysprompt;

        const tools = [];
        const prompt = convertGooglePrompt(request.body.messages, model, useSystemPrompt, getPromptNames(request));
        let safetySettings = GEMINI_SAFETY;

        if (blockNoneModels.includes(model)) {
            safetySettings = GEMINI_SAFETY.map(setting => ({ ...setting, threshold: 'BLOCK_NONE' }));
        }

        if (enableWebSearch && !enableImageModality && !isGemma && !isLearnLM && !noSearchModels.includes(model)) {
            const searchTool = model.includes('1.5')
                ? ({ google_search_retrieval: {} })
                : ({ google_search: {} });
            tools.push(searchTool);
        }

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0 && !enableImageModality && !isGemma) {
            const functionDeclarations = [];
            for (const tool of request.body.tools) {
                if (tool.type === 'function') {
                    if (tool.function.parameters?.$schema) {
                        delete tool.function.parameters.$schema;
                    }
                    if (tool.function.parameters?.properties && Object.keys(tool.function.parameters.properties).length === 0) {
                        delete tool.function.parameters;
                    }
                    functionDeclarations.push(tool.function);
                }
            }
            tools.push({ function_declarations: functionDeclarations });
        }

        if (isThinkingConfigModel(model)) {
            const thinkingConfig = { includeThoughts: includeReasoning };

            const thinkingBudget = calculateGoogleBudgetTokens(generationConfig.maxOutputTokens, reasoningEffort, model);
            if (Number.isInteger(thinkingBudget)) {
                thinkingConfig.thinkingBudget = thinkingBudget;
            }

            // Vertex doesn't allow mixing disabled thinking with includeThoughts
            if (useVertexAi && thinkingBudget === 0 && thinkingConfig.includeThoughts) {
                console.info('Thinking budget is 0, but includeThoughts is true. Thoughts will not be included in the response.');
                thinkingConfig.includeThoughts = false;
            }

            generationConfig.thinkingConfig = thinkingConfig;
        }

        let body = {
            contents: prompt.contents,
            safetySettings: safetySettings,
            generationConfig: generationConfig,
        };

        if (useSystemPrompt && Array.isArray(prompt.system_instruction.parts) && prompt.system_instruction.parts.length) {
            body.systemInstruction = prompt.system_instruction;
        }

        if (tools.length) {
            body.tools = tools;
        }

        return body;
    }

    const body = getGeminiBody();
    console.debug(`${apiName} request:`, body);

    try {
        const controller = new AbortController();
        request.socket.removeAllListeners('close');
        request.socket.on('close', function () {
            controller.abort();
        });

        const apiVersion = getConfigValue('gemini.apiVersion', 'v1beta');
        const responseType = (stream ? 'streamGenerateContent' : 'generateContent');

        let url;
        let headers = {
            'Content-Type': 'application/json',
        };

        if (useVertexAi) {
            if (authType === 'express') {
                // For Express mode (API key authentication), use the key parameter
                const keyParam = authHeader.replace('Bearer ', '');
                const region = request.body.vertexai_region || 'us-central1';
                const projectId = request.body.vertexai_express_project_id;
                const baseUrl = region === 'global'
                    ? 'https://aiplatform.googleapis.com'
                    : `https://${region}-aiplatform.googleapis.com`;
                url = projectId
                    ? `https://aiplatform.googleapis.com/v1/projects/${projectId}/locations/${region}/publishers/google/models/${model}:generateContent?key=${keyParam}${stream ? '&alt=sse' : ''}`
                    : `${baseUrl}/v1/publishers/google/models/${model}:generateContent?key=${keyParam}${stream ? '&alt=sse' : ''}`;
            } else if (authType === 'full') {
                // For Full mode (service account authentication), use project-specific URL
                // Get project ID from Service Account JSON
                const serviceAccountJson = readSecret(request.user.directories, SECRET_KEYS.VERTEXAI_SERVICE_ACCOUNT);
                if (!serviceAccountJson) {
                    console.warn('Vertex AI Service Account JSON is missing.');
                    return response.status(400).send({ error: true });
                }

                let projectId;
                try {
                    const serviceAccount = JSON.parse(serviceAccountJson);
                    projectId = getProjectIdFromServiceAccount(serviceAccount);
                } catch (error) {
                    console.error('Failed to extract project ID from Service Account JSON:', error);
                    return response.status(400).send({ error: true });
                }
                const region = request.body.vertexai_region || 'us-central1';
                // Handle global region differently - no region prefix in hostname
                if (region === 'global') {
                    url = `https://aiplatform.googleapis.com/v1/projects/${projectId}/locations/${region}/publishers/google/models/${model}:${responseType}${stream ? '?alt=sse' : ''}`;
                } else {
                    url = `https://${region}-aiplatform.googleapis.com/v1/projects/${projectId}/locations/${region}/publishers/google/models/${model}:${responseType}${stream ? '?alt=sse' : ''}`;
                }
                headers['Authorization'] = authHeader;
            } else {
                // For proxy mode, use the original URL with Authorization header
                url = `${apiUrl.toString().replace(/\/$/, '')}/v1/publishers/google/models/${model}:${responseType}${stream ? '?alt=sse' : ''}`;
                headers['Authorization'] = authHeader;
            }
        } else {
            url = `${apiUrl.toString().replace(/\/$/, '')}/${apiVersion}/models/${model}:${responseType}?key=${apiKey}${stream ? '&alt=sse' : ''}`;
        }

        const generateResponse = await fetch(url, {
            body: JSON.stringify(body),
            method: 'POST',
            headers: headers,
            signal: controller.signal,
        });

        if (stream) {
            try {
                // Pipe remote SSE stream to Express response
                forwardFetchResponse(generateResponse, response);
            } catch (error) {
                console.error('Error forwarding streaming response:', error);
                if (!response.headersSent) {
                    return response.status(500).send({ error: true });
                }
            }
        } else {
            if (!generateResponse.ok) {
                console.warn(`${apiName} API returned error: ${generateResponse.status} ${generateResponse.statusText} ${await generateResponse.text()}`);
                return response.status(500).send({ error: true });
            }

            /** @type {any} */
            const generateResponseJson = await generateResponse.json();

            const candidates = generateResponseJson?.candidates;
            if (!candidates || candidates.length === 0) {
                let message = `${apiName} API returned no candidate`;
                console.warn(message, generateResponseJson);
                if (generateResponseJson?.promptFeedback?.blockReason) {
                    message += `\nPrompt was blocked due to : ${generateResponseJson.promptFeedback.blockReason}`;
                }
                return response.send({ error: { message } });
            }

            const responseContent = candidates[0].content ?? candidates[0].output;
            const functionCall = (candidates?.[0]?.content?.parts ?? []).some(part => part.functionCall);
            const inlineData = (candidates?.[0]?.content?.parts ?? []).some(part => part.inlineData);
            console.debug(`${apiName} response:`, util.inspect(generateResponseJson, { depth: 5, colors: true }));

            const responseText = typeof responseContent === 'string' ? responseContent : responseContent?.parts?.filter(part => !part.thought)?.map(part => part.text)?.join('\n\n');
            if (!responseText && !functionCall && !inlineData) {
                let message = `${apiName} Candidate text empty`;
                console.warn(message, generateResponseJson);
                return response.send({ error: { message } });
            }

            // Wrap it back to OAI format
            const reply = { choices: [{ 'message': { 'content': responseText } }], responseContent };
            return response.send(reply);
        }
    } catch (error) {
        console.error(`Error communicating with ${apiName} API:`, error);
        if (!response.headersSent) {
            return response.status(500).send({ error: true });
        }
    }
}

/**
 * Sends a request to AI21 API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendAI21Request(request, response) {
    if (!request.body) return response.sendStatus(400);

    const apiKey = readSecret(request.user.directories, SECRET_KEYS.AI21);
    if (!apiKey) {
        console.warn('AI21 API key is missing.');
        return response.status(400).send({ error: true });
    }

    const bodyParams = {};
    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });
    // Hack to support JSON schema
    if (request.body.json_schema) {
        bodyParams.response_format = {
            type: 'json_object',
        };
        const message = {
            role: 'user',
            content: `JSON schema for the response:\n${JSON.stringify(request.body.json_schema.value, null, 4)}`,
        };
        request.body.messages.push(message);
    }
    const convertedPrompt = convertAI21Messages(request.body.messages, getPromptNames(request));
    const body = {
        messages: convertedPrompt,
        model: request.body.model,
        max_tokens: request.body.max_tokens,
        temperature: request.body.temperature,
        top_p: request.body.top_p,
        stop: request.body.stop,
        stream: request.body.stream,
        tools: request.body.tools,
        ...bodyParams,
    };
    const options = {
        method: 'POST',
        headers: {
            accept: 'application/json',
            'content-type': 'application/json',
            Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
    };

    console.debug('AI21 request:', body);

    try {
        const generateResponse = await fetch(API_AI21 + '/chat/completions', options);
        if (request.body.stream) {
            forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`AI21 API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('AI21 response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with AI21 API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to MistralAI API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendMistralAIRequest(request, response) {
    const apiUrl = new URL(request.body.reverse_proxy || API_MISTRAL).toString();
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MISTRALAI);

    if (!apiKey) {
        console.warn('MistralAI API key is missing.');
        return response.status(400).send({ error: true });
    }

    try {
        const messages = convertMistralMessages(request.body.messages, getPromptNames(request));
        const controller = new AbortController();
        request.socket.removeAllListeners('close');
        request.socket.on('close', function () {
            controller.abort();
        });

        const requestBody = {
            'model': request.body.model,
            'messages': messages,
            'temperature': request.body.temperature,
            'top_p': request.body.top_p,
            'frequency_penalty': request.body.frequency_penalty,
            'presence_penalty': request.body.presence_penalty,
            'max_tokens': request.body.max_tokens,
            'stream': request.body.stream,
            'safe_prompt': request.body.safe_prompt,
            'random_seed': request.body.seed === -1 ? undefined : request.body.seed,
            'stop': Array.isArray(request.body.stop) && request.body.stop.length > 0 ? request.body.stop : undefined,
        };

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            requestBody['tools'] = request.body.tools;
            requestBody['tool_choice'] = request.body.tool_choice;
        }

        if (request.body.json_schema) {
            requestBody['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    description: request.body.json_schema.description,
                    schema: request.body.json_schema.value,
                    strict: request.body.json_schema.strict ?? true,
                },
            };
        }

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
            timeout: 0,
        };

        console.debug('MisralAI request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);
        if (request.body.stream) {
            forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`MistralAI API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('MistralAI response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with MistralAI API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to Cohere API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendCohereRequest(request, response) {
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.COHERE);
    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    if (!apiKey) {
        console.warn('Cohere API key is missing.');
        return response.status(400).send({ error: true });
    }

    try {
        const convertedHistory = convertCohereMessages(request.body.messages, getPromptNames(request));
        const tools = [];

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            tools.push(...request.body.tools);
            tools.forEach(tool => {
                if (tool?.function?.parameters?.$schema) {
                    delete tool.function.parameters.$schema;
                }
            });
        }

        // https://docs.cohere.com/reference/chat
        const requestBody = {
            stream: Boolean(request.body.stream),
            model: request.body.model,
            messages: convertedHistory.chatHistory,
            temperature: request.body.temperature,
            max_tokens: request.body.max_tokens,
            k: request.body.top_k,
            p: request.body.top_p,
            seed: request.body.seed,
            stop_sequences: request.body.stop,
            frequency_penalty: request.body.frequency_penalty,
            presence_penalty: request.body.presence_penalty,
            documents: [],
            tools: tools,
        };

        const canDoSafetyMode = String(request.body.model).endsWith('08-2024');
        if (canDoSafetyMode) {
            requestBody.safety_mode = 'OFF';
        }

        if (request.body.json_schema) {
            requestBody.response_format = {
                type: 'json_schema',
                schema: request.body.json_schema.value,
            };
        }

        console.debug('Cohere request:', requestBody);

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
            timeout: 0,
        };

        const apiUrl = API_COHERE_V2 + '/chat';

        if (request.body.stream) {
            const stream = await fetch(apiUrl, config);
            forwardFetchResponse(stream, response);
        } else {
            const generateResponse = await fetch(apiUrl, config);
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`Cohere API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('Cohere response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with Cohere API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to DeepSeek API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendDeepSeekRequest(request, response) {
    const apiUrl = new URL(request.body.reverse_proxy || API_DEEPSEEK).toString();
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.DEEPSEEK);

    if (!apiKey && !request.body.reverse_proxy) {
        console.warn('DeepSeek API key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        let bodyParams = {};

        if (request.body.logprobs > 0) {
            bodyParams['top_logprobs'] = request.body.logprobs;
            bodyParams['logprobs'] = true;
        }

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;

            // DeepSeek doesn't permit empty required arrays
            bodyParams.tools.forEach(tool => {
                const required = tool?.function?.parameters?.required;
                if (Array.isArray(required) && required.length === 0) {
                    delete tool.function.parameters.required;
                }
            });
        }

        // Hack to support JSON schema
        if (request.body.json_schema) {
            bodyParams.response_format = {
                type: 'json_object',
            };
            const message = {
                role: 'user',
                content: `JSON schema for the response:\n${JSON.stringify(request.body.json_schema.value, null, 4)}`,
            };
            request.body.messages.push(message);
        }

        const postProcessType = String(request.body.model).endsWith('-reasoner')
            ? PROMPT_PROCESSING_TYPE.STRICT_TOOLS
            : PROMPT_PROCESSING_TYPE.SEMI_TOOLS;
        const processedMessages = addAssistantPrefix(postProcessPrompt(request.body.messages, postProcessType, getPromptNames(request)), bodyParams.tools, 'prefix');

        const requestBody = {
            'messages': processedMessages,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.max_tokens,
            'stream': request.body.stream,
            'presence_penalty': request.body.presence_penalty,
            'frequency_penalty': request.body.frequency_penalty,
            'top_p': request.body.top_p,
            'stop': request.body.stop,
            'seed': request.body.seed,
            ...bodyParams,
        };

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug('DeepSeek request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);

        if (request.body.stream) {
            forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`DeepSeek API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('DeepSeek response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with DeepSeek API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to XAI API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendXaiRequest(request, response) {
    const apiUrl = new URL(request.body.reverse_proxy || API_XAI).toString();
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.XAI);

    if (!apiKey && !request.body.reverse_proxy) {
        console.warn('xAI API key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        let bodyParams = {};

        if (request.body.logprobs > 0) {
            bodyParams['top_logprobs'] = request.body.logprobs;
            bodyParams['logprobs'] = true;
        }

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;
        }

        if (Array.isArray(request.body.stop) && request.body.stop.length > 0) {
            bodyParams['stop'] = request.body.stop;
        }

        if (request.body.reasoning_effort && ['grok-3-mini-beta', 'grok-3-mini-fast-beta'].includes(request.body.model)) {
            bodyParams['reasoning_effort'] = request.body.reasoning_effort === 'high' ? 'high' : 'low';
        }

        if (request.body.enable_web_search) {
            bodyParams['search_parameters'] = {
                mode: 'on',
                sources: [
                    { type: 'web', safe_search: false },
                    { type: 'news', safe_search: false },
                    { type: 'x' },
                ],
            };
        }

        if (request.body.json_schema) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    strict: request.body.json_schema.strict ?? true,
                    schema: request.body.json_schema.value,
                },
            };
        }

        const processedMessages = request.body.messages = convertXAIMessages(request.body.messages, getPromptNames(request));

        const requestBody = {
            'messages': processedMessages,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.max_tokens,
            'max_completion_tokens': request.body.max_completion_tokens,
            'stream': request.body.stream,
            'presence_penalty': request.body.presence_penalty,
            'frequency_penalty': request.body.frequency_penalty,
            'top_p': request.body.top_p,
            'seed': request.body.seed,
            'n': request.body.n,
            ...bodyParams,
        };

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug('xAI request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);

        if (request.body.stream) {
            forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`xAI API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('xAI response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with xAI API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to AI/ML API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendAimlapiRequest(request, response) {
    const apiUrl = API_AIMLAPI;
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.AIMLAPI);

    if (!apiKey) {
        console.warn('AI/ML API key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        let bodyParams = {};

        if (request.body.logprobs > 0) {
            bodyParams['top_logprobs'] = request.body.logprobs;
            bodyParams['logprobs'] = true;
        }

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;
        }

        if (Array.isArray(request.body.stop) && request.body.stop.length > 0) {
            bodyParams['stop'] = request.body.stop;
        }

        if (request.body.reasoning_effort) {
            bodyParams['reasoning_effort'] = request.body.reasoning_effort;
        }

        if (request.body.json_schema) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    description: request.body.json_schema.description,
                    schema: request.body.json_schema.value,
                    strict: request.body.json_schema.strict ?? true,
                },
            };
        }

        const requestBody = {
            'messages': request.body.messages,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.max_tokens,
            'stream': request.body.stream,
            'presence_penalty': request.body.presence_penalty,
            'frequency_penalty': request.body.frequency_penalty,
            'top_p': request.body.top_p,
            'seed': request.body.seed,
            'n': request.body.n,
            ...bodyParams,
        };

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
                ...AIMLAPI_HEADERS,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug('AI/ML API request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);

        if (request.body.stream) {
            forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`AI/ML API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('AI/ML API response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with AI/ML API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

export const router = express.Router();

router.post('/status', async function (request, statusResponse) {
    if (!request.body) return statusResponse.sendStatus(400);

    let apiUrl;
    let apiKey;
    let headers;

    if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENAI) {
        apiUrl = new URL(request.body.reverse_proxy || API_OPENAI).toString();
        apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.OPENAI);
        headers = {};
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENROUTER) {
        apiUrl = 'https://openrouter.ai/api/v1';
        apiKey = readSecret(request.user.directories, SECRET_KEYS.OPENROUTER);
        // OpenRouter needs to pass the Referer and X-Title: https://openrouter.ai/docs#requests
        headers = { ...OPENROUTER_HEADERS };
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MISTRALAI) {
        apiUrl = new URL(request.body.reverse_proxy || API_MISTRAL).toString();
        apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MISTRALAI);
        headers = {};
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CUSTOM) {
        apiUrl = request.body.custom_url;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.CUSTOM);
        headers = {};
        mergeObjectWithYaml(headers, request.body.custom_include_headers);
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.COHERE) {
        apiUrl = API_COHERE_V1;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.COHERE);
        headers = {};
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.NANOGPT) {
        apiUrl = API_NANOGPT;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.NANOGPT);
        headers = {};
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.DEEPSEEK) {
        apiUrl = new URL(request.body.reverse_proxy || API_DEEPSEEK.replace('/beta', ''));
        apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.DEEPSEEK);
        headers = {};
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.XAI) {
        apiUrl = new URL(request.body.reverse_proxy || API_XAI);
        apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.XAI);
        headers = {};
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.AIMLAPI) {
        apiUrl = API_AIMLAPI;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.AIMLAPI);
        headers = { ...AIMLAPI_HEADERS };
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.POLLINATIONS) {
        apiUrl = 'https://text.pollinations.ai';
        apiKey = 'NONE';
        headers = {};
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.GROQ) {
        apiUrl = API_GROQ;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.GROQ);
        headers = {};
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.COMETAPI) {
        apiUrl = API_COMETAPI;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.COMETAPI);
        headers = {};
        throw new Error('This provider is temporarily disabled.');
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MOONSHOT) {
        apiUrl = API_MOONSHOT;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.MOONSHOT);
        headers = {};
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.FIREWORKS) {
        apiUrl = API_FIREWORKS;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.FIREWORKS);
        headers = {};
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MAKERSUITE) {
        apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MAKERSUITE);
        apiUrl = trimTrailingSlash(request.body.reverse_proxy || API_MAKERSUITE);
        const apiVersion = getConfigValue('gemini.apiVersion', 'v1beta');
        const modelsUrl = !apiKey && request.body.reverse_proxy
            ? `${apiUrl}/${apiVersion}/models`
            : `${apiUrl}/${apiVersion}/models?key=${apiKey}`;

        if (!apiKey && !request.body.reverse_proxy) {
            console.warn('Google AI Studio API key is missing.');
            return statusResponse.status(400).send({ error: true });
        }

        try {
            const response = await fetch(modelsUrl);

            if (response.ok) {
                /** @type {any} */
                const data = await response.json();
                // Transform Google AI Studio models to OpenAI format
                const models = data.models
                    ?.filter(model => model.supportedGenerationMethods?.includes('generateContent'))
                    ?.map(model => ({
                        id: model.name.replace('models/', ''),
                    })) || [];

                console.info('Available Google AI Studio models:', models.map(m => m.id));
                return statusResponse.send({ data: models });
            } else {
                console.warn('Google AI Studio models endpoint failed:', response.status, response.statusText);
                return statusResponse.send({ error: true, bypass: true, data: { data: [] } });
            }
        } catch (error) {
            console.error('Error fetching Google AI Studio models:', error);
            return statusResponse.send({ error: true, bypass: true, data: { data: [] } });
        }
    } else {
        console.warn('This chat completion source is not supported yet.');
        return statusResponse.status(400).send({ error: true });
    }

    if (!apiKey && !request.body.reverse_proxy && request.body.chat_completion_source !== CHAT_COMPLETION_SOURCES.CUSTOM) {
        console.warn('Chat Completion API key is missing.');
        return statusResponse.status(400).send({ error: true });
    }

    try {
        const response = await fetch(apiUrl + '/models', {
            method: 'GET',
            headers: {
                'Authorization': 'Bearer ' + apiKey,
                ...headers,
            },
        });

        if (response.ok) {
            /** @type {any} */
            let data = await response.json();

            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.POLLINATIONS && Array.isArray(data)) {
                data = { data: data.map(model => ({ id: model.name, ...model })) };
            }

            statusResponse.send(data);

            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.COHERE && Array.isArray(data?.models)) {
                data.data = data.models.map(model => ({ id: model.name, ...model }));
            }

            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENROUTER && Array.isArray(data?.data)) {
                let models = [];

                data.data.forEach(model => {
                    const context_length = model.context_length;
                    const tokens_dollar = Number(1 / (1000 * model.pricing?.prompt));
                    const tokens_rounded = (Math.round(tokens_dollar * 1000) / 1000).toFixed(0);
                    models[model.id] = {
                        tokens_per_dollar: tokens_rounded + 'k',
                        context_length: context_length,
                    };
                });

                console.info('Available OpenRouter models:', models);
            } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MISTRALAI) {
                const models = data?.data;
                console.info(models);
            } else {
                const models = data?.data;

                if (Array.isArray(models)) {
                    const modelIds = models.filter(x => x && typeof x === 'object').map(x => x.id).sort();
                    console.info('Available models:', modelIds);
                } else {
                    console.warn('Chat Completion endpoint did not return a list of models.');
                }
            }
        }
        else {
            console.error('Chat Completion status check failed. Either Access Token is incorrect or API endpoint is down.');
            statusResponse.send({ error: true, data: { data: [] } });
        }
    } catch (e) {
        console.error(e);

        if (!statusResponse.headersSent) {
            statusResponse.send({ error: true });
        } else {
            statusResponse.end();
        }
    }
});

router.post('/bias', async function (request, response) {
    if (!request.body || !Array.isArray(request.body))
        return response.sendStatus(400);

    try {
        const result = {};
        const model = getTokenizerModel(String(request.query.model || ''));

        // no bias for claude
        if (model == 'claude') {
            return response.send(result);
        }

        let encodeFunction;

        if (sentencepieceTokenizers.includes(model)) {
            const tokenizer = getSentencepiceTokenizer(model);
            const instance = await tokenizer?.get();
            if (!instance) {
                console.error('Tokenizer not initialized:', model);
                return response.send({});
            }
            encodeFunction = (text) => new Uint32Array(instance.encodeIds(text));
        } else if (webTokenizers.includes(model)) {
            const tokenizer = getWebTokenizer(model);
            const instance = await tokenizer?.get();
            if (!instance) {
                console.warn('Tokenizer not initialized:', model);
                return response.send({});
            }
            encodeFunction = (text) => new Uint32Array(instance.encode(text));
        } else {
            const tokenizer = getTiktokenTokenizer(model);
            encodeFunction = (tokenizer.encode.bind(tokenizer));
        }

        for (const entry of request.body) {
            if (!entry || !entry.text) {
                continue;
            }

            try {
                const tokens = getEntryTokens(entry.text, encodeFunction);

                for (const token of tokens) {
                    result[token] = entry.value;
                }
            } catch {
                console.warn('Tokenizer failed to encode:', entry.text);
            }
        }

        // not needed for cached tokenizers
        //tokenizer.free();
        return response.send(result);

        /**
         * Gets tokenids for a given entry
         * @param {string} text Entry text
         * @param {(string) => Uint32Array} encode Function to encode text to token ids
         * @returns {Uint32Array} Array of token ids
         */
        function getEntryTokens(text, encode) {
            // Get raw token ids from JSON array
            if (text.trim().startsWith('[') && text.trim().endsWith(']')) {
                try {
                    const json = JSON.parse(text);
                    if (Array.isArray(json) && json.every(x => typeof x === 'number')) {
                        return new Uint32Array(json);
                    }
                } catch {
                    // ignore
                }
            }

            // Otherwise, get token ids from tokenizer
            return encode(text);
        }
    } catch (error) {
        console.error(error);
        return response.send({});
    }
});


router.post('/generate', function (request, response) {
    if (!request.body) return response.status(400).send({ error: true });


	    // Optional transparent reconstruction to preserve cache hits under limited context
	    applyFirstAnchorReconstruction(request);

    const postProcessingType = request.body.custom_prompt_post_processing;
    if (Array.isArray(request.body.messages) && postProcessingType) {
        console.info('Applying custom prompt post-processing of type', postProcessingType);
        request.body.messages = postProcessPrompt(
            request.body.messages,
            postProcessingType,
            getPromptNames(request));
    }

    if (request.body.json_schema?.value) {
        request.body.json_schema.value = flattenSchema(request.body.json_schema.value, request.body.chat_completion_source);
    }

    switch (request.body.chat_completion_source) {
        case CHAT_COMPLETION_SOURCES.CLAUDE: return sendClaudeRequest(request, response);
        case CHAT_COMPLETION_SOURCES.AI21: return sendAI21Request(request, response);
        case CHAT_COMPLETION_SOURCES.MAKERSUITE: return sendMakerSuiteRequest(request, response);
        case CHAT_COMPLETION_SOURCES.VERTEXAI: return sendMakerSuiteRequest(request, response);
        case CHAT_COMPLETION_SOURCES.MISTRALAI: return sendMistralAIRequest(request, response);
        case CHAT_COMPLETION_SOURCES.COHERE: return sendCohereRequest(request, response);
        case CHAT_COMPLETION_SOURCES.DEEPSEEK: return sendDeepSeekRequest(request, response);
        case CHAT_COMPLETION_SOURCES.AIMLAPI: return sendAimlapiRequest(request, response);
        case CHAT_COMPLETION_SOURCES.XAI: return sendXaiRequest(request, response);
    }

    let apiUrl;
    let apiKey;
    let headers;
    let bodyParams;
    const isTextCompletion = Boolean(request.body.model && TEXT_COMPLETION_MODELS.includes(request.body.model)) || typeof request.body.messages === 'string';

    if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENAI) {
        apiUrl = new URL(request.body.reverse_proxy || API_OPENAI).toString();
        apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.OPENAI);
        headers = {};
        bodyParams = {
            logprobs: request.body.logprobs,
            top_logprobs: undefined,
        };

        // Adjust logprobs params for Chat Completions API, which expects { top_logprobs: number; logprobs: boolean; }
        if (!isTextCompletion && bodyParams.logprobs > 0) {
            bodyParams.top_logprobs = bodyParams.logprobs;
            bodyParams.logprobs = true;
        }

        if (getConfigValue('openai.randomizeUserId', false, 'boolean')) {
            bodyParams['user'] = uuidv4();
        }
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENROUTER) {
        apiUrl = 'https://openrouter.ai/api/v1';
        apiKey = readSecret(request.user.directories, SECRET_KEYS.OPENROUTER);
        // OpenRouter needs to pass the Referer and X-Title: https://openrouter.ai/docs#requests
        headers = { ...OPENROUTER_HEADERS };
        bodyParams = {
            'transforms': getOpenRouterTransforms(request),
            'plugins': getOpenRouterPlugins(request),
            'include_reasoning': Boolean(request.body.include_reasoning),
        };

        // Ensure OpenRouter streams include usage in the final event
        if (request.body.stream) {
            bodyParams['stream_options'] = { include_usage: true };
        }

        if (request.body.min_p !== undefined) {
            bodyParams['min_p'] = request.body.min_p;
        }

        if (request.body.top_a !== undefined) {
            bodyParams['top_a'] = request.body.top_a;
        }

        if (request.body.repetition_penalty !== undefined) {
            bodyParams['repetition_penalty'] = request.body.repetition_penalty;
        }

        if (Array.isArray(request.body.provider) && request.body.provider.length > 0) {
            bodyParams['provider'] = {
                allow_fallbacks: request.body.allow_fallbacks ?? true,
                order: request.body.provider ?? [],
            };
        }

        if (request.body.use_fallback) {
            bodyParams['route'] = 'fallback';
        }

        if (request.body.reasoning_effort) {
            bodyParams['reasoning'] = { effort: request.body.reasoning_effort };
        }

        if (request.body.json_schema) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    strict: request.body.json_schema.strict ?? true,
                    schema: request.body.json_schema.value,
                },
            };
        }

        const cachingAtDepth = getConfigValue('claude.cachingAtDepth', -1, 'number');
        const isClaude3or4 = /anthropic\/claude-(3|opus-4|sonnet-4)/.test(request.body.model);
        const cacheTTL = getConfigValue('claude.extendedTTL', false, 'boolean') ? '1h' : '5m';
        if (Number.isInteger(cachingAtDepth) && cachingAtDepth >= 0 && isClaude3or4) {
            cachingAtDepthForOpenRouterClaude(request.body.messages, cachingAtDepth, cacheTTL);
        }

        // Apply system prompt caching for Claude via OpenRouter when enabled in config
        const enableSystemPromptCache = getConfigValue('claude.enableSystemPromptCache', false, 'boolean');
        if (enableSystemPromptCache && isClaude3or4 && Array.isArray(request.body.messages) && request.body.messages.length) {
            console.log(`System prompt caching enabled for Claude model: ${request.body.model}`);

            // Tag the last system message in the leading system segment (before the first non-system)
            let leadingSystemCount = 0;
            for (let i = 0; i < request.body.messages.length; i++) {
                const msg = request.body.messages[i];
                if (msg && msg.role === 'system') {
                    leadingSystemCount++;
                    continue;
                }
                break;
            }

            console.log(`Found ${leadingSystemCount} leading system messages`);

            if (leadingSystemCount > 0) {
                const tagIndex = leadingSystemCount - 1;
                const sysMsg = request.body.messages[tagIndex];

                console.log(`Tagging system message at index ${tagIndex} for caching`);

                if (typeof sysMsg.content === 'string') {
                    const truncatedText = sysMsg.content.slice(0, 50) + (sysMsg.content.length > 50 ? '...' : '');
                    console.log(`Converting string content to array format - (${truncatedText})`);

                    sysMsg.content = [{
                        type: 'text',
                        text: sysMsg.content,
                        cache_control: { type: 'ephemeral', ttl: cacheTTL },
                    }];

                    console.log(`System cache breakpoint is at ${tagIndex} - (${truncatedText})`);
                } else if (Array.isArray(sysMsg.content) && sysMsg.content.length) {
                    console.log(`System message content is already array format with ${sysMsg.content.length} parts`);

                    // Prefer tagging the last text block if present, otherwise tag the last part
                    let partIndex = -1;
                    for (let j = sysMsg.content.length - 1; j >= 0; j--) {
                        if (sysMsg.content[j] && sysMsg.content[j].type === 'text') {
                            partIndex = j;
                            break;
                        }
                    }
                    const idx = partIndex !== -1 ? partIndex : sysMsg.content.length - 1;

                    console.log(`Tagging content part at index ${idx} (${partIndex !== -1 ? 'text block' : 'last part'})`);

                    sysMsg.content[idx].cache_control = { type: 'ephemeral', ttl: cacheTTL };

                    const truncatedText = sysMsg.content[idx].text ?
                        sysMsg.content[idx].text.slice(0, 50) + (sysMsg.content[idx].text.length > 50 ? '...' : '') :
                        '[non-text content]';

                    console.log(`System cache breakpoint is at ${tagIndex} - (${truncatedText})`);
                }
            } else {
                console.log('No leading system messages found for caching');
            }
        } else {
            if (!enableSystemPromptCache) {
                console.log('System prompt caching is disabled in config');
            } else if (!isClaude3or4) {
                console.log('System prompt caching not applied - not a Claude 3/4 model');
            } else {
                console.log('System prompt caching not applied - no messages found');
            }
        }

        const isGemini = /google\/gemini/.test(request.body.model);
        if (isGemini) {
            bodyParams['safety_settings'] = GEMINI_SAFETY;
        }
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CUSTOM) {
        apiUrl = request.body.custom_url;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.CUSTOM);
        headers = {};
        bodyParams = {
            logprobs: request.body.logprobs,
            top_logprobs: undefined,
        };

        // Adjust logprobs params for Chat Completions API, which expects { top_logprobs: number; logprobs: boolean; }
        if (!isTextCompletion && bodyParams.logprobs > 0) {
            bodyParams.top_logprobs = bodyParams.logprobs;
            bodyParams.logprobs = true;
        }

        mergeObjectWithYaml(bodyParams, request.body.custom_include_body);
        mergeObjectWithYaml(headers, request.body.custom_include_headers);
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.PERPLEXITY) {
        apiUrl = API_PERPLEXITY;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.PERPLEXITY);
        headers = {};
        bodyParams = {
            reasoning_effort: request.body.reasoning_effort,
        };
        request.body.messages = postProcessPrompt(request.body.messages, PROMPT_PROCESSING_TYPE.STRICT, getPromptNames(request));
        if (request.body.json_schema) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    schema: request.body.json_schema.value,
                },
            };
        }
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.GROQ) {
        apiUrl = API_GROQ;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.GROQ);
        headers = {};
        bodyParams = {};
        if (request.body.json_schema) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    description: request.body.json_schema.description,
                    schema: request.body.json_schema.value,
                    strict: request.body.json_schema.strict ?? true,
                },
            };
        }
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.FIREWORKS) {
        apiUrl = API_FIREWORKS;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.FIREWORKS);
        headers = {};
        bodyParams = {};
        if (request.body.json_schema) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    description: request.body.json_schema.description,
                    schema: request.body.json_schema.value,
                    strict: request.body.json_schema.strict ?? true,
                },
            };
        }
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.NANOGPT) {
        apiUrl = API_NANOGPT;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.NANOGPT);
        headers = {};
        bodyParams = {};
        if (request.body.enable_web_search && !/:online$/.test(request.body.model)) {
            request.body.model = `${request.body.model}:online`;
        }
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.POLLINATIONS) {
        apiUrl = API_POLLINATIONS;
        apiKey = 'NONE';
        headers = {
            'Authorization': '',
        };
        bodyParams = {
            reasoning_effort: request.body.reasoning_effort,
            private: true,
            referrer: 'sillytavern',
            seed: request.body.seed ?? Math.floor(Math.random() * 99999999),
        };
        if (request.body.json_schema) {
            setJsonObjectFormat(bodyParams, request.body.messages, request.body.json_schema);
        }
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MOONSHOT) {
        apiUrl = API_MOONSHOT;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.MOONSHOT);
        headers = {};
        bodyParams = {};
        request.body.json_schema
            ? setJsonObjectFormat(bodyParams, request.body.messages, request.body.json_schema)
            : addAssistantPrefix(request.body.messages, [], 'partial');
    } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.COMETAPI) {
        apiUrl = API_COMETAPI;
        apiKey = readSecret(request.user.directories, SECRET_KEYS.COMETAPI);
        headers = {};
        bodyParams = {
            reasoning_effort: request.body.reasoning_effort,
        };
        throw new Error('This provider is temporarily disabled.');
    } else {
        console.warn('This chat completion source is not supported yet.');
        return response.status(400).send({ error: true });
    }

    // A few of OpenAIs reasoning models support reasoning effort
    if (request.body.reasoning_effort && [CHAT_COMPLETION_SOURCES.CUSTOM, CHAT_COMPLETION_SOURCES.OPENAI].includes(request.body.chat_completion_source)) {
        const reasoningEffortModels = [
            'o1',
            'o3-mini',
            'o3-mini-2025-01-31',
            'o4-mini',
            'o4-mini-2025-04-16',
            'o3',
            'o3-2025-04-16',
            'gpt-5',
            'gpt-5-2025-08-07',
            'gpt-5-mini',
            'gpt-5-mini-2025-08-07',
            'gpt-5-nano',
            'gpt-5-nano-2025-08-07',
        ];
        const reasoningEffortMap = {
            min: 'minimal',
        };
        if (reasoningEffortModels.includes(request.body.model)) {
            bodyParams['reasoning_effort'] = reasoningEffortMap[request.body.reasoning_effort] ?? request.body.reasoning_effort;
        }
    }

    if (!apiKey && !request.body.reverse_proxy && request.body.chat_completion_source !== CHAT_COMPLETION_SOURCES.CUSTOM) {
        console.warn('OpenAI API key is missing.');
        return response.status(400).send({ error: true });
    }

    // Add custom stop sequences
    if (Array.isArray(request.body.stop) && request.body.stop.length > 0) {
        bodyParams['stop'] = request.body.stop;
    }

    const textPrompt = isTextCompletion ? convertTextCompletionPrompt(request.body.messages) : '';
    const endpointUrl = isTextCompletion && request.body.chat_completion_source !== CHAT_COMPLETION_SOURCES.OPENROUTER ?
        `${apiUrl}/completions` :
        `${apiUrl}/chat/completions`;

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    if (!isTextCompletion && Array.isArray(request.body.tools) && request.body.tools.length > 0) {
        bodyParams['tools'] = request.body.tools;
        bodyParams['tool_choice'] = request.body.tool_choice;
    }

    if (request.body.json_schema && !bodyParams['response_format']) {
        bodyParams['response_format'] = {
            type: 'json_schema',
            json_schema: {
                name: request.body.json_schema.name,
                strict: request.body.json_schema.strict ?? true,
                schema: request.body.json_schema.value,
            },
        };
    }

    const requestBody = {
        'messages': isTextCompletion === false ? request.body.messages : undefined,
        'prompt': isTextCompletion === true ? textPrompt : undefined,
        'model': request.body.model,
        'temperature': request.body.temperature,
        'max_tokens': request.body.max_tokens,
        'max_completion_tokens': request.body.max_completion_tokens,
        'stream': request.body.stream,
        'presence_penalty': request.body.presence_penalty,
        'frequency_penalty': request.body.frequency_penalty,
        'top_p': request.body.top_p,
        'top_k': request.body.top_k,
        'stop': isTextCompletion === false ? request.body.stop : undefined,
        'logit_bias': request.body.logit_bias,
        'seed': request.body.seed,
        'n': request.body.n,
        ...bodyParams,
    };

    if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CUSTOM) {
        excludeKeysByYaml(requestBody, request.body.custom_exclude_body);
    }

    /** @type {import('node-fetch').RequestInit} */
    const config = {
        method: 'post',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + apiKey,
            ...headers,
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
    };

    // Compact, formatted context preview for quick verification
    try {
        if (true) {
            const previewLen = 15;
            const hasCacheBreakpoint = (/** @type {any} */ msg) => {
                if (msg && typeof msg === 'object' && msg.cache_control) return true;
                const c = msg?.content;
                if (Array.isArray(c)) {
                    return c.some(p => p && typeof p === 'object' && p.cache_control);
                }
                return false;
            };
            const quote = (s) => `"${String(s).replace(/\s+/g, ' ').trim().slice(0, previewLen)}"`;

            let logLines = [];
            if (isTextCompletion) {
                logLines.push('Sent prompt: ' + quote(requestBody.prompt ?? ''));
            } else if (Array.isArray(requestBody.messages)) {
                const msgs = requestBody.messages;
                const keepTotal = 24; // show first 6 and last 6 if large
                const headCount = 12;
                const tailCount = 12;
                const useFold = msgs.length > keepTotal;
                const toShow = useFold ? [...msgs.slice(0, headCount), '…', ...msgs.slice(-tailCount)] : msgs;
                logLines.push('Sent context:');
                for (const m of toShow) {
                    if (m === '…') { logLines.push('...'); continue; }
                    const role = m.role ?? '?';
                    let text = '';
                    if (typeof m.content === 'string') {
                        text = m.content;
                    } else if (Array.isArray(m.content)) {
                        const textPart = m.content.find(p => p && (p.type === 'text' || p.type === 'input_text'));
                        text = textPart?.text ?? '';
                    } else if (typeof m.content === 'object' && m.content !== null) {
                        if (m.content.type === 'text' && m.content.text) text = m.content.text;
                    }
                    const marker = hasCacheBreakpoint(m) ? ' (📦 cache breakpoint)' : '';
                    logLines.push(`(${role})\t${quote(text)}${marker}`);
                }
            }
            if (logLines.length) {
                console.debug(logLines.join('\n'));
            }
        }
    } catch (e) {
        console.warn('Failed to format context preview:', e);
    }

    makeRequest(config, response, request);

    /**
     * Makes a fetch request to the OpenAI API endpoint.
     * @param {import('node-fetch').RequestInit} config Fetch config
     * @param {express.Response} response Express response
     * @param {express.Request} request Express request
     * @param {Number} retries Number of retries left
     * @param {Number} timeout Request timeout in ms
     */
    async function makeRequest(config, response, request, retries = 5, timeout = 5000) {
        try {
            controller.signal.throwIfAborted();
            const fetchResponse = await fetch(endpointUrl, config);

            if (request.body.stream) {
                console.info('Streaming request in progress');
                forwardFetchResponse(fetchResponse, response);
                return;
            }

            if (fetchResponse.ok) {
                /** @type {any} */
                let json = await fetchResponse.json();
                response.send(json);
                console.debug(json);
                console.debug(json?.choices?.[0]?.message);
            } else if (fetchResponse.status === 429 && retries > 0) {
                console.warn(`Out of quota, retrying in ${Math.round(timeout / 1000)}s`);
                setTimeout(() => {
                    timeout *= 2;
                    makeRequest(config, response, request, retries - 1, timeout);
                }, timeout);
            } else {
                await handleErrorResponse(fetchResponse);
            }
        } catch (error) {
            console.error('Generation failed', error);
            const message = error.code === 'ECONNREFUSED'
                ? `Connection refused: ${error.message}`
                : error.message || 'Unknown error occurred';

            if (!response.headersSent) {
                response.status(502).send({ error: { message, ...error } });
            } else {
                response.end();
            }
        }
    }

    /**
     * @param {import("node-fetch").Response} errorResponse
     */
    async function handleErrorResponse(errorResponse) {
        const responseText = await errorResponse.text();
        const errorData = tryParse(responseText);

        const message = errorResponse.statusText || 'Unknown error occurred';
        const quota_error = errorResponse.status === 429 && errorData?.error?.type === 'insufficient_quota';
        console.error('Chat completion request error: ', message, responseText);

        if (!response.headersSent) {
            response.send({ error: { message }, quota_error: quota_error });
        } else if (!response.writableEnded) {
            response.write(errorResponse);
        } else {
            response.end();
        }
    }
});

const pollinations = express.Router();

pollinations.post('/models/multimodal', async (_req, res) => {
    try {
        const response = await fetch('https://text.pollinations.ai/models');

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();

        if (!Array.isArray(data)) {
            return res.json([]);
        }

        const multimodalModels = data.filter(m => m?.vision).map(m => m.name);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

router.use('/pollinations', pollinations);

const aimlapi = express.Router();

aimlapi.post('/models/multimodal', async (_req, res) => {
    try {
        const response = await fetch('https://api.aimlapi.com/v1/models');

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();

        if (!Array.isArray(data?.data)) {
            return res.json([]);
        }

        const multimodalModels = data.data.filter(m => m?.features?.includes('openai/chat-completion.vision')).map(m => m.id);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

router.use('/aimlapi', aimlapi);
