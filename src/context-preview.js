// Shared utilities for building a compact, formatted preview of a request payload's messages/prompt
// ESM module

/**
 * Extract TTL from a Claude-style message or any nested content parts that may carry cache_control.
 * @param {any} msg
 * @returns {string|null}
 */
export const getCacheTTL = (msg) => {
    try {
        if (msg && typeof msg === 'object' && msg.cache_control && typeof msg.cache_control === 'object') {
            const t = msg.cache_control.ttl;
            if (t) return String(t);
        }
        const c = msg?.content;
        if (Array.isArray(c)) {
            for (const p of c) {
                if (p && typeof p === 'object' && p.cache_control && typeof p.cache_control === 'object') {
                    const t = p.cache_control.ttl;
                    if (t) return String(t);
                }
            }
        } else if (c && typeof c === 'object') {
            if (c.cache_control && typeof c.cache_control === 'object' && c.cache_control.ttl) {
                return String(c.cache_control.ttl);
            }
        }
    } catch {}
    return null;
};

/**
 * Simple hash for unique content identification (non-cryptographic).
 * @param {any} content
 * @returns {string}
 */
export const hashContent = (content) => {
    const str = typeof content === 'string' ? content : JSON.stringify(content);
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // 32-bit int
    }
    return Math.abs(hash).toString(16).padStart(8, '0').slice(0, 8);
};

// Compute a stable identity hash for a full message object by recursively
// sorting object keys and hashing the exact JSON bytes. We deliberately do
// not normalize whitespace or coerce types to avoid masking cache risks.
const sortKeysDeep = (val) => {
    if (Array.isArray(val)) return val.map(sortKeysDeep);
    if (val && typeof val === 'object') {
        const out = {};
        const keys = Object.keys(val).sort();
        for (const k of keys) {
            out[k] = sortKeysDeep(val[k]);
        }
        return out;
    }
    return val;
};

export const identityHashForMessage = (_msg) => {

    const msg = structuredClone(_msg);

    if (Array.isArray(msg?.content)) {
        for (const p of msg.content) {
            if (p && typeof p === 'object') {
                delete p.cache_control;
            }
        }
    } else if (msg?.content && typeof msg.content === 'object') {
        delete msg.content.cache_control;
    }

    return hashContent(msg);
};

// Global neighbor sanity tracker: remembers first-seen neighbors for each
// identity hash across the process lifetime.
const neighborSanityMap = new Map(); // hash -> { left: string|null, right: string|null }

const fieldIdentityMap = new Map();

// Allow external callers (e.g., runner manual mode) to reset neighbor tracking
export function resetNeighborSanityTracker() {
    neighborSanityMap.clear();
}

// Snapshot/restore API for time-travel in manual runner
export function snapshotNeighborSanityTracker() {
    const snap = {};
    for (const [k, v] of neighborSanityMap.entries()) {
        snap[k] = { left: v?.left ?? null, right: v?.right ?? null };
    }
    return snap;
}

export function restoreNeighborSanityTracker(snapshot) {
    neighborSanityMap.clear();
    if (!snapshot || typeof snapshot !== 'object') return;
    for (const k of Object.keys(snapshot)) {
        const v = snapshot[k];
        neighborSanityMap.set(k, { left: v?.left ?? null, right: v?.right ?? null });
    }
}



/**
 * Generate a stable hash for any JSON-serializable value.
 * @param {any} value
 * @returns {string|null}
 */
const hashForFieldValue = (value) => {
    if (value === undefined) {
        return null;
    }
    if (value === null) {
        return hashContent('null');
    }
    const type = typeof value;
    if (type === 'string') {
        return hashContent(value);
    }
    if (type === 'number' || type === 'boolean' || type === 'bigint') {
        return hashContent(String(value));
    }
    if (type === 'object') {
        return hashContent(sortKeysDeep(value));
    }
    return hashContent(String(value));
};

const fieldChangeMarker = (key, hash) => {
    if (!hash) {
        return '';
    }
    const prev = fieldIdentityMap.get(key);
    if (!prev) {
        fieldIdentityMap.set(key, hash);
        return ' (🔷)';
    }
    if (prev === hash) {
        return ' (🟢)';
    }
    fieldIdentityMap.set(key, hash);
    return ' (🔴)';
};

const previewPrimitive = (value, previewLen) => {
    if (value === null) {
        return 'null';
    }
    if (value === undefined) {
        return 'undefined';
    }
    if (typeof value === 'string') {
        return `"${createPreview(value, previewLen)}"`;
    }
    return String(value);
};

const previewArray = (value, previewLen) => {
    if (!value.length) {
        return '[]';
    }
    const isPrimitiveList = value.length <= 3 && value.every((item) => item === null || ['string', 'number', 'boolean', 'bigint'].includes(typeof item));
    if (isPrimitiveList) {
        const parts = value.map((item) => previewPrimitive(item, previewLen));
        return `[${parts.join(', ')}]`;
    }
    const serialized = createPreview(JSON.stringify(sortKeysDeep(value)), previewLen);
    return `Array(len=${value.length}) ${serialized}`;
};

const previewObject = (value, previewLen) => {
    if (!value || typeof value !== 'object') {
        return previewPrimitive(value, previewLen);
    }
    const keys = Object.keys(value).sort();
    if (!keys.length) {
        return '{}';
    }
    const keyHints = keys.slice(0, 3).join(', ');
    const hintSuffix = keys.length > 3 ? ', …' : '';
    const highlightKeys = ['name', 'id', 'type', 'role'];
    const highlights = highlightKeys
        .filter((k) => Object.prototype.hasOwnProperty.call(value, k) && (typeof value[k] === 'string' || typeof value[k] === 'number'))
        .map((k) => `${k}=${previewPrimitive(value[k], previewLen)}`);
    const serialized = createPreview(JSON.stringify(sortKeysDeep(value)), previewLen);
    const hint = highlights.length ? highlights.join(' ') : `keys=${keyHints}${hintSuffix}`;
    return `Object{${hint}} ${serialized}`;
};

const previewValue = (value, previewLen) => {
    if (Array.isArray(value)) {
        return previewArray(value, previewLen);
    }
    if (value && typeof value === 'object') {
        return previewObject(value, previewLen);
    }
    return previewPrimitive(value, previewLen);
};

const buildFieldLines = (requestBody, previewLen) => {
    if (!requestBody || typeof requestBody !== 'object') {
        return [];
    }
    const lines = [];
    const seenKeys = new Set();
    const topLevelKeys = Object.keys(requestBody)
        .filter((key) => key !== 'messages' && requestBody[key] !== undefined)
        .sort();

    for (const key of topLevelKeys) {
        const value = requestBody[key];
        const hash = hashForFieldValue(value);
        const marker = fieldChangeMarker(key, hash);
        const summary = previewValue(value, previewLen);
        const hashSuffix = hash ? ` (${hash})` : '';
        lines.push(`${key}: ${summary}${hashSuffix}${marker}`);
        seenKeys.add(key);

        if (Array.isArray(value) && value.length && value.length <= 5) {
            value.forEach((item, index) => {
                if (item === undefined) {
                    return;
                }
                const itemKey = `${key}[${index}]`;
                const itemHash = hashForFieldValue(item);
                const itemMarker = fieldChangeMarker(itemKey, itemHash);
                const itemSummary = previewValue(item, previewLen);
                const itemHashSuffix = itemHash ? ` (${itemHash})` : '';
                lines.push(`  [${index}] ${itemSummary}${itemHashSuffix}${itemMarker}`);
                seenKeys.add(itemKey);
            });
        }
    }

    for (const existingKey of Array.from(fieldIdentityMap.keys())) {
        if (!seenKeys.has(existingKey)) {
            fieldIdentityMap.delete(existingKey);
            lines.push(`Removed field: ${existingKey}`);
        }
    }

    return lines;
};



/**
 * Create a trimmed preview with head/tail around an ellipsis.
 * @param {string} text
 * @param {number} previewLen
 * @returns {string}
 */
export const createPreview = (text, previewLen = 20) => {
    const cleaned = String(text).replace(/\s+/g, ' ').trim();
    if (cleaned.length <= previewLen * 2 + 6) {
        return cleaned;
    }
    const start = cleaned.slice(0, previewLen);
    const end = cleaned.slice(-previewLen);
    return `${start}... ...${end}`;
};

/**
 * Build preview lines for either a text completion (prompt) or a chat style messages array.
 * Returns an array of strings to print or join with \"\n\".
 * @param {{ requestBody: any, isTextCompletion: boolean, previewLen?: number, metaForIndex?: (index: number, msg: any) => string, trackNeighbors?: boolean }} params
 * @returns {string[]}
 */
export function buildContextPreviewLines({ requestBody, isTextCompletion, previewLen = 20, metaForIndex = undefined, trackNeighbors = false }) {
    const lines = [];

    if (isTextCompletion) {
        const prompt = requestBody?.prompt ?? '';
        const preview = createPreview(prompt, previewLen);
        const hash = hashContent(prompt);
        lines.push(`Sent prompt: ${preview} (${hash})`);
        return lines;
    }

    const msgs = Array.isArray(requestBody?.messages) ? requestBody.messages : null;
    if (!msgs) {
        const fieldLines = buildFieldLines(requestBody, previewLen);
        lines.push(...fieldLines);
        return lines;
    }

    lines.push(`Sent context (${msgs.length} total messages):`);

    // Compute identity hashes for all messages first
    const hashes = msgs.map(identityHashForMessage);

    for (let i = 0; i < msgs.length; i++) {
        const m = msgs[i];
        const role = m?.role === 'user' ? '🙍' : m?.role === 'assistant' ? '🤖' : m?.role === 'system' ? '⚙️' : m?.role ? `❓${m.role}` : '?';

        let text = '';
        if (typeof m?.content === 'string') {
            text = m.content;
        } else if (Array.isArray(m?.content)) {
            const textPart = m.content.find((p) => p && (p.type === 'text' || p.type === 'input_text'));
            text = textPart?.text ?? '';
        } else if (typeof m?.content === 'object' && m?.content !== null) {
            if (m.content.type === 'text' && m.content.text) text = m.content.text;
        }

        const preview = createPreview(text, previewLen);
        const hash = hashes[i];
        const ttl = getCacheTTL(m);
        const marker = ttl ? ` (📦 ttl=${ttl})` : '';

        // Neighbor sanity: detect order drift
        let neighborMarker = '';
        if (trackNeighbors) {
            const left = i > 0 ? hashes[i - 1] : null;
            const right = i < hashes.length - 1 ? hashes[i + 1] : null;
            const prev = neighborSanityMap.get(hash);
            if (!prev) {
                neighborSanityMap.set(hash, { left, right });
                neighborMarker = ' (🔷)';
            } else if (prev.left !== left || prev.right !== right) {
                neighborMarker = ' (🔴)';
                // Update baseline so only unexpected future drifts continue to flag
                neighborSanityMap.set(hash, { left, right });
            } else {
                neighborMarker = ' (🟢)';
            }
        }

        const extra = typeof metaForIndex === 'function' ? (metaForIndex(i, m) || '') : '';
        lines.push(`[${i + 1}] ${role}: ${preview} (${hash})${marker}${neighborMarker}${extra}`);
    }

    const fieldLines = buildFieldLines(requestBody, previewLen);
    if (fieldLines.length) {
        lines.push('Other request params:');
        lines.push(...fieldLines);
    }

    return lines;
}

/**
 * Convenience wrapper to print the preview directly.
 * @param {{ requestBody: any, isTextCompletion: boolean, previewLen?: number }} params
 */
export function logContextPreview(params) {
    const lines = buildContextPreviewLines(params);
    if (lines.length) {
        // eslint-disable-next-line no-console
        console.debug(lines.join('\n'));
    }
}
