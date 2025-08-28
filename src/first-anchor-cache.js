import { getConfigValue } from './util.js';

// --- First-Anchor Context Reconstruction (global matching) ---
// Transparent, optional pre-processing to preserve cache hits under limited context.
// Controlled by config: claude.firstAnchorCaching.enabled / ttlSeconds / allowRewind
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

/**
 * Apply First-Anchor Context Reconstruction to preserve cache hits under limited context.
 * Modifies request.body.messages in-place when reconstruction is applied.
 * 
 * @param {Object} request - Express request object with body.messages
 */
export function applyFirstAnchorReconstruction(request) {
    try {
        const enabled = getConfigValue('claude.firstAnchorCaching.enabled', false, 'boolean');
        if (!enabled) return;
        if (!Array.isArray(request.body?.messages) || request.body.messages.length === 0) return;

        console.log("\n\n");

        // Split leading system block
        const sys = [];
        let i = 0;
        while (i < request.body.messages.length && request.body.messages[i]?.role === 'system') {
            sys.push(request.body.messages[i]);
            i++;
        }

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

            // Rebuild full context: system + lynchpin + trailing system
            request.body.messages = [...sys, ...best.msgs, ...trailingSys];
            const sysPrev = sys.length ? `${msgPreview(sys.at(-1))}` : '(no system)';
            console.log('[FirstAnchor] Reconstructed context applied. System last:', sysPrev, '| Lynchpin len:', best.msgs.length);
            return; // all or nothing
        }

        // No right-ended overlap: try rewind if enabled and we can find a contiguous subsequence match
        const allowRewind = getConfigValue('claude.firstAnchorCaching.allowRewind', true, 'boolean');
        if (allowRewind) {
            // Try rewind by contiguous subsequence match across all lynchpins
            for (const entry of FIRST_ANCHOR_STORE) {
                const s = indexOfSubsequence(entry.fps, windowFP);
                if (s !== -1) {
                    const e = s + windowFP.length - 1;
                    const prevTail = entry.msgs.length ? `${msgPreview(entry.msgs.at(-1))}` : '(empty)';
                    const newTail = `${msgPreview(entry.msgs[e])}`;
                    const cutCount = Math.max(0, entry.msgs.length - (e + 1));
                    const prevIndex = entry.msgs.length - 1;
                    const rewindToIdx = e;
                    const jumpBack = Math.max(0, prevIndex - rewindToIdx);
                    console.log('[FirstAnchor] Rewind detected. Jumping back', jumpBack, 'steps to:', newTail);
                    console.log(`[FirstAnchor] PrevTail (index ${prevIndex}):`, prevTail);
                    console.log(`[FirstAnchor] PrevTail (rewind to ${prevIndex - jumpBack}):`, newTail);
                    console.log('[FirstAnchor] CutCount=', cutCount, '| LynchpinLen=', e + 1);

                    entry.msgs = entry.msgs.slice(0, e + 1);
                    entry.fps = entry.fps.slice(0, e + 1);
                    entry.expireAt = Date.now() + ttlMs;

                    request.body.messages = [...sys, ...entry.msgs, ...trailingSys];
                    console.log('[FirstAnchor] Reconstructed context applied after rewind. Tail is now at:', newTail);
                    return;
                }
            }

            // Rewind + edit: allow prefix match of the window against the lynchpin when only the tail differs
            const windowLen = windowFP.length;
            if (windowLen >= 2) {
                for (const entry of FIRST_ANCHOR_STORE) {
                    const pref = findLongestPrefixMatch(entry.fps, windowFP);
                    const appendedCount = windowLen - pref.length;
                    if (pref.startIndex !== -1 && pref.length >= windowLen - 1 && appendedCount > 0) {
                        const cutTo = pref.startIndex + pref.length - 1;
                        const prevTail = entry.msgs.length ? `${msgPreview(entry.msgs.at(-1))}` : '(empty)';
                        const newTailMsgObj = windowMsgs[windowLen - 1];
                        const newTail = `${msgPreview(newTailMsgObj)}`;
                        console.log('[FirstAnchor] Rewind + edit detected. Prefix match=', pref.length);
                        console.log(`[FirstAnchor] PrevTail (index ${entry.msgs.length - 1}):`, prevTail);
                        console.log(`[FirstAnchor] Cut to index ${cutTo}, append edited messages:`, appendedCount);

                        entry.msgs = entry.msgs.slice(0, cutTo + 1).concat(windowMsgs.slice(pref.length));
                        entry.fps = entry.fps.slice(0, cutTo + 1).concat(windowFP.slice(pref.length));
                        entry.expireAt = Date.now() + ttlMs;

                        request.body.messages = [...sys, ...entry.msgs, ...trailingSys];
                        console.log('[FirstAnchor] Reconstructed context applied after rewind+edit. NewTail:', newTail, '| LynchpinLen=', entry.msgs.length);
                        return;
                    }
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

// Export helper functions for testing
export {
    fingerprintMessage,
    longestRightEndedOverlap,
    indexOfSubsequence,
    findLongestPrefixMatch,
    msgPreview,
    FIRST_ANCHOR_STORE
};
