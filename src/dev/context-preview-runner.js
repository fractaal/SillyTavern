// Visual sanity check: animate the megaprompt/context as it grows, showing cache anchors
// Run: node src/dev/context-preview-runner.js --messages 100 --delay 500 --previewLen 20 --ttl 5m --source openrouter --cad -1 [--no-megaprompt]

import { buildContextPreviewLines, resetNeighborSanityTracker, snapshotNeighborSanityTracker, restoreNeighborSanityTracker } from '../context-preview.js';

function countBlocks(msg) {
    if (!msg) return 0;
    if (Array.isArray(msg.content)) return msg.content.length;
    if (typeof msg.content === 'string') return msg.content.length > 0 ? 1 : 0;
    return 0;
}

function ensureArrayContent(msg) {
    if (!msg) return [];
    if (Array.isArray(msg.content)) return msg.content;
    const text = typeof msg.content === 'string' ? msg.content : String(msg.content ?? '');
    msg.content = [{ type: 'text', text }];
    return msg.content;
}

function setCacheOnUserMessage(msg, ttl) {
    if (!msg || msg.role !== 'user') return false;
    const content = ensureArrayContent(msg);
    if (content.length === 0) return false;
    let idx = -1;
    for (let i = content.length - 1; i >= 0; i--) {
        if (content[i] && content[i].type === 'text') { idx = i; break; }
    }
    if (idx === -1) idx = content.length - 1;
    if (!content[idx]) return false;
    content[idx].cache_control = { type: 'ephemeral', ttl };
    return true;
}

function setCacheOnAnyMessage(msg, ttl) {
    const content = ensureArrayContent(msg);
    if (content.length === 0) return false;
    let idx = -1;
    for (let i = content.length - 1; i >= 0; i--) {
        if (content[i] && content[i].type === 'text') { idx = i; break; }
    }
    if (idx === -1) idx = content.length - 1;
    if (!content[idx]) return false;
    content[idx].cache_control = { type: 'ephemeral', ttl };
    return true;
}

function applyAnchors(messages, { ttl, cachingAtDepth = -1, spacing = 20, maxAnchors = 3 }) {
    if (!Array.isArray(messages) || messages.length === 0) return;

    let anchorsPlaced = 0;
    let cumulativeBlocks = 0;
    let nextThreshold = spacing;
    let lastUserInWindowIndex = null;
    let lastAnchorIndex = -1;

    for (let i = 0; i < messages.length && anchorsPlaced < maxAnchors; i++) {
        const msg = messages[i];
        cumulativeBlocks += countBlocks(msg);
        if (msg?.role === 'user') lastUserInWindowIndex = i;

        while (cumulativeBlocks >= nextThreshold && anchorsPlaced < maxAnchors) {
            const idx = (lastUserInWindowIndex != null) ? lastUserInWindowIndex : i;
            const target = messages[idx];
            if (target?.role === 'user') {
                if (setCacheOnUserMessage(target, ttl)) {
                    anchorsPlaced++;
                    lastAnchorIndex = idx;
                }
            } else {
                if (setCacheOnAnyMessage(target, ttl)) {
                    anchorsPlaced++;
                    lastAnchorIndex = idx;
                }
            }
            nextThreshold += spacing;
            lastUserInWindowIndex = null;
        }
    }

    if (anchorsPlaced < maxAnchors && Number.isInteger(cachingAtDepth) && cachingAtDepth >= 0) {
        let passedPrefill = false; let seen = 0; let targetUser = null; let targetIdx = -1;
        for (let i = messages.length - 1; i >= 0; i--) {
            const m = messages[i];
            if (!passedPrefill && m.role === 'assistant') continue;
            passedPrefill = true;
            if (m.role === 'user') {
                if (seen === cachingAtDepth) { targetUser = m; targetIdx = i; break; }
                seen++;
            }
        }
        if (targetUser) {
            let tailUsers = 0;
            for (let i = lastAnchorIndex + 1; i < messages.length; i++) {
                if (messages[i]?.role === 'user') tailUsers++;
            }
            if (tailUsers >= cachingAtDepth + 1) {
                if (setCacheOnUserMessage(targetUser, ttl)) {
                    anchorsPlaced++;
                    lastAnchorIndex = targetIdx;
                }
            }
        }
    }
}

/** Parse CLI args without external deps */
function parseArgs(argv) {
    const out = {};
    for (let i = 2; i < argv.length; i++) {
        const arg = argv[i];
        const next = argv[i + 1];
        const isFlag = (name, short) => arg === `--${name}` || arg === `-${short}`;
        if (isFlag('messages', 'm')) { out.messages = Number(next); i++; continue; }
        if (isFlag('delay', 'd')) { out.delay = Number(next); i++; continue; }
        if (isFlag('previewLen', 'p')) { out.previewLen = Number(next); i++; continue; }
        if (isFlag('ttl', 't')) { out.ttl = next; i++; continue; }
        if (isFlag('source', 's')) { out.source = String(next); i++; continue; }
        if (isFlag('cad', 'c')) { out.cachingAtDepth = Number(next); i++; continue; }
        if (arg === '--pairs') { out.pairs = true; continue; }
        if (arg === '--user-only') { out.pairs = false; continue; }
        if (arg === '--no-megaprompt') { out.megaprompt = false; continue; }
        if (isFlag('manual', 'M')) { out.manual = true; continue; }
    }
    return out;
}

const sleep = (ms) => new Promise(res => setTimeout(res, ms));

/** Create a synthetic messages array; if pairs=true, alternate user/assistant */
function generateMessages(total, pairs = true) {
    const arr = [];
    for (let i = 1; i <= total; i++) {
        arr.push({
            role: 'user',
            content: [{
                type: 'text',
                text: `User message #${i} — lorem ipsum dolor sit amet.`
            }]
        });
        if (pairs) {
            arr.push({
                role: 'assistant',
                content: [{
                    type: 'text',
                    text: `Assistant reply #${i} — consectetuer adipiscing elit.`
                }]
            });
        }
    }
    return arr;
}

async function main() {
    const args = parseArgs(process.argv);
    const maxMessages = Number.isFinite(args.messages) ? Math.max(0, args.messages) : 100;
    const delay = Number.isFinite(args.delay) ? Math.max(0, args.delay) : 500;
    const previewLen = Number.isFinite(args.previewLen) ? Math.max(5, args.previewLen) : 20;
    const ttl = args.ttl || '5m';
    const source = (args.source || 'openrouter').toLowerCase(); // 'openrouter' | 'claude'
    const cachingAtDepth = Number.isFinite(args.cachingAtDepth) ? args.cachingAtDepth : -1; // -1 disables tail-depth anchor
    const pairs = args.pairs !== false; // default true unless --user-only
    const megaprompt = args.megaprompt !== false; // default true, disable with --no-megaprompt
    const manual = !!args.manual; // manual playback mode

    const base = generateMessages(maxMessages, pairs);

    // Prepare prompt-converters (dynamic) so we can run real megaprompt sealing without reading config files
    let converters = null;
    if (megaprompt) {
        try {
            // Provide env fallbacks so prompt-converters doesn't try to read config.yaml
            process.env.SILLYTAVERN_PROMPTPLACEHOLDER = process.env.SILLYTAVERN_PROMPTPLACEHOLDER || "Let's get started.";
            process.env.SILLYTAVERN_CLAUDE_MEGAPROMPT_ENABLED = process.env.SILLYTAVERN_CLAUDE_MEGAPROMPT_ENABLED || 'true';
            process.env.SILLYTAVERN_CLAUDE_MAXANCHORS = process.env.SILLYTAVERN_CLAUDE_MAXANCHORS || '3';
            process.env.SILLYTAVERN_CLAUDE_ANCHORSPACINGBLOCKS = process.env.SILLYTAVERN_CLAUDE_ANCHORSPACINGBLOCKS || '20';
            process.env.SILLYTAVERN_CLAUDE_MEGAPROMPT_PREANCHORSEALED = process.env.SILLYTAVERN_CLAUDE_MEGAPROMPT_PREANCHORSEALED || 'false';
            // Megaprompt compaction knobs
            process.env.SILLYTAVERN_CLAUDE_MEGAPROMPT_TURNMULTIPLE = process.env.SILLYTAVERN_CLAUDE_MEGAPROMPT_TURNMULTIPLE || '60';
            process.env.SILLYTAVERN_CLAUDE_MEGAPROMPT_MINLIVETAILTURNS = process.env.SILLYTAVERN_CLAUDE_MEGAPROMPT_MINLIVETAILTURNS || '10';
            process.env.SILLYTAVERN_CLAUDE_EXTENDEDTTL = process.env.SILLYTAVERN_CLAUDE_EXTENDEDTTL || 'false';

            converters = await import('../prompt-converters.js');
        } catch (e) {
            // eslint-disable-next-line no-console
            console.warn('prompt-converters not available; falling back to local anchoring only. Reason:', e?.message || e);
        }
    }

    // Snapshots of neighbor sanity tracker per step for manual time-travel
    let preSnaps = [];
    let postSnaps = [];

    const renderStep = (i) => {
        // Take the first i messages for display
        let current = JSON.parse(JSON.stringify(base.slice(0, i)));

        // Apply megaprompt sealing (if enabled and converters loaded)
        if (megaprompt && converters?.applyMegapromptCompaction) {
            try {
                current = converters.applyMegapromptCompaction(current, cachingAtDepth, { enabled: true, ttl });
            } catch (e) {
                // eslint-disable-next-line no-console
                console.warn('applyMegapromptCompaction failed; continuing without sealing:', e?.message || e);
            }
        }

        // Apply anchoring policy to visualize breakpoints
        try {
            if (converters?.cachingAtDepthForOpenRouterClaude) {
                converters.cachingAtDepthForOpenRouterClaude(current, cachingAtDepth, ttl);
            } else {
                applyAnchors(current, { ttl, cachingAtDepth, spacing: 20, maxAnchors: 3 });
            }
        } catch (e) {
            // eslint-disable-next-line no-console
            console.warn('Failed to apply anchoring (continuing without anchors):', e?.message || e);
        }

        // Determine anchor order for display (A1, A2, ...) and block counts
        const anchorOrdinalByIndex = new Map();
        let aNum = 1;
        for (let idx = 0; idx < current.length; idx++) {
            const parts = Array.isArray(current[idx]?.content) ? current[idx].content : [];
            const hasAnchor = parts.some((p) => p && p.cache_control && p.cache_control.ttl);
            if (hasAnchor) anchorOrdinalByIndex.set(idx, aNum++);
        }

        const requestBody = { messages: current };
        const logLines = buildContextPreviewLines({
            requestBody,
            isTextCompletion: false,
            previewLen,
            trackNeighbors: true,
            metaForIndex: (index, msg) => {
                const blocks = countBlocks(msg);
                const a = anchorOrdinalByIndex.get(index);
                const aMark = a ? ` [a${a}]` : '';
                return ` [b=${blocks}]${aMark}`;
            }
        });

        // Visual: clear and print this step
        // eslint-disable-next-line no-console
        console.clear();
        // eslint-disable-next-line no-console
        console.log(`Megaprompt preview — step ${i}/${maxMessages} (source=${source}, cad=${cachingAtDepth}, ttl=${ttl})`);
        if (manual) {
            // eslint-disable-next-line no-console
            console.log('Manual mode: [W] next  [B] back  [R] reset  [Q] quit');
        }
        // eslint-disable-next-line no-console
        console.log(logLines.join('\n'));
    };

    const renderManaged = (i, record = true) => {
        if (!preSnaps[i]) {
            if (i === 0) {
                resetNeighborSanityTracker();
                preSnaps[0] = snapshotNeighborSanityTracker();
            } else if (postSnaps[i - 1]) {
                preSnaps[i] = postSnaps[i - 1];
            } else {
                resetNeighborSanityTracker();
                preSnaps[i] = snapshotNeighborSanityTracker();
            }
        }
        restoreNeighborSanityTracker(preSnaps[i]);
        renderStep(i);
        if (record && !postSnaps[i]) {
            postSnaps[i] = snapshotNeighborSanityTracker();
            preSnaps[i + 1] = postSnaps[i];
        }
    };

    if (manual && process.stdin.isTTY) {
        let step = 0;
        preSnaps = [];
        postSnaps = [];
        resetNeighborSanityTracker();
        preSnaps[0] = snapshotNeighborSanityTracker();
        renderManaged(step);
        process.stdin.setRawMode(true);
        process.stdin.resume();
        process.stdin.setEncoding('utf8');
        process.stdin.on('data', (key) => {
            if (key == null) return;
            const s = String(key);
            if (s === '\u0003') { // Ctrl-C
                process.exit(0);
            }
            const ch = s.toLowerCase();
            if (ch === 'w') {
                if (step < maxMessages) step++;
                renderManaged(step);
            } else if (ch === 'b') {
                if (step > 0) step--;
                renderManaged(step, false);
            } else if (ch === 'r') {
                step = 0;
                preSnaps = [];
                postSnaps = [];
                resetNeighborSanityTracker();
                preSnaps[0] = snapshotNeighborSanityTracker();
                renderManaged(step);
            } else if (ch === 'q') {
                process.exit(0);
            }
        });
        await new Promise(() => {}); // keep process running
    } else {
        for (let i = 0; i <= maxMessages; i++) {
            renderStep(i);
            await sleep(delay);
        }
    }
}

main().catch(err => {
    // eslint-disable-next-line no-console
    console.error('Runner failed:', err);
    process.exitCode = 1;
});

