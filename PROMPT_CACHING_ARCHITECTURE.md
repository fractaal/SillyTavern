# SillyTavern Prompt Caching Architecture (Claude / OpenRouter)

This document explains the complete design and implementation of the prompt caching stack in SillyTavern, focusing on the three interacting systems:

1) First Anchor Context Reconstruction (1h TTL refreshed on use)
2) Megaprompt sealing/compaction (default threshold 60 turns) with a stable cache_control breakpoint
3) Custom caching-at-depth for Claude, including the OpenRouter path

It also details the order of operations, edge cases (rewinds, trailing systems), Anthropic’s breakpoint limits, and OpenRouter-specific behavior.


## High-level goals

- Preserve cache stability as the chat grows or shifts (edits/rewinds)
- Keep Anthropic cache_control breakpoints under the 4-breakpoint limit
- Maintain a sealed “megaprompt” chunk that rarely changes (stable cached bytes)
- Treat OpenRouter as a first-class provider for Claude models (anchoring + system caching)


## Source of truth (key files)

- src/first-anchor-cache.js
  - applyFirstAnchorReconstruction() and helpers
- src/prompt-converters.js
  - applyMegapromptCompaction()
  - stripClaudeCacheBreakpoints()
  - cachingAtDepthForClaude()
  - cachingAtDepthForOpenRouterClaude()
  - convertClaudeMessages() (re-attaches sealed cache_control after MERGE)
- src/endpoints/backends/chat-completions.js
  - End-to-end request routing, invokes reconstruction, compaction, anchoring
- src/tests/*.test.js
  - first-anchor-cache.test.js, megaprompt-compaction.test.js


## 1) First Anchor Context Reconstruction

Purpose: Transparent, optional pre-processing that reconstructs a “lynchpin” context window across requests so Anthropic caches continue to hit even when the live request window is short or the chat is edited.

Entry point:
- chat-completions route: applyFirstAnchorReconstruction(request) is called early in router.post('/generate', …)

Key configuration:
- claude.firstAnchorCaching.enabled (boolean)
- claude.firstAnchorCaching.ttlSeconds (default 3600 = 1 hour)
- claude.firstAnchorCaching.allowRewind (default true)

Data structures and normalization:
- In-memory FIRST_ANCHOR_STORE: [{ msgs, fps, expireAt }]
- Messages are fingerprinted with ephemeral fields removed (cache_control/id) and content normalized to arrays of parts => JSON string fingerprints

Window selection and system handling:
- Leading system messages are peeled off and preserved in sys[]
- Trailing “floating” system messages are trimmed out of the active window but saved as trailingSys[]
- Window for reconstruction = remainder non-system messages (windowMsgs)

Matching and extension:
- Find right-ended overlap length k = longestRightEndedOverlap(existing.fps, windowFP)
- If k > 0: extend lynchpin with the new right tail (windowMsgs.slice(k)) and refresh TTL
- Rebuild request: messages = [sys, …lynchpin.msgs, …trailingSys] (trailing systems are re-attached)

Rewind and rewind+edit handling:
- If no right-ended overlap:
  - allowRewind: find contiguous subsequence s = indexOfSubsequence(entry.fps, windowFP)
    - If found, cut lynchpin to end e = s + windowLen - 1, refresh TTL, reattach sys + trailingSys
  - Rewind+edit (prefix match): findLongestPrefixMatch(entry.fps, windowFP);
    - If prefix covers all but last message, cut lynchpin to the matched prefix, append edited suffix from windowMsgs, refresh TTL

TTL refresh and cleanup:
- expireAt is refreshed on every successful hit/extension/rewind
- purgeExpiredFirstAnchor() removes stale entries before search

Outcome:
- applyFirstAnchorReconstruction(request) mutates request.body.messages only when a match is found; otherwise it seeds FIRST_ANCHOR_STORE without modifying the request.


## 2) Megaprompt sealing/compaction

Purpose: Seal archival UA turns into a single cached user message that rarely changes, keeping a live editable tail in native form. This stabilizes cached bytes and reduces breakpoint churn.

Entry point and timing:
- After First Anchor Reconstruction, but before downstream provider-specific conversion, the route may compact the prompt.
- In chat-completions.js, the server slices off leading system messages, compacts the remainder via applyMegapromptCompaction(), then reattaches the leading systems.

Key configuration:
- claude.megaprompt.enabled (boolean)
- claude.megaprompt.turnMultiple (default 60; tests may use smaller, e.g., 4)
- claude.megaprompt.minLiveTailTurns (default 10; tests use 2)
- claude.extendedTTL (true => 1h; false => 5m)
- claude.cachingAtDepth (feeds “tail size” derivation)

Algorithm (UA = user/assistant turns only):
- Compute UA indices uaIdx within messages (tools excluded from UA set)
- Derive base tail size from cachingAtDepth: baseTailTurns = max(2*cachingAtDepth, minLiveTail)
- archivalTurns = max(0, uaIdx.length - baseTailTurns)
- If archivalTurns < turnMultiple: return (do not seal yet)
- lastMultiple = floor(archivalTurns / turnMultiple) * turnMultiple
- effectiveTailTurns = max(baseTailTurns, archivalTurns - lastMultiple)
- sealedMsgIdx = first lastMultiple UA indices
- tailMsgIdx = last effectiveTailTurns UA indices
- Build deterministic sealed text from sealedMsgIdx (text only; no ids/timestamps; tool content excluded)
- Create a single user message:
  - content: [{ type: 'text', text, cache_control: { type: 'ephemeral', ttl } }]
  - marker: _megapromptSealed: { ttl }
- Final prompt: [sealedMegaprompt(user)] + [live tail from first tail UA index to end], preserving original order (tools remain in tail)

Keeping sealed cache_control through merges:
- Post-processing or conversion steps may merge messages/parts. The sealed message is tagged with _megapromptSealed so convertClaudeMessages() can re-attach cache_control to its first text part afterwards.

Stability properties:
- Sealed text grows only when crossing the next completed UA multiple (e.g., from 1–4 to 1–8 turns), providing byte stability across many turns
- Tools are excluded from sealed text by design but preserved in the live tail


## 3) Caching-at-depth (Claude + OpenRouter)

Purpose: Place Anthropic cache_control anchors predictably while respecting the provider’s 4-breakpoint limit and keeping anchors stable as the prompt evolves.

Key entry points:
- Direct Anthropic (Messages API): cachingAtDepthForClaude(messages, depth, ttl)
  - Called on convertedPrompt.messages after convertClaudeMessages()
  - Adds headers: prompt-caching-2024-07-31 and extended-cache-ttl-2025-04-11 when enabled
- OpenRouter (Chat Completions-style): cachingAtDepthForOpenRouterClaude(messages, depth, ttl)
  - Called on request.body.messages

Shared rules and configuration:
- stripClaudeCacheBreakpoints(messages) is always run first to remove stale cache_control blocks, avoiding breakpoint overflow
- claude.maxAnchors (default 3) = message-anchors budget; 1 additional anchor is reserved for system prompt caching (total ≤ 4 overall)
- claude.anchorSpacingBlocks (default 20) = fixed spacing window for anchors (block = one content part)
- claude.extendedTTL toggles cache TTL between 5m and 1h
- Depth semantics: “nth user from the end”, where n=0 is the most recent user message; assistant prefill is skipped until first user is seen
- Backward coverage: Anthropic anchors cache the prefix up to the anchor; we do not pre-anchor the sealed megaprompt — spacing anchors cover it when within W

Phase 1 – spacing anchors (fixed W‑multiples):
- Treat spacing as fixed thresholds at W, 2W, 3W (W = claude.anchorSpacingBlocks)
- For each threshold crossed, place an anchor at the last user at/before the threshold; if none, anchor the current non‑user (warn)

Phase 2 – optional tail‑depth anchor:
- If budget remains (anchorsPlaced < MAX_ANCHORS) and the tail has ≥ n+1 users after the last anchor, place one “floating” depth anchor on the nth user from the tail
- No retargeting and no spacing adjustments; if budget is exhausted or tail insufficient, skip

Phase 3 – diagnostics:
- If total content blocks exceed MAX_ANCHORS × anchorSpacingBlocks, log a warning that parts of the prompt may be uncached

System prompt caching:
- Direct Anthropic (Messages API): if enableSystemPromptCache and using a system prompt, tag the last part of the system array with cache_control
- OpenRouter: tag the last leading system message’s last text part (or last part) with cache_control
- Net effect: 1 reserved system anchor + up to MAX_ANCHORS message anchors (sealed counts toward message anchors) keeps us under 4

OpenRouter specifics:
- cachingAtDepthForOpenRouterClaude ensures message.content is an array when placing anchors, converting strings to [{type:'text',text}] when required
- Anchoring logs clearly denote OpenRouter path when falling back to a non-user message or skipping depth anchor due to spacing


## Order of operations (why it matters)

- First Anchor Reconstruction runs first so later stages act on the reconstructed, stable long-tail window (and TTL refresh happens on use)
- Megaprompt compaction runs next to seal archival content into a fixed user message; this produces a durable _megapromptSealed marker
- Provider-specific conversions (MERGE/convert) happen after, and the sealed cache_control is re-attached during Claude conversion
- Anchoring (caching-at-depth) runs last, and begins with a pre-pass that strips any stale cache_control anchors; then applies fixed‑multiple spacing (W, 2W, 3W) and an optional tail‑depth anchor if budget remains
- System prompt caching is applied in the provider-specific path (Anthropic or OpenRouter), consuming one of the 4 total breakpoints available at Anthropic


## Sequence diagrams

```mermaid
sequenceDiagram
  participant C as Client
  participant S as SillyTavern Server
  participant A as Anthropic (Claude)

  C->>S: /generate (messages)
  Note over S: 1) FirstAnchor: reconstruct sys + lynchpin + trailingSys
  Note over S: 2) Megaprompt: seal archival UA to single user
  Note over S: 3) Convert (Claude Messages API)
  Note over S: 4) Anchors: strip stale; spacing at W,2W,3W; optional depth if budget remains
  Note over S: 5) System prompt cache (if enabled)

  S->>A: messages + cache_control
  A-->>S: response
  S-->>C: stream/final
```

```mermaid
sequenceDiagram
  participant C as Client
  participant S as SillyTavern Server
  participant O as OpenRouter (Claude)

  C->>S: /generate (messages)
  Note over S: 1) FirstAnchor
  Note over S: 2) Megaprompt
  Note over S: 3) Post-process (MERGE/SEMI/STRICT) as configured
  Note over S: 4) Anchors (OpenRouter path): strip stale; spacing at W,2W,3W; optional depth if budget remains
  Note over S: 5) System prompt cache (tag last leading system)

  S->>O: messages + cache_control
  O-->>S: response
  S-->>C: stream/final
```


## Edge cases and guarantees

- Jump-back edits (11 → 11E): First Anchor handles rewinds and rewind+edit via subsequence/prefix matching, then rebuilds the full message array as [sys + lynchpin + trailingSys]; TTL refreshed
- Trailing system messages: Explicitly trimmed from the reconstruction window and always re-attached at the end of the rebuilt array
- Tool messages: Excluded from sealed text; preserved in the live tail, in original order
- Prefill assistant: Depth search for “nth user from end” skips assistant prefills until after the first user message
- Stale anchors: All anchoring functions begin by stripClaudeCacheBreakpoints(messages) to avoid exceeding Anthropic’s breakpoint limit
- Breakpoint limit: With 1 system + up to MAX_ANCHORS (default 3) message anchors, total anchors remain ≤ 4 (Anthropic max)
- Stability: Sealed megaprompt remains byte-stable across turns until the next complete multiple threshold is crossed
- Overbudget context: If total blocks exceed MAX_ANCHORS × spacing, a warning is logged; anchoring still proceeds conservatively


## Configuration reference (relevant)

- claude.firstAnchorCaching.enabled / ttlSeconds / allowRewind
- claude.megaprompt.enabled / turnMultiple / minLiveTailTurns
- claude.cachingAtDepth (>= 0 to enable anchoring) 
- claude.maxAnchors (message anchors budget; default 3)
- claude.anchorSpacingBlocks (default 20)
- claude.extendedTTL (true = 1h; false = 5m)
- claude.enableSystemPromptCache (true to tag system prompt)


## Code path map (where things happen)

- First Anchor
  - chat-completions.js: router.post('/generate') → applyFirstAnchorReconstruction(request)
  - first-anchor-cache.js: matching/rewind/TTL
- Megaprompt
  - chat-completions.js: slice leading systems; applyMegapromptCompaction() to remainder; reattach systems
  - prompt-converters.js: applyMegapromptCompaction(); sealed marker + cache_control assignment; convertClaudeMessages() re-attaches sealed cache_control after MERGE
- Anchoring
  - Anthropic (Claude): convertClaudeMessages() → cachingAtDepthForClaude(); set Anthropic beta headers for caching features
  - OpenRouter (Claude): post-processed messages → cachingAtDepthForOpenRouterClaude(); system prompt cache tagging for last leading system message


## Testing and verification

- Unit tests demonstrate:
  - Stability and thresholds for sealing; exclusion of tools; retention of sealed cache_control post-merge
  - Reconstruction behaviors for extend/rewind/rewind+edit; trailing system handling

To run the tests locally:
- Use your standard node test runner (see src/tests/…); tests set env flags for config values at module init.


## Notes and design choices

- Sealed-first anchoring: If a sealed megaprompt exists, it’s anchored first to maximize cache hits on the (largest) sealed chunk
- Conservative anchors by default: The default of 2 message anchors (+1 system = ≤3 total) leaves margin under Anthropic’s 4-breakpoint max
- “Blocks” not “messages”: Spacing is in content blocks (parts), allowing multi-modal prompts while keeping the window at ~20 blocks
- Fail-open approach: On errors in compaction or reconstruction, functions return the original prompt without disruption


## Summary

- FirstAnchor reconstructs a stable, TTL-refreshed context (handles rewinds + edits) and reattaches trailing system messages before any compaction
- Megaprompt compaction seals archival UA turns into a stable, cached user message and leaves a live tail; the sealed marker ensures cache_control persists through merges/conversions
- Caching-at-depth applies spacing + depth rules, strips stale anchors, prioritizes the sealed chunk, and keeps anchors under provider limits; system prompt caching is applied in both Anthropic and OpenRouter paths

This combination keeps cache hits high, cache bytes stable, and behavior predictable across providers and editing workflows.

