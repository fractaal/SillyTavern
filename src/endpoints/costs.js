import fs from 'node:fs';
import path from 'node:path';
import express from 'express';
import { sync as writeFileAtomicSync } from 'write-file-atomic';

// Global, per SillyTavern instance lifetime cost tally for OpenRouter usage
// Persists under DATA_ROOT/openrouter-costs.json

const FILE_PATH = path.join(globalThis.DATA_ROOT, 'openrouter-costs.json');

/**
 * @typedef {Object} CostStore
 * @property {number} total_base
 * @property {number} total_actual
 * @property {number} count
 * @property {number} updated_at
 */

/** @type {CostStore} */
let store = {
    total_base: 0,
    total_actual: 0,
    count: 0,
    updated_at: 0,
};

function loadStore() {
    try {
        if (fs.existsSync(FILE_PATH)) {
            const raw = fs.readFileSync(FILE_PATH, 'utf-8');
            const data = JSON.parse(raw);
            if (data && typeof data === 'object') {
                store = {
                    total_base: Number(data.total_base) || 0,
                    total_actual: Number(data.total_actual) || 0,
                    count: Number(data.count) || 0,
                    updated_at: Number(data.updated_at) || Date.now(),
                };
            }
        }
    } catch (e) {
        console.warn('Failed to load openrouter-costs.json; starting fresh.', e);
    }
}

function persistStore() {
    try {
        store.updated_at = Date.now();
        writeFileAtomicSync(FILE_PATH, JSON.stringify(store));
    } catch (e) {
        console.error('Failed to persist openrouter-costs.json', e);
    }
}

loadStore();

export const router = express.Router();

router.post('/add', (req, res) => {
    try {
        const base = Number(req.body?.base);
        const actual = Number(req.body?.actual);
        if (!Number.isFinite(base) || !Number.isFinite(actual)) {
            return res.status(400).send({ error: 'Invalid base/actual values' });
        }

        const saved = base - actual;
        store.total_base += base;
        store.total_actual += actual;
        store.count += 1;
        persistStore();

        console.info(`[Costs] Added: base=$${base.toFixed(6)} actual=$${actual.toFixed(6)} saved=$${saved.toFixed(6)} | total_saved=$${(store.total_base - store.total_actual).toFixed(6)} count=${store.count}`);
        return res.send({ ok: true, total_base: store.total_base, total_actual: store.total_actual, total_saved: store.total_base - store.total_actual, count: store.count });
    } catch (e) {
        console.error('Failed to add costs', e);
        return res.status(500).send({ error: 'Internal error' });
    }
});

router.post('/get', (_req, res) => {
    const total_saved = store.total_base - store.total_actual;
    return res.send({ ...store, total_saved });
});

router.post('/reset', (_req, res) => {
    store = { total_base: 0, total_actual: 0, count: 0, updated_at: Date.now() };
    persistStore();
    console.info('[Costs] Reset totals');
    return res.send({ ok: true });
});

