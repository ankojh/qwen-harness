import assert from 'node:assert/strict'
import { after, before, describe, it } from 'node:test'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'
import { diffCaches, lcsRanges } from '../src/turnDiffs.js'

let vite
let CacheMatch
let CachePanel
let App

before(async () => {
  vite = await createServer({
    configFile: './vite.config.js',
    optimizeDeps: { noDiscovery: true, include: [] },
    server: { middlewareMode: true },
  })
  ;({ default: CacheMatch } = await vite.ssrLoadModule('/src/CacheMatch.jsx'))
  ;({ default: CachePanel } = await vite.ssrLoadModule('/src/CachePanel.jsx'))
  ;({ default: App } = await vite.ssrLoadModule('/src/App.jsx'))
})

after(async () => {
  await vite?.close()
})

const report = {
  token_ids: [10, 11, 12, 13],
  token_pieces: [10, 11, 12, 13].map((id) => ({ id, piece: `piece-${id}`, text: `piece-${id}` })),
  cache_prediction: {
    entry_id: 7,
    reused_tokens: 2,
    reason: 'would reuse 2 of 4 tokens from checkpoint entry #7',
  },
  cache_entries: [{
    id: 7,
    kind: 'checkpoint',
    tokens: 3,
    shared_prefix: 2,
    reusable_tokens: 2,
    reason: 'First different token at position 2.',
    token_window: [10, 11, 99].map((id, position) => ({ position, id, piece: `piece-${id}` })),
  }],
}

describe('live context and cache UI', () => {
  it('renders the App startup state without a temporal-dead-zone failure', () => {
    const html = renderToStaticMarkup(React.createElement(App))
    assert.match(html, /Qwen Harness/)
    assert.match(html, /connecting to the local backend/)
  })

  it('renders exact token positions and IDs around the cache match boundary', () => {
    const html = renderToStaticMarkup(
      React.createElement(CacheMatch, { report, reuseEnabled: true, running: false }),
    )
    assert.match(html, /Next request: 2 tokens of computation reusable · 2 to process/)
    assert.match(html, /positions 0–1/)
    assert.match(html, /First different token at position 2/)
    assert.match(html, /<td>0<\/td>/)
    assert.match(html, /<td>2<\/td>/)
    assert.match(html, /ID 10/)
    assert.match(html, /ID 99/)
    assert.match(html, /≠ first difference/)
  })

  it('keeps entry match metadata visible when cache reuse is disabled', () => {
    const disabledReport = {
      ...report,
      cache_prediction: { ...report.cache_prediction, reused_tokens: 0, reason: 'Cache reuse is OFF.' },
    }
    const html = renderToStaticMarkup(
      React.createElement(CacheMatch, { report: disabledReport, reuseEnabled: false, running: false }),
    )
    assert.match(html, /Next request: 0 tokens of computation reusable · 4 to process/)
    assert.match(html, /Reuse is OFF\. Matching entries are still shown for comparison\./)
    assert.match(html, /2 tokens of usable saved state \(disabled\)/)
    assert.match(html, /2 matching · 2 usable/)
  })

  it('renders the no-cache state when there are no saved entries', () => {
    const html = renderToStaticMarkup(
      React.createElement(CacheMatch, {
        report: { token_ids: [1], cache_prediction: null, cache_entries: [] },
        reuseEnabled: true,
        running: false,
      }),
    )
    assert.match(html, /No saved states yet/)
  })
})

describe('cache snapshot diffs', () => {
  it('marks relabeling as a modification when entry id and token count stay fixed', () => {
    const diff = diffCaches(
      { entries: [{ id: 7, token_ids: [10, 11, 12] }], total_bytes: 300 },
      { entries: [{ id: 7, token_ids: [10, 99, 12] }], total_bytes: 300 },
      [10, 99, 12, 13],
    )

    assert.equal(diff.added.length, 0)
    assert.equal(diff.evicted.length, 0)
    assert.equal(diff.memoryChange, 0)
    assert.equal(diff.modified.length, 1)
    assert.deepEqual(diff.modified[0].changes, [{ position: 1, before: 11, after: 99 }])
  })
})

describe('long context token diffs', () => {
  it('returns no changed ranges for 1500 identical tokens', () => {
    const ids = Array.from({ length: 1500 }, (_, index) => index)
    assert.deepEqual(lcsRanges(ids, [...ids]), {
      removed: [],
      added: [],
      approximate: false,
    })
  })

  it('isolates a single replacement in a 1500-token context exactly', () => {
    const before = Array.from({ length: 1500 }, (_, index) => index)
    const after = [...before]
    after[750] = 9999

    assert.deepEqual(lcsRanges(before, after), {
      removed: [{ start: 750, end: 751 }],
      added: [{ start: 750, end: 751 }],
      approximate: false,
    })
  })

  it('marks only a large changed interior approximate and leaves equal edges unmarked', () => {
    const prefix = Array.from({ length: 20 }, (_, index) => index)
    const beforeInterior = Array.from({ length: 1500 }, (_, index) => 100 + index)
    const afterInterior = Array.from({ length: 1500 }, (_, index) => 10_000 + index)
    const suffix = Array.from({ length: 20 }, (_, index) => 50_000 + index)
    const before = [...prefix, ...beforeInterior, ...suffix]
    const after = [...prefix, ...afterInterior, ...suffix]

    assert.deepEqual(lcsRanges(before, after), {
      removed: [{ start: 20, end: 1520 }],
      added: [{ start: 20, end: 1520 }],
      approximate: true,
    })
  })
})

describe('historical cache panel', () => {
  it('uses the selected turn prefill stats and disables clearing the live cache', () => {
    const html = renderToStaticMarkup(React.createElement(CachePanel, {
      state: {
        config: { cache_reuse: true, cache_checkpoints: 'last_message' },
        cache: { entries: [], total_bytes: 900, limit_bytes: 2000, utilization: 0.45, evictions_total: 3 },
      },
      preview: null,
      previewError: null,
      running: false,
      generation: {
        prompt: { token_ids: Array.from({ length: 80 }, (_, index) => index) },
        stats: { prefill: { processed_tokens: 42, seconds: 9.9 } },
      },
      busy: false,
      onClearCache: () => {},
      viewTurn: {
        id: 'turn-1',
        prefill: { processed_tokens: 7, seconds: 1.2 },
        finish: { phase: 'decode' },
        context: { token_ids: Array.from({ length: 9 }, (_, index) => index) },
        cache: {
          entries: [{ id: 8, kind: 'checkpoint', tokens: 8, token_ids: Array.from({ length: 8 }, (_, index) => index), nbytes: 400 }],
          total_bytes: 400,
          last_lookup: { entry_id: 8, reused_tokens: 1, reason: 'fixture lookup' },
        },
        edits: [],
      },
      prevTurn: null,
      pendingEdits: [],
      onConfigChange: () => {},
      onCacheAction: () => {},
    }))

    assert.match(html, /Last actual request · reused 1 tokens · processed 8 input tokens/)
    assert.match(html, /Input processing before the final token: 7 tokens in 1\.2s/)
    assert.doesNotMatch(html, /42 tokens in 9\.9s/)
    assert.match(html, /<button[^>]*disabled=""[^>]*>Clear cache<\/button>/)
  })
})
