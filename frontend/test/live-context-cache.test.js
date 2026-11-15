import assert from 'node:assert/strict'
import { after, before, describe, it } from 'node:test'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'
import { diffCaches, lcsRanges } from '../src/turnDiffs.js'
import { buildMapModel } from '../src/mapModel.js'

let vite
let Stage
let App
let Lens
let TapeMap

before(async () => {
  vite = await createServer({
    configFile: './vite.config.js',
    optimizeDeps: { noDiscovery: true, include: [] },
    server: { middlewareMode: true },
  })
  ;({ default: Stage } = await vite.ssrLoadModule('/src/Stage.jsx'))
  ;({ default: App } = await vite.ssrLoadModule('/src/App.jsx'))
  ;({ default: Lens } = await vite.ssrLoadModule('/src/Lens.jsx'))
  ;({ default: TapeMap } = await vite.ssrLoadModule('/src/TapeMap.jsx'))
})

after(async () => {
  await vite?.close()
})

// A two-turn conversation: context 2 shares a 10-token prefix with entry #1.
const T2 = ['<|im_start|>', 'user', '\n', 'My', ' name', ' is', ' Alice', '.', '<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n', 'Nice', ' to', ' meet', ' you', ',', ' Alice', '!', '<|im_end|>', '\n', '<|im_start|>', 'user', '\n', 'What', ' is', ' my', ' name', '?', '<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n', '<think>', '\n']
const ENTRY1 = T2.slice(0, 10)
const report = {
  token_ids: T2,
  token_pieces: T2.map((piece, index) => ({ id: index + 100, piece, text: piece, special: piece.startsWith('<') })),
  spans: [
    { role: 'user', index: 1, start: 0, end: 10, kind: 'message' },
    { role: 'assistant', index: 2, start: 10, end: 22, kind: 'message' },
    { role: 'user', index: 3, start: 22, end: 32, kind: 'message' },
    { role: 'assistant', start: 32, end: 37, kind: 'opener' },
  ],
  cache_prediction: { entry_id: 1, reused_tokens: 10, reason: 'would reuse 10 of 37 tokens from checkpoint entry #1' },
  cache_entries: [],
}

function stageProps(overrides = {}) {
  return {
    state: {
      config: { cache_reuse: true, cache_checkpoints: 'last_message', system_prompt: '' },
      cache: {
        entries: [
          { id: 1, kind: 'checkpoint', tokens: 10, nbytes: 5_000_000, age_seconds: 5, relabels: 0 },
          { id: 2, kind: 'after-generation', tokens: 13, nbytes: 6_000_000, age_seconds: 2, relabels: 0 },
        ],
        total_bytes: 11_000_000,
        limit_bytes: 2_147_483_648,
        utilization: 0.01,
        evictions_total: 0,
      },
      model_loaded: true,
    },
    session: { mode: 'messages', messages: [] },
    draft: '',
    preview: report,
    previewError: null,
    generation: null,
    tokens: [],
    running: false,
    busy: false,
    modelLoaded: true,
    turns: [],
    viewTurn: null,
    viewTurnId: null,
    prevTurn: null,
    pendingEdits: [],
    systemPrompt: '',
    runProgress: null,
    hoveredMessage: null,
    bubbleClick: null,
    onViewTurnChange() {},
    onApplyStaged() {},
    onCacheAction() {},
    onConfigChange() {},
    onClearCache() {},
    onEditingChange() {},
    onLoadModel() {},
    ...overrides,
  }
}

describe('the visual stage', () => {
  it('renders the App startup state without a temporal-dead-zone failure', () => {
    const html = renderToStaticMarkup(React.createElement(App))
    assert.match(html, /Qwen Harness/)
    assert.match(html, /connecting to the local backend/)
  })

  it('draws the ruler, the pipeline and the caption with the server reuse numbers', () => {
    const html = renderToStaticMarkup(React.createElement(Stage, stageProps()))
    assert.match(html, /position<\/text>/)
    assert.match(html, /Cache lookup/)
    assert.match(html, />10 from #1</)
    assert.match(html, /CACHE ENTRIES \(saved states\)/)
    assert.match(html, /Context<\/text>/)
    assert.match(html, /reuse<\/label>|reuse/)
  })

  it('shows the reuse-off caption and still draws the bars', () => {
    const model = buildMapModel({
      report,
      entries: [{ id: 1, kind: 'checkpoint', token_ids: ENTRY1 }],
      reuseEnabled: false,
    })
    assert.equal(model.reused, 0)
    const text = model.caption.map((part) => part.text).join('')
    assert.match(text, /reuse is off/)
    assert.equal(model.rows.length, 1)
    assert.equal(model.rows[0].breakAt, null)
    assert.equal(model.rows[0].match, 10)
  })

  it('renders the empty page: ghost ruler, empty tape, no saved states', () => {
    const html = renderToStaticMarkup(
      React.createElement(Stage, stageProps({
        preview: { token_ids: [], token_pieces: [], spans: [], cache_prediction: null, cache_entries: [] },
        state: { config: {}, cache: { entries: [], total_bytes: 0, limit_bytes: 2000, utilization: 0 }, model_loaded: true },
      })),
    )
    assert.match(html, /type a message to see its tokens/)
    assert.match(html, /no saved states yet/)
  })

  it('renders a historical turn read-only with its recorded reuse', () => {
    const html = renderToStaticMarkup(
      React.createElement(Stage, stageProps({
        viewTurn: {
          id: 'turn-1',
          n: 1,
          input: 37,
          reused: 10,
          generated: 8,
          thinking: 0,
          context: report,
          cache: {
            entries: [{ id: 1, kind: 'checkpoint', tokens: 10, token_ids: ENTRY1, nbytes: 5_000_000 }],
            total_bytes: 5_000_000,
            last_lookup: { entry_id: 1, reused_tokens: 10, reason: 'reused 10 tokens from checkpoint entry #1' },
          },
          edits: [],
        },
        viewTurnId: 'turn-1',
      })),
    )
    assert.match(html, /Historical snapshot · read-only/)
    assert.match(html, /Turn 1/)
  })
})

describe('the map view model (plan cases)', () => {
  it('marks a full-prefix entry usable and chosen; the longest match breaks at the first difference', () => {
    const model = buildMapModel({
      report,
      entries: [
        { id: 1, kind: 'checkpoint', token_ids: ENTRY1 },
        { id: 2, kind: 'after-generation', token_ids: [...T2.slice(0, 13), '<think>', 'x'] },
      ],
      reuseEnabled: true,
    })
    assert.equal(model.reused, 10)
    assert.equal(model.computed, 27)
    const entry1 = model.rows.find((row) => row.id === 1)
    assert.equal(entry1.match, 10)
    assert.equal(entry1.breakAt, null)
    assert.equal(entry1.usable, true)
    assert.equal(entry1.chosen, true)
    const entry2 = model.rows.find((row) => row.id === 2)
    assert.equal(entry2.breakAt, 13)
    assert.equal(entry2.usable, false)
  })

  it('an edit at position 6 breaks every bar and drops reuse to 0', () => {
    const edited = T2.map((piece, index) => (index === 6 ? ' Bob' : piece))
    const model = buildMapModel({
      report: {
        token_ids: edited,
        token_pieces: edited.map((piece, index) => ({ id: index + 100, piece, text: piece })),
        spans: report.spans,
        cache_prediction: { entry_id: null, reused_tokens: 0, reason: 'no compatible cache entry' },
        cache_entries: [],
      },
      entries: [{ id: 1, kind: 'checkpoint', token_ids: ENTRY1 }],
      reuseEnabled: true,
    })
    assert.equal(model.reused, 0)
    const entry = model.rows[0]
    assert.equal(entry.match, 6)
    assert.equal(entry.breakAt, 6)
    assert.equal(entry.usable, false)
    assert.match(model.caption.map((part) => part.text).join(''), /matches to position 6 then breaks/)
  })

  it('warns when the client rule disagrees with the server assessment', () => {
    const originalWarn = console.warn
    const warnings = []
    console.warn = (message) => warnings.push(message)
    try {
      buildMapModel({
        report: {
          ...report,
          // The server claims this full-prefix entry cannot be reused.
          cache_entries: [{ id: 1, kind: 'checkpoint', tokens: 10, shared_prefix: 10, reusable_tokens: 0, reason: 'x' }],
        },
        entries: [{ id: 1, kind: 'checkpoint', token_ids: ENTRY1 }],
        reuseEnabled: true,
      })
    } finally {
      console.warn = originalWarn
    }
    assert.equal(warnings.length, 1)
    assert.match(warnings[0], /disagrees with the server/)
  })
})

describe('Stage states (SSR)', () => {
  const noPrediction = { ...report, cache_prediction: { entry_id: null, reused_tokens: 0, reason: 'no compatible cache entry' } }
  const turn = (n) => ({ id: `turn-${n}`, n, context: report, cache: { entries: [] } })

  it('a typed turn-2 state shows the prediction in the pipeline and the loading caption', () => {
    const html = renderToStaticMarkup(
      React.createElement(Stage, stageProps({
        turns: [turn(1), turn(2)],
        prevTurn: turn(1),
        draft: 'What is my name?',
      })),
    )
    assert.match(html, />10 from #1</)
    assert.match(html, />27 tokens</)
    // Live entry token lists load in an effect, so SSR shows the pending caption.
    assert.match(html, /reading the saved states on the server/)
    assert.match(html, /Cache lookup/)
  })

  it('reuse off shows no match and every token computed', () => {
    const html = renderToStaticMarkup(
      React.createElement(Stage, stageProps({
        state: { ...stageProps().state, config: { cache_reuse: false } },
      })),
    )
    assert.doesNotMatch(html, />10 from #1</)
    assert.match(html, />no match</)
    assert.match(html, />37 tokens</)
  })

  it('a prediction of nothing reused reads as no match', () => {
    const html = renderToStaticMarkup(React.createElement(Stage, stageProps({ preview: noPrediction })))
    assert.match(html, />no match</)
    assert.match(html, />37 tokens</)
  })

  it('a historical turn draws its bars, the restored entry and the recorded caption', () => {
    const entry2 = { id: 2, kind: 'after-generation', token_ids: [...T2.slice(0, 13), '<think>', 'x'], nbytes: 1 }
    const html = renderToStaticMarkup(
      React.createElement(Stage, stageProps({
        viewTurn: {
          id: 'turn-3',
          n: 3,
          context: report,
          thinking: 2,
          generated: 6,
          cache: {
            entries: [{ id: 1, kind: 'checkpoint', token_ids: ENTRY1, nbytes: 1 }, entry2],
            last_lookup: { entry_id: 1, reused_tokens: 10 },
          },
        },
        viewTurnId: 'turn-3',
      })),
    )
    assert.match(html, /Turn 3: /)
    assert.match(html, /restores 10 tokens from #1/)
    assert.match(html, /✓ restored/)
    assert.match(html, /breaks at 13/)
    assert.match(html, />2 thinking · 4 answer</)
  })
})

describe('Lens (SSR)', () => {
  const entries = [
    { id: 1, kind: 'checkpoint', token_ids: ENTRY1 },
    { id: 2, kind: 'after-generation', token_ids: [...T2.slice(0, 6), 'DIFFERENT', ...T2.slice(7, 12)] },
  ]
  const model = buildMapModel({ report, entries })

  it('marks same chips before the break and a diff chip at the break', () => {
    const html = renderToStaticMarkup(React.createElement(Lens, { model, selectedEntryId: 2 }))
    assert.match(html, /entry #2 breaks at position 6/)
    assert.match(html, /chip same/)
    assert.equal((html.match(/chip diff/g) || []).length, 2) // context row + entry row
    assert.match(html, /chip after/)
    assert.match(html, /after-reply/)
    // Entry labels are shown as raw ids when no decoded pieces are supplied.
    assert.match(html, /⟨DIFFERENT⟩/)
  })

  it('a fully matching entry has no diff chip and says so', () => {
    const html = renderToStaticMarkup(React.createElement(Lens, { model, selectedEntryId: 1 }))
    assert.match(html, /entry #1 matches all 10 of its tokens/)
    assert.doesNotMatch(html, /chip diff/)
    assert.match(html, /chip same/)
  })

  it('with no selection it prompts for a bar', () => {
    const html = renderToStaticMarkup(React.createElement(Lens, { model, selectedEntryId: null }))
    assert.match(html, /select a bar to compare/)
    assert.match(html, /no bar selected/)
  })
})

describe('TapeMap (SSR)', () => {
  const entries = [
    { id: 1, kind: 'checkpoint', token_ids: ENTRY1 },
    { id: 2, kind: 'after-generation', token_ids: [...T2.slice(0, 6), 'DIFFERENT', ...T2.slice(7, 12)] },
  ]
  const render = (model, props = {}) =>
    renderToStaticMarkup(React.createElement(TapeMap, { model, ...props }))

  it('draws ruler ticks, one broken bar and one usable ✓ bar', () => {
    const html = render(buildMapModel({ report, entries }))
    assert.match(html, /position<\/text>/)
    assert.match(html, />0<\/text>/)
    assert.match(html, /class="m-tick"/)
    assert.match(html, /breaks at 6/)
    assert.match(html, /✕/)
    assert.match(html, /✓ restored/)
    assert.equal((html.match(/breaks at/g) || []).length, 1)
  })

  it('an unchosen full-prefix bar reads "usable"', () => {
    const html = render(buildMapModel({
      report: { ...report, cache_prediction: { entry_id: null, reused_tokens: 0 } },
      entries: [entries[0]],
    }))
    assert.match(html, /✓ usable/)
    assert.doesNotMatch(html, /✓ restored/)
  })

  it('marks new bars, evicted ghosts and "+" ticks', () => {
    const model = buildMapModel({
      report,
      entries: [entries[0]],
      prevEntries: [{ id: 9, kind: 'checkpoint', token_ids: ENTRY1 }],
      prevIds: T2.slice(0, 20),
      comparePrev: true,
    })
    const html = render(model)
    assert.match(html, />NEW <\/tspan>/)
    assert.match(html, />evicted</)
    assert.match(html, /class="m-ghost"|m-ghost/)
    assert.equal((html.match(/>\+<\/text>/g) || []).length, model.ticks.length)
  })

  it('folds bars past maxRows into a "+N more" row', () => {
    const html = render(buildMapModel({ report, entries }), { maxRows: 1 })
    assert.match(html, /\+1 more saved state \(longest first\)/)
  })

  it('shows the relabel warning on a relabeled bar', () => {
    const html = render(buildMapModel({ report, entries: [{ ...entries[0], relabels: 1 }] }))
    assert.match(html, /label ≠ state/)
  })

  it('the empty page draws the ghost tape and the empty-cache line', () => {
    const html = render(buildMapModel({ report: { token_ids: [], token_pieces: [], spans: [] }, entries: [] }))
    assert.match(html, /type a message to see its tokens/)
    assert.match(html, /no saved states yet/)
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
