import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { addedPositions, assessEntry, buildMapModel, kindLabel, rulerStep } from '../src/mapModel.js'

// A 15-token context; entries below are prefixes of it, or diverge from it.
const CTX = ['<|im_start|>', 'user', '\n', 'Hi', '.', '<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n', 'Yo', '<|im_end|>', '\n', '<|im_start|>', 'assistant']

function makeReport(ids = CTX, extra = {}) {
  return {
    token_ids: ids,
    token_pieces: ids.map((piece, index) => ({ id: index + 100, piece, text: piece })),
    spans: [
      { role: 'user', index: 1, start: 0, end: 7, kind: 'message' },
      { role: 'assistant', index: 2, start: 7, end: ids.length, kind: 'message' },
    ],
    cache_prediction: null,
    cache_entries: [],
    ...extra,
  }
}

const prefix = (n) => CTX.slice(0, n)
const captionText = (model) => model.caption.map((part) => part.text).join('')

// Run a build with console.warn captured, then restore it.
function captureWarnings(fn) {
  const original = console.warn
  const warnings = []
  console.warn = (...args) => warnings.push(args.join(' '))
  try {
    return { result: fn(), warnings }
  } finally {
    console.warn = original
  }
}

describe('the reuse rule (assessEntry)', () => {
  it('a full-prefix checkpoint matches all of itself and is usable', () => {
    const row = assessEntry({ id: 1, kind: 'checkpoint', token_ids: prefix(7) }, CTX, true)
    assert.equal(row.match, 7)
    assert.equal(row.len, 7)
    assert.equal(row.breakAt, null)
    assert.equal(row.usable, true)
  })

  it('a longer entry that diverges breaks at the first difference and is not usable', () => {
    const entry = { id: 2, kind: 'after-generation', token_ids: [...prefix(6), 'DIFFERENT', ...CTX.slice(7)] }
    const row = assessEntry(entry, CTX, true)
    assert.equal(row.match, 6)
    assert.equal(row.breakAt, 6)
    assert.equal(row.usable, false)
  })

  it('an entry that covers the whole context is not usable: nothing would be left to compute', () => {
    const exact = assessEntry({ id: 3, token_ids: [...CTX] }, CTX, true)
    assert.equal(exact.match, CTX.length)
    assert.equal(exact.breakAt, null)
    assert.equal(exact.usable, false)

    const longer = assessEntry({ id: 4, token_ids: [...CTX, 'extra'] }, CTX, true)
    assert.equal(longer.match, CTX.length)
    assert.equal(longer.breakAt, CTX.length)
    assert.equal(longer.usable, false)
  })

  it('a zero-length context matches nothing and nothing is usable', () => {
    const row = assessEntry({ id: 5, token_ids: ['a', 'b'] }, [], true)
    assert.equal(row.match, 0)
    assert.equal(row.breakAt, 0)
    assert.equal(row.usable, false)
    const empty = assessEntry({ id: 6, token_ids: [] }, [], true)
    assert.equal(empty.usable, false)
  })

  it('reads token_ids, falls back to tokens, and carries relabel/size metadata', () => {
    const row = assessEntry({ id: 7, tokens: prefix(3), relabels: 2, nbytes: 10, age_seconds: 4, after_message: 3 }, CTX, true)
    assert.equal(row.len, 3)
    assert.equal(row.relabeled, true)
    assert.equal(row.relabelCount, 2)
    assert.equal(row.nbytes, 10)
    assert.equal(row.ageSeconds, 4)
    assert.equal(row.afterMessage, 3)
    assert.equal(kindLabel('manual'), 'saved by you')
    assert.equal(kindLabel('after-generation'), 'after-reply')
  })
})

describe('the band split uses the server numbers', () => {
  const entries = [{ id: 1, kind: 'checkpoint', token_ids: prefix(7) }]

  it('next mode: reused comes from cache_prediction.reused_tokens', () => {
    const model = buildMapModel({
      report: makeReport(CTX, { cache_prediction: { entry_id: 1, reused_tokens: 7 } }),
      entries,
    })
    assert.equal(model.reused, 7)
    assert.equal(model.computed, 8)
    assert.equal(model.totalProcessed, 8)
    assert.equal(model.phase, 'next')
  })

  it('next mode trusts the server even when the client rule would give a different number', () => {
    const model = buildMapModel({
      report: makeReport(CTX, { cache_prediction: { entry_id: null, reused_tokens: 0 } }),
      entries,
    })
    assert.equal(model.reused, 0)
    assert.equal(model.computed, 15)
    assert.equal(model.chosenEntryId, null)
  })

  it('running mode: lookup gives reused, progress.processed_tokens fills the computed band', () => {
    const model = buildMapModel({
      report: makeReport(),
      entries,
      running: true,
      lookup: { entry_id: 1, reused_tokens: 7 },
      progress: { processed_tokens: 3, total_tokens: 8 },
    })
    assert.equal(model.reused, 7)
    assert.equal(model.computed, 3)
    assert.equal(model.totalProcessed, 3)
    assert.equal(model.phase, 'compute')

    const done = buildMapModel({
      report: makeReport(),
      entries,
      running: true,
      lookup: { entry_id: 1, reused_tokens: 7 },
      progress: { processed_tokens: 8, total_tokens: 8 },
      generated: [{ channel: 'thinking', piece: 'hm' }],
    })
    assert.equal(done.computed, 8)
    assert.equal(done.phase, 'generate')
  })

  it('running mode before the lookup lands: nothing reused, phase is lookup', () => {
    const model = buildMapModel({ report: makeReport(), entries, running: true })
    assert.equal(model.reused, 0)
    assert.equal(model.phase, 'lookup')
  })

  it('running mode: processed tokens never exceed the total still to compute', () => {
    const model = buildMapModel({
      report: makeReport(),
      entries,
      running: true,
      lookup: { entry_id: 1, reused_tokens: 7 },
      progress: { processed_tokens: 50, total_tokens: 8 },
    })
    assert.equal(model.computed, 8)
  })

  it('historical mode: lookup/prefill give reused and computed is the rest', () => {
    const fromLookup = buildMapModel({
      report: makeReport(),
      entries,
      historical: true,
      turnNumber: 4,
      lookup: { entry_id: 1, reused_tokens: 7 },
    })
    assert.equal(fromLookup.reused, 7)
    assert.equal(fromLookup.computed, 8)
    assert.equal(fromLookup.chosenEntryId, 1)
    assert.equal(fromLookup.phase, 'turn')

    const fromPrefill = buildMapModel({
      report: makeReport(),
      entries,
      historical: true,
      prefill: { reused_tokens: 7, processed_tokens: 8 },
    })
    assert.equal(fromPrefill.reused, 7)
    assert.equal(fromPrefill.totalProcessed, 8)
    // Reuse is known from the prefill but there is no lookup to name an entry.
    assert.equal(fromPrefill.chosenEntryId, null)
  })

  it('historical mode ignores a live prediction on the report', () => {
    const model = buildMapModel({
      report: makeReport(CTX, { cache_prediction: { entry_id: 1, reused_tokens: 7 } }),
      entries,
      historical: true,
      lookup: { entry_id: null, reused_tokens: 0 },
    })
    assert.equal(model.reused, 0)
    assert.equal(model.computed, 15)
  })
})

describe('reuse off', () => {
  const entries = [
    { id: 1, kind: 'checkpoint', token_ids: prefix(7) },
    { id: 2, kind: 'after-generation', token_ids: [...prefix(6), 'DIFFERENT', 'x'] },
  ]
  const report = makeReport(CTX, { cache_prediction: { entry_id: 1, reused_tokens: 7 } })

  it('forces reused to 0, the band to all computed, and clears the chosen entry', () => {
    const model = buildMapModel({ report, entries, reuseEnabled: false })
    assert.equal(model.reused, 0)
    assert.equal(model.computed, 15)
    assert.equal(model.chosenEntryId, null)
    assert.equal(model.chosenRelabeled, false)
    assert.ok(model.rows.every((row) => row.chosen === false))
  })

  it('says reuse is off in the caption with the full token count', () => {
    const text = captionText(buildMapModel({ report, entries, reuseEnabled: false }))
    assert.match(text, /reuse is off/)
    assert.match(text, /all 15 tokens will be computed/)
  })

  it('still assesses the bars (match / breakAt), but nothing is usable', () => {
    const model = buildMapModel({ report, entries, reuseEnabled: false })
    const one = model.rows.find((row) => row.id === 1)
    const two = model.rows.find((row) => row.id === 2)
    assert.equal(one.match, 7)
    assert.equal(one.usable, false)
    assert.equal(two.match, 6)
    assert.equal(two.breakAt, 6)
  })

  it('also overrides running-mode lookups', () => {
    const model = buildMapModel({
      report,
      entries,
      running: true,
      reuseEnabled: false,
      lookup: { entry_id: 1, reused_tokens: 7 },
      progress: { processed_tokens: 4, total_tokens: 15 },
    })
    assert.equal(model.reused, 0)
    assert.equal(model.chosenEntryId, null)
  })

  it('does not warn when the server says an entry is usable but reuse is off', () => {
    const { warnings } = captureWarnings(() =>
      buildMapModel({
        report: { ...report, cache_entries: [{ id: 1, reusable_tokens: 7 }] },
        entries,
        reuseEnabled: false,
      }),
    )
    assert.equal(warnings.length, 0)
  })
})

describe('the chosen entry', () => {
  const entries = [
    { id: 1, kind: 'checkpoint', token_ids: prefix(5) },
    { id: 2, kind: 'checkpoint', token_ids: prefix(7) },
  ]

  it('prediction mode: cache_prediction.entry_id marks exactly one row chosen', () => {
    const model = buildMapModel({
      report: makeReport(CTX, { cache_prediction: { entry_id: 2, reused_tokens: 7 } }),
      entries,
    })
    assert.equal(model.chosenEntryId, 2)
    assert.deepEqual(model.rows.filter((row) => row.chosen).map((row) => row.id), [2])
    assert.equal(model.lensCenter, 7)
  })

  it('running mode: lookup.entry_id is chosen', () => {
    const model = buildMapModel({
      report: makeReport(),
      entries,
      running: true,
      lookup: { entry_id: 1, reused_tokens: 5 },
    })
    assert.equal(model.chosenEntryId, 1)
    assert.deepEqual(model.rows.filter((row) => row.chosen).map((row) => row.id), [1])
  })

  it('historical mode: lookup.entry_id is chosen', () => {
    const model = buildMapModel({
      report: makeReport(),
      entries,
      historical: true,
      lookup: { entry_id: 2, reused_tokens: 7 },
    })
    assert.equal(model.chosenEntryId, 2)
  })

  it('no entry is chosen when nothing was reused, even if an id is reported', () => {
    const model = buildMapModel({
      report: makeReport(CTX, { cache_prediction: { entry_id: 2, reused_tokens: 0 } }),
      entries,
    })
    assert.equal(model.chosenEntryId, null)
    assert.ok(model.rows.every((row) => !row.chosen))
  })

  it('chosenRelabeled follows the prediction / lookup relabeled flag', () => {
    const predicted = buildMapModel({
      report: makeReport(CTX, { cache_prediction: { entry_id: 2, reused_tokens: 7, relabeled: true } }),
      entries,
    })
    assert.equal(predicted.chosenRelabeled, true)

    const running = buildMapModel({
      report: makeReport(),
      entries,
      running: true,
      lookup: { entry_id: 2, reused_tokens: 7, relabeled: true },
    })
    assert.equal(running.chosenRelabeled, true)

    const historical = buildMapModel({
      report: makeReport(),
      entries,
      historical: true,
      lookup: { entry_id: 2, reused_tokens: 7, relabeled: true },
    })
    assert.equal(historical.chosenRelabeled, true)

    const plain = buildMapModel({
      report: makeReport(CTX, { cache_prediction: { entry_id: 2, reused_tokens: 7 } }),
      entries,
    })
    assert.equal(plain.chosenRelabeled, false)
  })
})

describe('the caption', () => {
  it('says "restores R tokens from #id, computes N" for a reused prefix', () => {
    const model = buildMapModel({
      report: makeReport(CTX, { cache_prediction: { entry_id: 1, reused_tokens: 7 } }),
      entries: [{ id: 1, kind: 'checkpoint', token_ids: prefix(7) }],
    })
    const text = captionText(model)
    assert.match(text, /restores 7 tokens from #1/)
    assert.match(text, /computes 8/)
    assert.doesNotMatch(text, /label ≠ state/)
    assert.equal(model.caption.find((part) => part.text.startsWith('restores')).kind, 'ok')
  })

  it('uses the singular for a single restored token and prefixes the turn number when historical', () => {
    const model = buildMapModel({
      report: makeReport(),
      entries: [{ id: 1, kind: 'checkpoint', token_ids: prefix(1) }],
      historical: true,
      turnNumber: 3,
      lookup: { entry_id: 1, reused_tokens: 1 },
    })
    const text = captionText(model)
    assert.match(text, /^Turn 3: /)
    assert.match(text, /restores 1 token from #1[^s]/)
  })

  it('adds the "label ≠ state" part when the chosen entry is relabeled', () => {
    const model = buildMapModel({
      report: makeReport(CTX, { cache_prediction: { entry_id: 1, reused_tokens: 7, relabeled: true } }),
      entries: [{ id: 1, kind: 'checkpoint', token_ids: prefix(7), relabels: 1 }],
    })
    const warn = model.caption.find((part) => part.kind === 'warn')
    assert.ok(warn)
    assert.match(warn.text, /label ≠ state/)
  })

  it('says all N tokens are computed when the cache is empty', () => {
    const model = buildMapModel({
      report: makeReport(CTX, { cache_prediction: { entry_id: null, reused_tokens: 0 } }),
      entries: [],
    })
    const text = captionText(model)
    assert.match(text, /the cache is empty/)
    assert.match(text, /all 15 tokens are computed/)
  })

  it('names the longest broken entry: "matches to position 6 then breaks"', () => {
    const model = buildMapModel({
      report: makeReport(CTX, { cache_prediction: { entry_id: null, reused_tokens: 0 } }),
      entries: [
        { id: 1, kind: 'checkpoint', token_ids: [...prefix(3), 'x', 'y'] },
        { id: 2, kind: 'checkpoint', token_ids: [...prefix(6), 'x', 'y'] },
      ],
    })
    const text = captionText(model)
    assert.match(text, /The longest, #2, matches to position 6 then breaks/)
    assert.match(text, /all 15 tokens are computed/)
    assert.equal(model.caption.find((part) => part.kind === 'code').text, '2')
  })

  it('falls back to "no entry is a full prefix" when nothing breaks and nothing is reused', () => {
    // A full-length entry covering the whole context: not usable, no break to report.
    const model = buildMapModel({
      report: makeReport(CTX, { cache_prediction: { entry_id: null, reused_tokens: 0 } }),
      entries: [{ id: 1, kind: 'checkpoint', token_ids: [...CTX] }],
    })
    assert.match(captionText(model), /no entry is a full prefix of this request/)
  })

  it('the empty page invites the user to type', () => {
    const model = buildMapModel({ report: makeReport([], { spans: [] }), entries: [] })
    assert.equal(model.total, 0)
    assert.equal(model.tape.length, 0)
    assert.equal(captionText(model), 'Type a message to see its tokens on the tape.')
    // and with no report at all
    assert.equal(captionText(buildMapModel({})), 'Type a message to see its tokens on the tape.')
  })

  it('cacheKnown=false shows the loading caption instead of "the cache is empty"', () => {
    const model = buildMapModel({
      report: makeReport(CTX, { cache_prediction: { entry_id: null, reused_tokens: 0 } }),
      entries: [],
      cacheKnown: false,
    })
    assert.match(captionText(model), /reading the saved states/)
  })

  it('cacheKnown=false does not hide a caption once entries are known, or on the empty page', () => {
    const withEntries = buildMapModel({
      report: makeReport(CTX, { cache_prediction: { entry_id: 1, reused_tokens: 7 } }),
      entries: [{ id: 1, kind: 'checkpoint', token_ids: prefix(7) }],
      cacheKnown: false,
    })
    assert.match(captionText(withEntries), /restores 7 tokens/)
    const empty = buildMapModel({ report: makeReport([], { spans: [] }), cacheKnown: false })
    assert.match(captionText(empty), /Type a message/)
  })
})

describe('new and evicted bars (prevEntries)', () => {
  const report = makeReport(CTX, { cache_prediction: { entry_id: null, reused_tokens: 0 } })
  const e = (id, len) => ({ id, kind: 'checkpoint', token_ids: prefix(len) })

  it('turn 1 (prevEntries = []) marks every entry new and has no ghosts', () => {
    const model = buildMapModel({ report, entries: [e(1, 5), e(2, 7)], prevEntries: [] })
    assert.deepEqual(model.rows.map((row) => row.isNew), [true, true])
    assert.equal(model.ghosts.length, 0)
  })

  it('a later turn marks only added ids new and removed ids ghost', () => {
    const model = buildMapModel({
      report,
      entries: [e(2, 7), e(3, 9)],
      prevEntries: [e(1, 5), e(2, 7)],
    })
    const byId = Object.fromEntries(model.rows.map((row) => [row.id, row]))
    assert.equal(byId[3].isNew, true)
    assert.equal(byId[2].isNew, false)
    assert.equal(byId[1].ghost, true)
    assert.equal(byId[1].isNew, false)
    assert.deepEqual(model.ghosts.map((row) => row.id), [1])
    // rows are longest first, ghosts included
    assert.deepEqual(model.rows.map((row) => row.id), [3, 2, 1])
  })

  it('ghosts never feed the broken-entry caption', () => {
    const model = buildMapModel({
      report,
      entries: [],
      prevEntries: [{ id: 9, kind: 'checkpoint', token_ids: [...prefix(6), 'x', 'y'] }],
    })
    assert.equal(model.ghosts.length, 1)
    assert.doesNotMatch(captionText(model), /breaks/)
  })

  it('prevEntries = null (not tracked) marks nothing', () => {
    const model = buildMapModel({ report, entries: [e(1, 5)], prevEntries: null })
    assert.equal(model.rows[0].isNew, false)
    assert.equal(model.ghosts.length, 0)
  })

  it('historical views never mark new or ghost', () => {
    const model = buildMapModel({ report, entries: [e(1, 5)], prevEntries: [e(2, 7)], historical: true })
    assert.equal(model.rows[0].isNew, false)
    assert.equal(model.ghosts.length, 0)
  })
})

describe('"+" ticks (comparePrev)', () => {
  const report = (ids) => makeReport(ids)

  it('a single edit at position k yields one tick at k', () => {
    const edited = CTX.map((piece, index) => (index === 4 ? 'CHANGED' : piece))
    const model = buildMapModel({ report: report(edited), prevIds: CTX, comparePrev: true })
    assert.deepEqual(model.ticks, [4])
  })

  it('a pure append yields ticks only for the appended positions', () => {
    const model = buildMapModel({ report: report([...CTX, 'a', 'b', 'c']), prevIds: CTX, comparePrev: true })
    assert.deepEqual(model.ticks, [15, 16, 17])
  })

  it('no ticks unless comparePrev is on, and none without a previous context', () => {
    const next = report([...CTX, 'a'])
    assert.deepEqual(buildMapModel({ report: next, prevIds: CTX, comparePrev: false }).ticks, [])
    assert.deepEqual(buildMapModel({ report: next, prevIds: null, comparePrev: true }).ticks, [])
    assert.deepEqual(buildMapModel({ report: next, prevIds: [], comparePrev: true }).ticks, [])
  })

  it('identical contexts give no ticks', () => {
    assert.deepEqual(buildMapModel({ report: report(CTX), prevIds: [...CTX], comparePrev: true }).ticks, [])
  })
})

describe('server / client disagreement', () => {
  const entries = [{ id: 11, kind: 'checkpoint', token_ids: prefix(7) }]

  it('console.warn names the entry id, and the console is restored afterwards', () => {
    const before = console.warn
    const { warnings } = captureWarnings(() =>
      buildMapModel({
        report: makeReport(CTX, { cache_entries: [{ id: 11, reusable_tokens: 0 }] }),
        entries,
      }),
    )
    assert.equal(console.warn, before)
    assert.equal(warnings.length, 1)
    assert.match(warnings[0], /#11/)
    assert.match(warnings[0], /client rule says usable, server says not usable/)
  })

  it('warns the other way too: server says usable, client says not', () => {
    const { warnings } = captureWarnings(() =>
      buildMapModel({
        report: makeReport(CTX, { cache_entries: [{ id: 12, reusable_tokens: 5 }] }),
        entries: [{ id: 12, kind: 'checkpoint', token_ids: [...prefix(3), 'x'] }],
      }),
    )
    assert.equal(warnings.length, 1)
    assert.match(warnings[0], /#12/)
    assert.match(warnings[0], /client rule says not usable, server says usable/)
  })

  it('stays quiet when the two agree, or when the server has no assessment', () => {
    const agree = captureWarnings(() =>
      buildMapModel({
        report: makeReport(CTX, { cache_entries: [{ id: 11, reusable_tokens: 7 }] }),
        entries,
      }),
    )
    assert.equal(agree.warnings.length, 0)
    const none = captureWarnings(() => buildMapModel({ report: makeReport(CTX), entries }))
    assert.equal(none.warnings.length, 0)
  })

  it('shows the server answer on the row (serverUsable) either way', () => {
    const { result } = captureWarnings(() =>
      buildMapModel({
        report: makeReport(CTX, { cache_entries: [{ id: 11, reusable_tokens: 0 }] }),
        entries,
      }),
    )
    assert.equal(result.rows[0].serverUsable, false)
  })
})

describe('helpers', () => {
  it('rulerStep picks a 1/2/5 × 10ⁿ step for about thirteen ticks', () => {
    assert.equal(rulerStep(10), 1)
    assert.equal(rulerStep(13), 2)
    assert.equal(rulerStep(40), 5)
    assert.equal(rulerStep(130), 20)
    assert.equal(rulerStep(650), 100)
    assert.equal(rulerStep(1300), 200)
    for (const domain of [10, 37, 90, 500, 4000, 32000]) {
      const step = rulerStep(domain)
      assert.ok(step >= 1)
      assert.ok(domain / step <= 14, `domain ${domain} step ${step}`)
    }
  })

  it('addedPositions lists appended and replaced positions in the next context', () => {
    assert.deepEqual(addedPositions(['a', 'b', 'c'], ['a', 'b', 'c', 'd', 'e']), [3, 4])
    assert.deepEqual(addedPositions(['a', 'b', 'c'], ['a', 'X', 'c']), [1])
    assert.deepEqual(addedPositions(['a', 'b', 'c'], ['a', 'b', 'c']), [])
  })

  it('addedPositions is empty without a previous context', () => {
    assert.deepEqual(addedPositions(null, ['a']), [])
    assert.deepEqual(addedPositions([], ['a']), [])
  })

  it('the domain covers the longest of context, generated tail and entries', () => {
    const model = buildMapModel({
      report: makeReport(),
      entries: [{ id: 1, kind: 'checkpoint', token_ids: [...CTX, ...CTX] }],
    })
    assert.equal(model.domain, 30)
    const tail = buildMapModel({ report: makeReport(), generated: Array.from({ length: 20 }, () => ({ channel: 'answer', piece: 'x' })) })
    assert.equal(tail.domain, 35)
    assert.equal(tail.answerCount, 20)
  })
})
