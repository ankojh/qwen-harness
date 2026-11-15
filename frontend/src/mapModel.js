// mapModel.js — the pure view model behind the "see context and cache" stage.
//
// Everything the map, pipeline and caption draw comes out of here as plain
// data: the tape segments (coloured by role), the restored/computed band
// split (always the SERVER's numbers), the generated tail, one row per cache
// entry with its match against the context, and the caption/pipeline facts.
//
// No React, no DOM: unit-testable in Node.  When the client-side reuse rule
// disagrees with the server's assessment of an entry, the server wins and a
// console.warn records the disagreement (that would be a bug to fix).

import { commonPrefix, lcsRanges } from './turnDiffs.js'

export const KIND_LABEL = {
  checkpoint: 'checkpoint',
  manual: 'saved by you',
  'after-generation': 'after-reply',
}

export function kindLabel(kind) {
  return KIND_LABEL[kind] || kind || 'entry'
}

// The reuse rule the view model mirrors (from the plan):
//   match(entry) = leading positions where the entry's token ids equal the
//   context's.  usable = compatible and match >= len and len < context length
//   (at least one token is always computed).  This model isn't trimmable, so
//   a partial match gives 0.  Compatibility is judged by the server; the
//   client treats every entry as compatible and logs when the server differs.
export function assessEntry(entry, contextIds, reuseEnabled) {
  const ids = Array.isArray(entry.token_ids)
    ? entry.token_ids
    : Array.isArray(entry.tokens)
      ? entry.tokens
      : []
  const len = ids.length
  const match = commonPrefix(ids, contextIds)
  return {
    id: entry.id,
    kind: entry.kind || 'entry',
    afterMessage: entry.after_message ?? null,
    len,
    ids,
    match,
    breakAt: match < len ? match : null,
    usable: reuseEnabled && match >= len && len < contextIds.length,
    relabeled: (entry.relabels || 0) > 0,
    relabelCount: entry.relabels || 0,
    nbytes: entry.nbytes ?? null,
    ageSeconds: entry.age_seconds ?? null,
    isNew: false,
    ghost: false,
  }
}

// Nice ruler step for a domain: 1/2/5 times a power of ten, roughly one tick
// per 70 px of a 900 px map (so about 13 ticks).
export function rulerStep(domain) {
  const target = domain / 13
  const power = 10 ** Math.floor(Math.log10(Math.max(1, target)))
  const norm = target / power
  const factor = norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1
  return factor * power
}

// Positions that were added in ``nextIds`` relative to ``prevIds`` (used for
// the small "+" ticks when comparing with the last turn).  LCS on the trimmed
// interior keeps this cheap for long, mostly-shared contexts.
export function addedPositions(prevIds, nextIds) {
  if (!prevIds || prevIds.length === 0) return []
  const diff = lcsRanges(prevIds, nextIds)
  const positions = []
  for (const range of diff.added) {
    for (let position = range.start; position < range.end; position += 1) {
      positions.push(position)
    }
  }
  return positions
}

// Caption parts.  One array of segments: { kind: 'text' | 'ok' | 'info' |
// 'bad' | 'warn' | 'code', text }.  The component renders them with the
// same colour grammar as the map.
function captionParts({ total, reused, chosen, computed, entries, reuseEnabled, historical, turnNumber }) {
  if (total === 0) {
    return [{ kind: 'text', text: 'Type a message to see its tokens on the tape.' }]
  }
  if (!reuseEnabled) {
    return [
      { kind: 'bad', text: 'reuse is off' },
      { kind: 'text', text: ` — the bars still show what would match, but all ${total} tokens will be computed.` },
    ]
  }
  if (reused > 0 && chosen) {
    const parts = [
      historical ? { kind: 'text', text: `Turn ${turnNumber}: ` } : null,
      { kind: 'ok', text: `restores ${reused} token${reused === 1 ? '' : 's'} from #${chosen.id}` },
      { kind: 'text', text: `, computes ${computed}` },
      chosen.relabeled
        ? [{ kind: 'warn', text: ' · label ≠ state: the saved computation is from the original tokens' }]
        : [],
    ].flat().filter(Boolean)
    return parts
  }
  const broken = entries
    .filter((entry) => entry.breakAt != null && !entry.ghost)
    .sort((a, b) => b.breakAt - a.breakAt)[0]
  if (broken && entries.length > 0) {
    return [
      historical ? { kind: 'text', text: `Turn ${turnNumber}: ` } : null,
      { kind: 'text', text: 'no saved state matches. The longest, #' },
      { kind: 'code', text: String(broken.id) },
      { kind: 'text', text: `, matches to position ${broken.breakAt} then breaks. This model can't rewind, so ` },
      { kind: 'info', text: `all ${total} tokens are computed` },
      { kind: 'text', text: '.' },
    ].filter(Boolean)
  }
  if (entries.length === 0) {
    return [
      historical ? { kind: 'text', text: `Turn ${turnNumber}: ` } : null,
      { kind: 'text', text: 'the cache is empty, so ' },
      { kind: 'info', text: `all ${total} tokens are computed` },
      { kind: 'text', text: '. The template wraps your message and adds an assistant opener ending in ⟨think⟩.' },
    ].filter(Boolean)
  }
  return [
    historical ? { kind: 'text', text: `Turn ${turnNumber}: ` } : null,
    { kind: 'info', text: `all ${total} tokens are computed` },
    { kind: 'text', text: ' — no entry is a full prefix of this request.' },
  ].filter(Boolean)
}

// Build the whole view model.  ``report`` is a prompt report (live preview,
// the submitted generation prompt, or a historical turn's context).  All
// other inputs are optional so the empty page renders too.
export function buildMapModel({
  report = null,
  entries = [],
  generated = [],        // live/recorded output tokens [{channel, piece, id}]
  lookup = null,         // cache.lookup event / generation.cache_lookup / turn snapshot lookup
  prefill = null,        // generation.stats.prefill / turn.prefill
  progress = null,       // latest prefill.chunk event {processed_tokens, total_tokens}
  running = false,
  reuseEnabled = true,
  historical = false,
  turnNumber = null,
  prevEntries = null,    // the previous snapshot's entries (NEW / ghost bars)
  prevIds = null,        // previous context token ids ("+ ticks" compare)
  comparePrev = false,
  historyCounts = null,  // {thinking, answer} for a finished historical turn
  savedSummary = null,   // {added, evicted, checkpoints} cache changes during this run
  cacheKnown = true,     // false while the live entry token lists are still loading
}) {
  const contextIds = report?.token_ids || []
  const pieces = report?.token_pieces || []
  const spans = report?.spans || []
  const total = contextIds.length
  const prediction = report?.cache_prediction || null
  const comparisons = report?.cache_entries || []
  const warnings = []

  // ---- band split: always the server's numbers --------------------------
  let reused
  let computed
  let chosenEntryId
  let chosenRelabeled
  if (historical) {
    reused = lookup?.reused_tokens ?? prefill?.reused_tokens ?? 0
    computed = total - reused
    chosenEntryId = reused > 0 ? lookup?.entry_id ?? null : null
    chosenRelabeled = Boolean(lookup?.relabeled)
  } else if (running) {
    reused = lookup?.reused_tokens ?? 0
    const processTotal = progress?.total_tokens ?? prefill
      ? (progress?.total_tokens ?? total - reused)
      : total - reused
    computed = Math.min(progress?.processed_tokens ?? prefill?.processed_tokens ?? 0, processTotal)
    chosenEntryId = reused > 0 ? lookup?.entry_id ?? null : null
    chosenRelabeled = Boolean(lookup?.relabeled)
  } else {
    reused = prediction?.reused_tokens ?? 0
    computed = total - reused
    chosenEntryId = reused > 0 ? prediction?.entry_id ?? null : null
    chosenRelabeled = Boolean(prediction?.relabeled)
  }
  // With reuse off the band is all blue and nothing is restored, whatever a
  // stale prediction might claim.
  if (!reuseEnabled) {
    reused = 0
    computed = total
    chosenEntryId = null
    chosenRelabeled = false
  }

  // ---- tape ---------------------------------------------------------------
  const tape = spans.length > 0
    ? spans.map((span) => ({
        start: span.start,
        end: span.end,
        role: span.kind === 'opener' ? 'template' : span.role || 'template',
        messageIndex: span.kind === 'opener' ? null : span.index ?? null,
      }))
    : (total > 0 ? [{ start: 0, end: total, role: 'user', messageIndex: null }] : [])

  // ---- generated tail -----------------------------------------------------
  const generatedTail = generated.map((token, index) => ({
    position: total + index,
    channel: token.channel === 'thinking' ? 'thinking' : 'answer',
    piece: token.piece ?? token.text ?? '',
    id: token.id ?? null,
  }))
  const thinkCount = generatedTail.filter((token) => token.channel === 'thinking').length
  const answerCount = generatedTail.length - thinkCount

  // ---- cache entries ------------------------------------------------------
  const serverById = new Map(comparisons.map((comparison) => [comparison.id, comparison]))
  const assessed = entries.map((entry) => {
    const row = assessEntry(entry, contextIds, reuseEnabled)
    const comparison = serverById.get(entry.id)
    const serverUsable = comparison ? comparison.reusable_tokens > 0 : null
    if (serverUsable !== null && reuseEnabled && serverUsable !== row.usable) {
      warnings.push(
        `mapModel: entry #${entry.id} disagrees with the server: client rule says ` +
        `${row.usable ? 'usable' : 'not usable'}, server says ${serverUsable ? 'usable' : 'not usable'}`,
      )
    }
    return {
      ...row,
      chosen: entry.id === chosenEntryId,
      serverUsable,
    }
  })
  const prevById = new Map((prevEntries || []).map((entry) => [entry.id, entry]))
  if (prevEntries && !historical) {
    for (const row of assessed) if (!prevById.has(row.id)) row.isNew = true
  }
  const ghosts = prevEntries && !historical
    ? prevEntries
        .filter((entry) => !assessed.some((row) => row.id === entry.id))
        .map((entry) => ({ ...assessEntry(entry, contextIds, reuseEnabled), ghost: true }))
    : []
  const rows = [...ghosts, ...assessed].sort((a, b) => b.len - a.len)

  // ---- "+" ticks (tokens added relative to the previous request) ----------
  const ticks = comparePrev && prevIds ? addedPositions(prevIds, contextIds) : []

  // ---- lens centre: the chosen entry's break (or its end), else the reuse
  // boundary -----------------------------------------------------------------
  const chosen = assessed.find((entry) => entry.chosen)
  const lensCenter = chosen ? chosen.breakAt ?? chosen.len : reused

  const domain = Math.max(
    10,
    total + generatedTail.length,
    ...rows.map((entry) => entry.len),
  )

  // ---- pipeline facts ------------------------------------------------------
  const messageCount = (report?.submitted_messages || []).length
  let phase
  if (historical) phase = 'turn'
  else if (running) {
    if (!lookup) phase = 'lookup'
    else if (progress && progress.processed_tokens < (progress.total_tokens ?? total - reused)) phase = 'compute'
    else if (generatedTail.length > 0) phase = 'generate'
    else phase = 'compute'
  } else phase = 'next'

  // ---- caption ---------------------------------------------------------------
  const caption = !cacheKnown && entries.length === 0 && total > 0
    ? [{ kind: 'text', text: 'reading the saved states on the server…' }]
    : captionParts({
        total,
        reused,
        chosen: chosen || null,
        computed,
        entries: rows,
        reuseEnabled,
        historical,
        turnNumber,
      })

  for (const warning of warnings) {
    // A disagreement between the client rule and the server is a bug: the
    // server's answer is shown, and the warning points at the entry.
    console.warn(warning)
  }

  const totalProcessed = historical
    ? prefill?.processed_tokens ?? computed
    : running
      ? progress?.processed_tokens ?? 0
      : computed

  return {
    contextIds,
    pieces,
    spans,
    total,
    tape,
    generatedTail,
    thinkCount,
    answerCount,
    reused,
    computed,
    totalProcessed,
    chosenEntryId,
    chosenRelabeled,
    prediction,
    lookup,
    rows,
    ghosts,
    ticks,
    lensCenter,
    domain,
    rulerStep: rulerStep(domain),
    caption,
    phase,
    messageCount,
    savedSummary,
    historyCounts,
    historyThinking: historical ? (historyCounts?.thinking ?? 0) : 0,
    historyAnswer: historical ? (historyCounts?.answer ?? 0) : 0,
  }
}
