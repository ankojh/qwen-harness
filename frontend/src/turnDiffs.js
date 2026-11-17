// Pure helpers that compare two turn snapshots.  Used by the Changes views
// of the Context and Cache panels.

// How many leading token ids two sequences share.
export function commonPrefix(a, b) {
  const limit = Math.min(a.length, b.length)
  let index = 0
  while (index < limit && a[index] === b[index]) index += 1
  return index
}

// Contiguous ranges where the flag is true.
function rangesOf(flags) {
  const ranges = []
  let start = null
  for (let index = 0; index <= flags.length; index += 1) {
    const on = index < flags.length && flags[index]
    if (on && start === null) start = index
    if (!on && start !== null) {
      ranges.push({ start, end: index })
      start = null
    }
  }
  return ranges
}

// Token-level diff of one span: which positions are removed (prev) / added
// (next).  A simple LCS over the token ids; spans larger than ~2M cells are
// marked fully changed instead of diffed.
export function lcsRanges(prevIds, nextIds) {
  // Trim equal edges first: identical/mostly unchanged long prompts should
  // remain exact without allocating a quadratic table for the whole context.
  const prefix = commonPrefix(prevIds, nextIds)
  let suffix = 0
  while (suffix < Math.min(prevIds.length, nextIds.length) - prefix &&
         prevIds[prevIds.length - 1 - suffix] === nextIds[nextIds.length - 1 - suffix]) suffix += 1
  const prev = prevIds.slice(prefix, prevIds.length - suffix)
  const next = nextIds.slice(prefix, nextIds.length - suffix)
  const n = prev.length
  const m = next.length
  const shift = (ranges) => ranges.map((r) => ({ start: r.start + prefix, end: r.end + prefix }))
  if (n === 0) return { removed: [], added: shift(rangesOf(next.map(() => true))), approximate: false }
  if (m === 0) return { removed: shift(rangesOf(prev.map(() => true))), added: [], approximate: false }
  if (n * m > 2_000_000) {
    return { removed: [{ start: prefix, end: prefix + n }], added: [{ start: prefix, end: prefix + m }], approximate: true }
  }
  const dp = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1))
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      dp[i][j] =
        prev[i] === next[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
    }
  }
  const keepPrev = new Array(n).fill(false)
  const keepNext = new Array(m).fill(false)
  let i = 0
  let j = 0
  while (i < n && j < m) {
    if (prev[i] === next[j]) {
      keepPrev[i] = true
      keepNext[j] = true
      i += 1
      j += 1
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      i += 1
    } else {
      j += 1
    }
  }
  return {
    removed: shift(rangesOf(keepPrev.map((keep) => !keep))),
    added: shift(rangesOf(keepNext.map((keep) => !keep))),
    approximate: false,
  }
}

// The submitted message a span belongs to (null for the assistant opener).
export function messageOf(report, span) {
  if (!span || span.kind === 'opener') return null
  return (
    (report?.submitted_messages || []).find((message) => message.index === span.index) ||
    null
  )
}

function spanKey(span) {
  return span.kind === 'opener' ? 'opener' : `${span.index}:${span.role}`
}

// Context: turn N vs turn N-1.
//   rows:      removed spans, then next-order rows (added / unchanged /
//              re-rendered / changed), each with token diff ranges when
//              they apply.
//   summary:   common prefix length, net token change, and the thinking
//              tokens the previous turn generated that are not carried.
export function diffContexts(prevReport, nextReport, prevThinking = 0) {
  const prevIds = prevReport?.token_ids || []
  const nextIds = nextReport?.token_ids || []
  const prevSpans = prevReport?.spans || []
  const nextSpans = nextReport?.spans || []

  const prevByKey = new Map(prevSpans.map((span) => [spanKey(span), span]))
  const nextKeys = new Set(nextSpans.map(spanKey))
  const removed = prevSpans
    .filter((span) => !nextKeys.has(spanKey(span)))
    .map((span) => ({ kind: 'removed', prev: span }))

  const rows = nextSpans.map((nextSpan) => {
    const prevSpan = prevByKey.get(spanKey(nextSpan)) || null
    if (!prevSpan) return { kind: 'added', next: nextSpan }
    const prevSlice = prevIds.slice(prevSpan.start, prevSpan.end)
    const nextSlice = nextIds.slice(nextSpan.start, nextSpan.end)
    const sameIds =
      prevSlice.length === nextSlice.length &&
      prevSlice.every((id, index) => id === nextSlice[index])
    if (sameIds) return { kind: 'unchanged', prev: prevSpan, next: nextSpan }
    const prevMessage = messageOf(prevReport, prevSpan)
    const nextMessage = messageOf(nextReport, nextSpan)
    const sameContent =
      prevMessage && nextMessage && prevMessage.content === nextMessage.content
    return {
      kind: sameContent ? 're-rendered' : 'changed',
      prev: prevSpan,
      next: nextSpan,
      diff: lcsRanges(prevSlice, nextSlice),
    }
  })

  return {
    commonPrefix: commonPrefix(prevIds, nextIds),
    prevTokens: prevIds.length,
    nextTokens: nextIds.length,
    netChange: nextIds.length - prevIds.length,
    thinkingNotCarried: prevThinking || 0,
    rows: [...removed, ...rows],
  }
}

// Cache: after turn N vs after turn N-1.
//   added / evicted: entry ids present in only one snapshot.
//   unusable:        entries whose token ids are not a full prefix of the
//                    selected context (the cache can't be trimmed, so they
//                    cannot contribute).  Each row carries the divergence
//                    position.
//   memoryChange:    accounted bytes after N minus after N-1.
export function diffCaches(prevSnapshot, nextSnapshot, contextIds = []) {
  const prevEntries = prevSnapshot?.entries || []
  const nextEntries = nextSnapshot?.entries || []
  const prevById = new Map(prevEntries.map((entry) => [entry.id, entry]))
  const nextById = new Map(nextEntries.map((entry) => [entry.id, entry]))
  const added = nextEntries.filter((entry) => !prevById.has(entry.id))
  const evicted = prevEntries.filter((entry) => !nextById.has(entry.id))
  const modified = nextEntries.filter((entry) => prevById.has(entry.id)).map((entry) => {
    const before = prevById.get(entry.id).token_ids || []
    const after = entry.token_ids || []
    const changes = []
    for (let position = 0; position < Math.max(before.length, after.length); position += 1) {
      if (before[position] !== after[position]) changes.push({ position, before: before[position], after: after[position] })
    }
    return { entry, changes }
  }).filter((row) => row.changes.length)
  const unusable = nextEntries
    .map((entry) => {
      const ids = entry.token_ids || []
      const common = commonPrefix(ids, contextIds)
      if (common >= ids.length && ids.length < contextIds.length) return null // leave one input token to process
      return {
        entry,
        divergence: common,
        extendsBeyond: common >= contextIds.length,
      }
    })
    .filter(Boolean)
  return {
    added,
    evicted,
    modified,
    unusable,
    memoryChange: (nextSnapshot?.total_bytes ?? 0) - (prevSnapshot?.total_bytes ?? 0),
  }
}

// Consequences of staged edits: which cache entries are a full prefix of the
// current request but stop being one for the edited request.  Each row
// carries the entry and its divergence position against the edited context.
export function stopsMatching(entries, beforeIds, afterIds) {
  return entries
    .filter((entry) => {
      const ids = entry.token_ids || []
      return (
        commonPrefix(ids, beforeIds) >= ids.length &&
        commonPrefix(ids, afterIds) < ids.length
      )
    })
    .map((entry) => {
      const ids = entry.token_ids || []
      const divergence = commonPrefix(ids, afterIds)
      return {
        entry,
        divergence,
        extendsBeyond: divergence >= afterIds.length,
      }
    })
}
