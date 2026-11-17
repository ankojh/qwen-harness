import { useEffect, useState } from 'react'
import { diffCacheEntry, formatError } from './api.js'
import { highlightSpecial } from './highlight.jsx'

// Side-by-side diff of one stored cache entry (left) against the previewed
// prompt (right).  The diff runs on token ids on the server and is shown as
// decoded text, one row per hunk.  Read-only and tokenizer-only: editing the
// messages re-previews the prompt, and the diff follows.

// Long blocks start collapsed to their head and tail: matching text sooner,
// since the interesting part is what differs.
const COLLAPSE_CHARS = 600
const COLLAPSE_CHANGE_CHARS = 2400

// Default to the entry that shares the longest prefix with the prompt: when
// it is not the one the run would use, that is usually the miss to explain.
function defaultEntryId(entries, prediction) {
  if (!entries.length) return null
  const predicted = prediction?.reused_tokens > 0 ? prediction.entry_id : null
  let best = entries[0]
  for (const entry of entries) {
    if (
      entry.common_prefix > best.common_prefix
      || (entry.common_prefix === best.common_prefix && entry.id === predicted)
    ) {
      best = entry
    }
  }
  return best.id
}

function entryLabel(entry, predictedId) {
  const parts = [`#${entry.id} ${entry.kind}`, `${entry.tokens} tokens`]
  if (entry.after_message != null) parts.push(`after message ${entry.after_message}`)
  parts.push(`common prefix ${entry.common_prefix}`)
  if (!entry.compatible) parts.push('incompatible')
  else if (entry.id === predictedId) parts.push('run will use')
  else if (entry.reusable_tokens === 0) parts.push('not usable')
  return parts.join(' · ')
}

function DiffText({ text, collapseAt, thinkStart, thinkEnd }) {
  if (!collapseAt || text.length <= collapseAt) {
    return highlightSpecial(text, thinkStart, thinkEnd)
  }
  const half = collapseAt / 2
  return (
    <>
      {highlightSpecial(text.slice(0, half), thinkStart, thinkEnd)}
      <span className="diff-elided"> … {text.length - collapseAt} characters … </span>
      {highlightSpecial(text.slice(-half), thinkStart, thinkEnd)}
    </>
  )
}

function DiffCell({ side, range, text, kind, collapseAt, thinkStart, thinkEnd }) {
  const empty = range[0] === range[1]
  return (
    <div className={`diff-cell ${empty ? 'empty' : kind}`} aria-label={side}>
      <span className="diff-pos">{empty ? '' : range[0]}</span>
      <span className="diff-text">
        {empty ? '' : (
          <DiffText text={text} collapseAt={collapseAt} thinkStart={thinkStart} thinkEnd={thinkEnd} />
        )}
      </span>
    </div>
  )
}

const CELL_KIND = {
  prefix: ['same', 'same'],
  equal: ['later', 'later'],
  delete: ['del', 'add'],
  insert: ['del', 'add'],
  replace: ['del', 'add'],
}

export default function CacheDiff({ preview, thinkStart, thinkEnd }) {
  const entries = preview.cache_entries || []
  const prediction = preview.cache_prediction
  const predictedId = prediction?.reused_tokens > 0 ? prediction.entry_id : null
  const [selectedId, setSelectedId] = useState(null)
  const [diff, setDiff] = useState(null)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState(() => new Set())

  const entryIds = entries.map((entry) => entry.id).join(',')
  const activeId = entries.some((entry) => entry.id === selectedId)
    ? selectedId
    : defaultEntryId(entries, prediction)

  useEffect(() => {
    setExpanded(new Set())
    if (activeId == null || !preview.token_ids?.length) {
      setDiff(null)
      setError(null)
      return undefined
    }
    let cancelled = false
    diffCacheEntry(activeId, preview.token_ids)
      .then((result) => {
        if (cancelled) return
        setDiff(result)
        setError(null)
      })
      .catch((err) => {
        if (cancelled) return
        setDiff(null)
        setError(formatError(err.detail || err))
      })
    return () => {
      cancelled = true
    }
    // The prompt hash stands in for token_ids; entryIds for the entry list.
  }, [activeId, preview.prompt_hash, entryIds])

  function toggle(index) {
    setExpanded((current) => {
      const next = new Set(current)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }

  return (
    <div className="inspector-block cache-diff">
      <h4>Context ↔ cache diff</h4>
      {entries.length === 0 ? (
        <p className="muted">No cache entries to compare yet. Generate once to store one.</p>
      ) : (
        <>
          <label className="cache-diff-picker">
            <span>Cache entry</span>
            <select value={activeId ?? ''} onChange={(event) => setSelectedId(Number(event.target.value))}>
              {entries.map((entry) => (
                <option key={entry.id} value={entry.id}>{entryLabel(entry, predictedId)}</option>
              ))}
            </select>
          </label>

          <div className="diff-legend">
            <span><i className="swatch same" /> identical prefix (reusable part)</span>
            <span><i className="swatch del" /> only in the cache entry</span>
            <span><i className="swatch add" /> only in the prompt (prefilled)</span>
            <span><i className="swatch later" /> matches again later, not reusable</span>
          </div>

          {error && <div className="note note-error">{error}</div>}

          {diff && (
            <>
              <div className="button-row wrap">
                <span className={`badge ${diff.reusable_tokens > 0 ? 'badge-ok' : 'badge-off'}`}>
                  {diff.reusable_tokens > 0
                    ? `reuses ${diff.reusable_tokens} of ${diff.prompt_tokens}`
                    : 'not usable'}
                </span>
                <span className="badge">{diff.prefill_tokens} to prefill</span>
                <span className="badge">common prefix {diff.common_prefix} of {diff.entry_tokens}</span>
                <span className="muted">{diff.reason}</span>
              </div>
              {diff.truncated && <div className="note note-warn">{diff.truncated}</div>}

              <div className="diff-grid">
                <div className="diff-head">cache entry #{diff.entry.id} · {diff.entry_tokens} tokens</div>
                <div className="diff-head">prompt · {diff.prompt_tokens} tokens</div>
                {diff.hunks.map((hunk, index) => {
                  const [leftKind, rightKind] = CELL_KIND[hunk.op]
                  const limit = hunk.op === 'prefix' || hunk.op === 'equal'
                    ? COLLAPSE_CHARS
                    : COLLAPSE_CHANGE_CHARS
                  const collapsible = Math.max(hunk.left_text.length, hunk.right_text.length) > limit
                  const collapseAt = collapsible && !expanded.has(index) ? limit : null
                  const tokens = Math.max(hunk.left[1] - hunk.left[0], hunk.right[1] - hunk.right[0])
                  return (
                    <div className="diff-row" key={index}>
                      <DiffCell side="cache entry" range={hunk.left} text={hunk.left_text} kind={leftKind} collapseAt={collapseAt} thinkStart={thinkStart} thinkEnd={thinkEnd} />
                      <DiffCell side="prompt" range={hunk.right} text={hunk.right_text} kind={rightKind} collapseAt={collapseAt} thinkStart={thinkStart} thinkEnd={thinkEnd} />
                      {collapsible && (
                        <button type="button" className="mini diff-toggle" onClick={() => toggle(index)}>
                          {collapseAt ? `show all ${tokens} tokens` : 'collapse'}
                        </button>
                      )}
                      {hunk.op === 'prefix' && (
                        <div className="diff-cut">
                          diverges at token {diff.common_prefix} · prefill starts at {diff.reusable_tokens}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
              <p className="note">
                Diffed on token ids, shown as decoded text; numbers are token positions.
                Each block is decoded separately, so a character split across a block edge
                can show as �. Tokenizer only, no model call. Entries change after a
                generation: preview again to refresh the list.
              </p>
            </>
          )}
        </>
      )}
    </div>
  )
}
