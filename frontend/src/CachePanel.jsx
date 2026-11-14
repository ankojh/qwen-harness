import { useEffect, useState } from 'react'
import { formatBytes } from './ConfigPanel.jsx'
import TokenChips, { chipText } from './TokenChips.jsx'
import { commonPrefix, diffCaches } from './turnDiffs.js'
import { getCacheEntry } from './api.js'

// The server-side prompt cache.  Three views: Overview (the summary as
// before; clicking an entry opens it), Inside an entry (its tokens decoded,
// the match against the viewed context, the per-layer state and a bytes
// bar), and Changes (what this turn added, reused, evicted or made
// unusable versus the previous turn).

function age(seconds) {
  if (seconds == null) return '—'
  if (seconds < 60) return `${Math.round(seconds)}s ago`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`
  return `${Math.round(seconds / 3600)}h ago`
}

const KIND_LABEL = {
  checkpoint: 'checkpoint',
  'after-generation': 'after reply',
}

function CacheTabs({ view, onChange }) {
  return (
    <div className="tabs" role="tablist">
      {[
        ['overview', 'Overview'],
        ['inside', 'Inside an entry'],
        ['changes', 'Changes'],
      ].map(([id, label]) => (
        <button
          key={id}
          type="button"
          role="tab"
          className={view === id ? 'active' : ''}
          onClick={() => onChange(id)}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

function InsideView({ detail, contextIds, contextPieces, onBack }) {
  const entryIds = detail.token_ids || []
  const common = commonPrefix(entryIds, contextIds)
  let matchNote
  let matchOk = false
  if (common >= entryIds.length) {
    matchOk = true
    matchNote = `Matches the first ${entryIds.length} tokens of the selected request.`
  } else if (common >= contextIds.length) {
    matchNote = `The request ends at token ${contextIds.length}; this entry continues past it.`
  } else {
    const entryToken = detail.tokens[common]
    const requestToken = contextPieces[common]
    matchNote = `Diverges at token ${common}: entry has ⟨${chipText(entryToken)}⟩, request has ⟨${chipText(requestToken)}⟩.`
  }

  const totals = detail.totals || {}
  const recurrent = totals.recurrent_bytes || 0
  const kvUsed = totals.kv_used_bytes || 0
  const kvAllocated = totals.kv_allocated_bytes || 0
  const kvEmpty = Math.max(0, kvAllocated - kvUsed)
  const entryBytes = recurrent + kvAllocated
  const pct = (value) => (entryBytes > 0 ? `${(value / entryBytes) * 100}%` : '0%')
  const marks = []
  if (common > 0) marks.push({ start: 0, end: common, kind: 'cached' })
  if (!matchOk) marks.push({ start: common, end: Math.min(common + 1, entryIds.length), kind: 'divergence' })

  return (
    <div className="cache-inside">
      <div className="row-between">
        <span>
          <strong>#{detail.id}</strong> {KIND_LABEL[detail.kind] || detail.kind}
          {detail.after_message ? (
            <span className="muted"> · after message {detail.after_message}</span>
          ) : null}
        </span>
        <button type="button" className="mini" onClick={onBack}>← all entries</button>
      </div>
      <p className="muted small">{entryIds.length} tokens · {formatBytes(entryBytes)} accounted</p>

      <p className={`note ${matchOk ? 'note-ok' : 'note-warn'}`}>{matchNote}</p>

      <TokenChips tokens={detail.tokens} spans={detail.spans} marks={marks} />

      <div className="bytes-bar" title="Recurrent state · KV in use · KV allocated but empty">
        <span className="recurrent" style={{ width: pct(recurrent) }} />
        <span className="kv-used" style={{ width: pct(kvUsed) }} />
        <span className="kv-empty" style={{ width: pct(kvEmpty) }} />
      </div>
      <div className="budget-legend muted">
        <span><span className="swatch recurrent" />recurrent {formatBytes(recurrent)} (fixed)</span>
        <span><span className="swatch kv-used" />KV in use {formatBytes(kvUsed)}</span>
        <span><span className="swatch kv-empty" />KV allocated but empty {formatBytes(kvEmpty)}</span>
        {totals.kv_bytes_per_token_per_layer != null && (
          <span>KV {totals.kv_bytes_per_token_per_layer} B per token per layer</span>
        )}
      </div>

      <div className="table-scroll tall">
        <table className="data-table">
          <thead>
            <tr><th>Layer</th><th>Class</th><th>Kind</th><th>Arrays (shape, dtype)</th><th>Used / allocated tokens</th><th>Bytes</th></tr>
          </thead>
          <tbody>
            {(detail.layers || []).map((layer) => (
              <tr key={layer.index}>
                <td>{layer.index}</td>
                <td className="mono">{layer.class}</td>
                <td>{layer.kind === 'linear_attention' ? 'linear attn' : 'full attn'}</td>
                <td className="mono small">
                  {layer.arrays.map((array) => (
                    <div key={array.name}>
                      {array.name} [{array.shape.join(', ')}] {array.dtype}
                    </div>
                  ))}
                </td>
                <td>
                  {layer.used_tokens == null
                    ? 'n/a (fixed)'
                    : `${layer.used_tokens} / ${layer.allocated_tokens}`}
                </td>
                <td>
                  {layer.used_tokens == null
                    ? formatBytes(layer.allocated_bytes)
                    : `${formatBytes(layer.used_bytes)} / ${formatBytes(layer.allocated_bytes)}`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function ChangesView({ currentCache, previousCache, contextIds, lookup }) {
  if (!previousCache) {
    return <p className="muted">First turn — nothing to compare.</p>
  }
  const diff = diffCaches(previousCache, currentCache, contextIds)
  const signed = (bytes) => `${bytes >= 0 ? '+' : ''}${formatBytes(bytes)}`
  const entryLabel = (entry) =>
    `#${entry.id} ${KIND_LABEL[entry.kind] || entry.kind}${entry.after_message ? ` after message ${entry.after_message}` : ''} · ${(entry.token_ids || []).length} tokens`
  return (
    <div className="change-list">
      <div className="change-row">
        <strong>Added</strong>
        {diff.added.length === 0 ? (
          <div className="muted">none</div>
        ) : (
          diff.added.map((entry) => (
            <div key={entry.id}><span className="badge badge-ok">+ #{entry.id}</span> {entryLabel(entry)}</div>
          ))
        )}
      </div>
      <div className="change-row">
        <strong>Evicted</strong>
        {diff.evicted.length === 0 ? (
          <div className="muted">none</div>
        ) : (
          diff.evicted.map((entry) => (
            <div key={entry.id}><span className="badge badge-off">− #{entry.id}</span> {entryLabel(entry)}</div>
          ))
        )}
      </div>
      <div className="change-row">
        <strong>Reused</strong>
        {lookup?.reused_tokens ? (
          <div>
            <span className="badge">#{lookup.entry_id}</span>{' '}
            {lookup.reused_tokens} tokens · {lookup.reason}
          </div>
        ) : (
          <div className="muted">nothing was reused{lookup ? ` (${lookup.reason})` : ''}</div>
        )}
      </div>
      <div className="change-row">
        <strong>Unusable for this conversation</strong>
        {diff.unusable.length === 0 ? (
          <div className="muted">none — every entry is a full prefix of the request</div>
        ) : (
          diff.unusable.map(({ entry, divergence, extendsBeyond }) => (
            <div key={entry.id}>
              <span className="badge badge-warn">#{entry.id}</span> {entryLabel(entry)} ·{' '}
              {extendsBeyond
                ? `the request ends at token ${divergence}; this entry continues past it`
                : `diverges at token ${divergence}`}
            </div>
          ))
        )}
      </div>
      <div className="change-row">
        <strong>Memory</strong>
        <div>
          {formatBytes(previousCache.total_bytes ?? 0)} → {formatBytes(currentCache.total_bytes ?? 0)} ·{' '}
          {signed(diff.memoryChange)}
        </div>
      </div>
    </div>
  )
}

export default function CachePanel({ state, preview, generation, busy, onClearCache, viewTurn, prevTurn }) {
  const [view, setView] = useState('overview')
  const [insideId, setInsideId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [detailError, setDetailError] = useState(null)

  const historical = Boolean(viewTurn)
  const liveEntries = state.cache?.entries || []
  const entries = historical ? viewTurn.cache?.entries || [] : liveEntries
  const currentCache = {
    entries,
    total_bytes: historical ? viewTurn.cache?.total_bytes ?? 0 : state.cache?.total_bytes ?? 0,
  }
  const contextIds = historical ? viewTurn.context?.token_ids || [] : preview?.token_ids || []
  const contextPieces = historical
    ? viewTurn.context?.token_pieces || []
    : preview?.token_pieces || []
  const prediction = preview?.cache_prediction
  const lastLookup = historical ? viewTurn.cache?.last_lookup : state.cache?.last_lookup
  const lastPrefill = generation?.stats?.prefill
  const longest = Math.max(...entries.map((entry) => entry.tokens || 0), 1)
  const planned = (preview?.checkpoints || []).filter(
    (entry) => entry.ok && entry.position > (prediction?.reused_tokens || 0),
  )
  const lookupForChanges = historical
    ? viewTurn.cache?.last_lookup
    : state.cache?.last_lookup
  const previousCache = prevTurn?.cache || null

  // Inside view: fetch the entry details on demand.
  useEffect(() => {
    if (view !== 'inside' || insideId == null) {
      setDetail(null)
      setDetailError(null)
      return undefined
    }
    let cancelled = false
    getCacheEntry(insideId)
      .then((data) => {
        if (!cancelled) {
          setDetail(data)
          setDetailError(null)
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setDetail(null)
          setDetailError(error.detail || String(error))
        }
      })
    return () => {
      cancelled = true
    }
  }, [view, insideId])

  if (historical && !viewTurn.cache) {
    return <p className="muted">No cache snapshot was recorded for this turn.</p>
  }

  return (
    <div className="cache-panel">
      <div className="ctx-headline">
        <span className="big-num">{entries.length}</span>
        <span className="muted">
          {entries.length === 1 ? 'entry' : 'entries'} · {formatBytes(currentCache.total_bytes)}
          {historical ? ' as of this turn' : ` of ${formatBytes(state.cache?.limit_bytes)}`} ·{' '}
          {state.cache?.evictions_total ?? 0} evicted
        </span>
      </div>
      <div className="budget-bar" title="Prompt cache memory">
        <div
          className="seg cached"
          style={{ width: `${Math.min(100, ((state.cache?.utilization || 0) * 100).toFixed(1))}%` }}
        />
        <div className="seg remaining" style={{ flex: 1 }} />
      </div>

      <CacheTabs view={view} onChange={(next) => setView(next)} />

      {view === 'inside' && (
        detailError ? (
          <p className="note note-error">
            {String(detailError)} — the entry may have been evicted.
          </p>
        ) : !detail ? (
          <p className="muted">Reading entry #{insideId}…</p>
        ) : (
          <InsideView
            detail={detail}
            contextIds={contextIds}
            contextPieces={contextPieces}
            onBack={() => setView('overview')}
          />
        )
      )}

      {view === 'changes' && (
        <ChangesView
          currentCache={currentCache}
          previousCache={previousCache}
          contextIds={contextIds}
          lookup={lookupForChanges}
        />
      )}

      {view === 'overview' && (
        <>
          <div className="cache-calls">
            {historical ? (
              <div className={`note ${lastLookup?.reused_tokens ? 'note-ok' : 'note-warn'}`}>
                <strong>This request: </strong>
                {lastLookup?.reused_tokens
                  ? `reused ${lastLookup.reused_tokens} tokens from #${lastLookup.entry_id}`
                  : 'reused nothing'}
                <div className="small">{lastLookup?.reason || 'no lookup recorded'}</div>
              </div>
            ) : (
              <>
                <div className={`note ${prediction?.reused_tokens ? 'note-ok' : ''}`}>
                  <strong>Next request: </strong>
                  {!preview
                    ? 'type a message to see a prediction'
                    : prediction?.reused_tokens
                      ? `reuses ${prediction.reused_tokens} of ${preview.budget?.input_tokens} tokens from #${prediction.entry_id}`
                      : `nothing reusable (${prediction?.reason || 'no prediction'})`}
                  {preview && (
                    <div className="small">
                      will save: {planned.length > 0 ? `a checkpoint at token ${planned.map((entry) => entry.position).join(', ')} and ` : ''}
                      the full exchange after the reply
                    </div>
                  )}
                </div>
                {lastLookup && (
                  <div className={`note ${lastLookup.reused_tokens ? 'note-ok' : 'note-warn'}`}>
                    <strong>Last request: </strong>
                    {lastLookup.reused_tokens
                      ? `reused ${lastLookup.reused_tokens} tokens from #${lastLookup.entry_id}`
                      : 'reused nothing'}
                    {lastPrefill ? `, read ${lastPrefill.processed_tokens} in ${lastPrefill.seconds}s` : ''}
                    <div className="small">{lastLookup.reason}</div>
                  </div>
                )}
              </>
            )}
          </div>

          {entries.length === 0 ? (
            <p className="muted">
              Empty. Each reply stores the conversation so far so a later request can
              skip re-reading it.
            </p>
          ) : (
            <ul className="cache-entries">
              {[...entries].reverse().map((entry) => {
                const next = !historical && prediction?.reused_tokens > 0 && prediction.entry_id === entry.id
                const last = !historical && lastLookup?.reused_tokens > 0 && lastLookup.entry_id === entry.id
                return (
                  <li key={entry.id} className={`cache-entry ${next ? 'is-next' : ''}`}>
                    <button
                      type="button"
                      className="cache-entry-open"
                      onClick={() => {
                        setInsideId(entry.id)
                        setView('inside')
                      }}
                    >
                      <span className="row-between">
                        <span>
                          <strong>#{entry.id}</strong> {KIND_LABEL[entry.kind] || entry.kind}
                          {entry.after_message ? <span className="muted"> · after message {entry.after_message}</span> : null}
                        </span>
                        <span className="ctx-tags">
                          {next && <span className="badge badge-ok">next request uses this</span>}
                          {last && <span className="badge">used last</span>}
                        </span>
                      </span>
                      <span className="entry-bar">
                        <span style={{ width: `${((entry.tokens || 0) / longest) * 100}%` }} />
                      </span>
                      <span className="muted small">
                        {entry.tokens ?? '?'} tokens · {formatBytes(entry.nbytes)} · {age(entry.age_seconds)}
                      </span>
                    </button>
                  </li>
                )
              })}
            </ul>
          )}

          <div className="row-between">
            <span className="muted small">
              A request reuses an entry only if it starts with exactly that entry's tokens.
            </span>
            <button type="button" className="mini" disabled={busy || entries.length === 0} onClick={onClearCache}>
              Clear cache
            </button>
          </div>
        </>
      )}
    </div>
  )
}
