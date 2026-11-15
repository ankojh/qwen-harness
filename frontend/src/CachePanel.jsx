import { useEffect, useState } from 'react'
import { formatBytes } from './ConfigPanel.jsx'
import CacheMatch, { ReuseExplainer } from './CacheMatch.jsx'
import TokenChips, { chipText } from './TokenChips.jsx'
import { commonPrefix, diffCaches } from './turnDiffs.js'
import { deleteCacheEntry, getCacheEntry, getCacheTokens, relabelEntry, restoreEntry } from './api.js'

// The server-side prompt cache.  Three views: Overview (the summary as
// before plus the quick switches; clicking an entry opens it), Inside an
// entry (its tokens decoded, the match against the viewed context, the
// per-layer state, a bytes bar, and the relabel/restore tools), and Changes
// (what this turn added, reused, evicted or made unusable versus the
// previous turn, plus the edits applied before it).

function age(seconds) {
  if (seconds == null) return '—'
  if (seconds < 60) return `${Math.round(seconds)}s ago`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`
  return `${Math.round(seconds / 3600)}h ago`
}

const KIND_LABEL = {
  checkpoint: 'checkpoint',
  manual: 'saved by you',
  'after-generation': 'after reply',
}

const CUT_SHORTER_HELP =
  'This model\'s cache cannot be trimmed: 24 of its 32 layers keep a running ' +
  'state that cannot be rewound, so an entry cannot be cut shorter.'

function CacheTabs({ view, onChange }) {
  return (
    <div className="tabs" role="tablist">
      {[
        ['overview', 'Current'],
        ['inside', 'Entry tools'],
        ['changes', 'History diff'],
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

function CacheSwitches({ config, busy, historical, onConfigChange }) {
  const reuse = config?.cache_reuse !== false
  const checkpoints = config?.cache_checkpoints || 'last_message'
  return (
    <div className="cache-switches">
      <label className="check" title="Whether requests look up and store prompt caches">
        <input
          type="checkbox"
          checked={reuse}
          disabled={busy || historical}
          onChange={(event) => onConfigChange({ cache_reuse: event.target.checked })}
        />
        reuse cache
      </label>
      <label className="cache-switch-select" title="Where prefill stores prompt-cache snapshots">
        <span className="muted small">checkpoints</span>
        <select
          value={checkpoints}
          disabled={busy || historical}
          onChange={(event) => onConfigChange({ cache_checkpoints: event.target.value })}
        >
          <option value="off">off</option>
          <option value="last_message">last message</option>
          <option value="every_message">every message</option>
        </select>
      </label>
      {historical && <span className="muted small">read-only for a past turn</span>}
    </div>
  )
}

function RelabelForm({ position, text, setText, error, busy, onSubmit, onCancel }) {
  return (
    <div className="relabel-form">
      <span className="muted small">relabel token {position}:</span>
      <input
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="exactly one token, e.g. ␣Bob → type “ Bob”"
        autoFocus
        onKeyDown={(event) => {
          if (event.key === 'Enter') onSubmit()
          if (event.key === 'Escape') onCancel()
        }}
      />
      <button type="button" className="mini primary" disabled={busy} onClick={onSubmit}>
        relabel
      </button>
      <button type="button" className="mini" disabled={busy} onClick={onCancel}>
        cancel
      </button>
      {error && <span className="note note-error">{error}</span>}
      <span className="muted small">
        This changes token labels only. The computed model state stays as it was.
        One token only; the count never changes. States later built from this entry may carry the effect even after Restore.
      </span>
    </div>
  )
}

function InsideView({
  detail,
  contextIds,
  contextPieces,
  relabelPosition,
  relabelText,
  setRelabelText,
  relabelError,
  relabelBusy,
  onOpenRelabel,
  onSubmitRelabel,
  onCancelRelabel,
  onRestore,
  restoreBusy,
  historical,
  busy,
  onBack,
}) {
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
  const tampered = (detail.relabels || []).map((item) => ({
    position: item.position,
    original_id: item.original_id,
    original_piece: item.original_piece,
  }))

  return (
    <div className="cache-inside">
      <div className="row-between">
        <span>
          <strong>#{detail.id}</strong> {KIND_LABEL[detail.kind] || detail.kind}
          {detail.after_message ? (
            <span className="muted"> · after message {detail.after_message}</span>
          ) : null}
          {tampered.length > 0 && (
            <span className="badge badge-off" title="tokens were relabelled; the stored state no longer matches the claimed tokens">
              tampered
            </span>
          )}
        </span>
        <button type="button" className="mini" onClick={onBack}>← all entries</button>
      </div>
      <p className="muted small">{entryIds.length} tokens · {formatBytes(entryBytes)} accounted</p>

      {tampered.length > 0 && !historical && (
        <div className="row-between">
          <span className="muted small">
            {tampered.length} token{tampered.length > 1 ? 's' : ''} relabelled:
            {tampered.map((item) => (
              <span key={item.position} className="mono">
                {' '}pos {item.position}: ⟨{chipText(detail.tokens[item.position])}⟩ was ⟨{JSON.stringify(item.original_piece)}⟩
              </span>
            ))}
          </span>
          <button
            type="button"
            className="mini"
            disabled={busy || restoreBusy}
            title="put the original token ids back and clear the relabels"
            onClick={onRestore}
          >
            Restore
          </button>
        </div>
      )}

      <p className={`note ${matchOk ? 'note-ok' : 'note-warn'}`}>{matchNote}</p>

      {relabelPosition != null && (
        <RelabelForm
          position={relabelPosition}
          text={relabelText}
          setText={setRelabelText}
          error={relabelError}
          busy={relabelBusy}
          onSubmit={onSubmitRelabel}
          onCancel={onCancelRelabel}
        />
      )}

      <TokenChips
        tokens={detail.tokens}
        spans={detail.spans}
        marks={marks}
        tampered={tampered}
        onTokenClick={historical ? undefined : onOpenRelabel}
      />

      <div className="row-between">
        <button
          type="button"
          className="mini"
          disabled
          title={CUT_SHORTER_HELP}
        >
          cut entry shorter
        </button>
        <span className="muted small">{CUT_SHORTER_HELP}</span>
      </div>

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

function EditLog({ edits, historical }) {
  const label = historical ? 'Before this turn you' : 'So far, before the next turn you'
  return (
    <div className="change-row edit-log">
      <strong>{label}:</strong>
      {edits && edits.length > 0 ? (
        <ul>
          {edits.map((line, index) => (
            <li key={index}>{line}</li>
          ))}
        </ul>
      ) : (
        <div className="muted">nothing was edited</div>
      )}
    </div>
  )
}

function ChangesView({ currentCache, previousCache, contextIds, lookup, editLog, historical }) {
  return (
    <div className="change-list">
      <EditLog edits={editLog} historical={historical} />
      {!previousCache ? (
        <p className="muted">First turn — nothing to compare.</p>
      ) : (
        <>
          {(() => {
            const diff = diffCaches(previousCache, currentCache, contextIds)
            const signed = (bytes) => `${bytes >= 0 ? '+' : ''}${formatBytes(bytes)}`
            const entryLabel = (entry) =>
              `${KIND_LABEL[entry.kind] || entry.kind}${entry.after_message ? ` after message ${entry.after_message}` : ''} · ${(entry.token_ids || []).length} tokens`
            return (
              <>
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
                  <strong>Removed since snapshot</strong>
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
                      {lookup.relabeled && <span className="badge badge-off"> tampered entry</span>}
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
                  <strong>Changed token labels</strong>
                  {diff.modified.length === 0 ? <div className="muted">none</div> : diff.modified.map(({ entry, changes }) => <div key={entry.id}><b>#{entry.id}</b>{changes.map((c) => <div key={c.position}>position {c.position}: <del>ID {c.before ?? 'absent'}</del> → <ins>ID {c.after ?? 'absent'}</ins></div>)}</div>)}
                </div>
                <div className="change-row">
                  <strong>Memory</strong>
                  <div>
                    {formatBytes(previousCache.total_bytes ?? 0)} → {formatBytes(currentCache.total_bytes ?? 0)} ·{' '}
                    {signed(diff.memoryChange)}
                  </div>
                </div>
              </>
            )
          })()}
        </>
      )}
    </div>
  )
}

export default function CachePanel({
  state,
  preview,
  previewError,
  running,
  generation,
  busy,
  onClearCache,
  viewTurn,
  prevTurn,
  pendingEdits,
  onConfigChange,
  onCacheAction,
}) {
  const [view, setView] = useState('overview')
  const [liveTokenEntries, setLiveTokenEntries] = useState(null)
  const [tokenError, setTokenError] = useState(null)
  const [insideId, setInsideId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [detailError, setDetailError] = useState(null)
  const [detailVersion, setDetailVersion] = useState(0)
  const [relabelPosition, setRelabelPosition] = useState(null)
  const [relabelText, setRelabelText] = useState('')
  const [relabelError, setRelabelError] = useState(null)
  const [relabelBusy, setRelabelBusy] = useState(false)
  const [restoreBusy, setRestoreBusy] = useState(false)

  const historical = Boolean(viewTurn)
  const liveEntries = state.cache?.entries || []
  const entries = historical ? viewTurn.cache?.entries || [] : liveEntries
  const currentCache = {
    entries: historical ? entries : (liveTokenEntries || []).map((entry) => ({ ...liveEntries.find((e) => e.id === entry.id), ...entry })),
    total_bytes: historical ? viewTurn.cache?.total_bytes ?? 0 : state.cache?.total_bytes ?? 0,
  }
  const contextIds = historical ? viewTurn.context?.token_ids || [] : preview?.token_ids || []
  const contextPieces = historical
    ? viewTurn.context?.token_pieces || []
    : preview?.token_pieces || []
  const prediction = preview?.cache_prediction
  const lastLookup = historical ? viewTurn.cache?.last_lookup : generation?.cache_lookup || state.cache?.last_lookup
  const actualRequest = historical || Boolean(generation?.cache_lookup)
  const lastPrefill = historical ? viewTurn.prefill : generation?.cache_lookup ? generation?.stats?.prefill : null
  const lastInputTokens = historical ? viewTurn.context?.token_ids?.length : generation?.prompt?.token_ids?.length
  const lastReused = lastLookup?.reused_tokens || 0
  const finalInputProcessed = lastPrefill && lastInputTokens != null && lastPrefill.processed_tokens + lastReused < lastInputTokens && (historical ? viewTurn.finish?.phase !== 'prefill' : generation?.finish?.phase !== 'prefill') ? 1 : 0
  const processedInput = lastPrefill ? lastPrefill.processed_tokens + finalInputProcessed : null
  const longest = Math.max(...entries.map((entry) => entry.tokens || 0), 1)
  const planned = (preview?.checkpoints || []).filter(
    (entry) => entry.ok && entry.position > (prediction?.reused_tokens || 0),
  )
  const lookupForChanges = historical
    ? viewTurn.cache?.last_lookup
    : state.cache?.last_lookup
  const previousCache = prevTurn?.cache || null
  const editLog = historical ? viewTurn.edits || [] : pendingEdits || []

  useEffect(() => {
    if (historical || running) return undefined
    let cancelled = false
    setLiveTokenEntries(null)
    setTokenError(null)
    getCacheTokens().then((data) => { if (!cancelled) setLiveTokenEntries(data.entries || []) })
      .catch((error) => { if (!cancelled) setTokenError(error.message) })
    return () => { cancelled = true }
  }, [historical, view, state.cache, running])

  // Inside view: fetch the entry details on demand (and after every edit).
  useEffect(() => {
    if (view !== 'inside' || insideId == null || historical || running) {
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
  }, [view, insideId, detailVersion, historical, running, state.cache])

  function openEntry(id) {
    setInsideId(id)
    setRelabelPosition(null)
    setRelabelError(null)
    setView('inside')
  }

  function handleDelete(entry) {
    onCacheAction(
      () => deleteCacheEntry(entry.id),
      `deleted cache entry #${entry.id} (${KIND_LABEL[entry.kind] || entry.kind}${entry.after_message ? ` after message ${entry.after_message}` : ''}, ${entry.tokens} tokens)`,
    ).catch(() => {})
    if (insideId === entry.id) {
      setInsideId(null)
      setView('overview')
    }
  }

  function handleOpenRelabel(position) {
    const token = detail?.tokens?.[position]
    setRelabelPosition(position)
    setRelabelError(null)
    setRelabelText(token?.text ?? token?.piece ?? '')
  }

  function handleSubmitRelabel() {
    const position = relabelPosition
    if (position == null || !detail) return
    const token = detail.tokens[position]
    const before = (token?.text ?? token?.piece ?? '').trim()
    const after = relabelText.trim()
    setRelabelBusy(true)
    setRelabelError(null)
    onCacheAction(
      () => relabelEntry(insideId, position, relabelText),
      `relabelled token ${position} of entry #${insideId}: '${before}' → '${after}'`,
      { silent: true },
    )
      .then(() => {
        setRelabelPosition(null)
        setRelabelText('')
        setDetailVersion((previous) => previous + 1)
      })
      .catch((error) => {
        setRelabelError(error.detail || String(error))
      })
      .finally(() => setRelabelBusy(false))
  }

  function handleRestore() {
    if (!detail) return
    const count = (detail.relabels || []).length
    setRestoreBusy(true)
    onCacheAction(
      () => restoreEntry(insideId),
      `restored entry #${insideId}'s original tokens (${count} relabel${count === 1 ? '' : 's'} undone)`,
    )
      .then(() => {
        setRelabelPosition(null)
        setDetailVersion((previous) => previous + 1)
      })
      .catch(() => {})
      .finally(() => setRestoreBusy(false))
  }

  if (historical && !viewTurn.cache) {
    return <p className="muted">No cache snapshot was recorded for this turn.</p>
  }

  return (
    <div className="cache-panel">
      <CacheSwitches
        config={state.config}
        busy={busy}
        historical={historical}
        onConfigChange={onConfigChange}
      />
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
        historical ? <p className="note">Historical snapshots preserve entry token IDs and summaries, not editable model state or layer arrays. Return to Current state to inspect a live entry.</p> : running ? <p className="note">Entry tools wait until generation finishes. The cache may be changing.</p> : insideId == null ? <p className="muted">Select an entry in Current to inspect, delete or relabel its tokens.</p> : detailError ? (
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
            relabelPosition={relabelPosition}
            relabelText={relabelText}
            setRelabelText={setRelabelText}
            relabelError={relabelError}
            relabelBusy={relabelBusy}
            onOpenRelabel={handleOpenRelabel}
            onSubmitRelabel={handleSubmitRelabel}
            onCancelRelabel={() => {
              setRelabelPosition(null)
              setRelabelError(null)
            }}
            onRestore={handleRestore}
            restoreBusy={restoreBusy}
            historical={historical}
            busy={busy}
            onBack={() => setView('overview')}
          />
        )
      )}

      {view === 'changes' && (
        tokenError && !historical ? <p className="note note-error">{tokenError}</p> : !historical && !liveTokenEntries ? <p className="muted">Reading current cache labels…</p> : <ChangesView
          currentCache={currentCache}
          previousCache={previousCache}
          contextIds={contextIds}
          lookup={lookupForChanges}
          editLog={editLog}
          historical={historical}
        />
      )}

      {view === 'overview' && (
        <>
          {!historical && <ReuseExplainer />}
          <div className="cache-current-layout"><div className="saved-states">
          <h4>{historical ? 'Saved entries after this turn' : 'All saved states now'}</h4>
          {tokenError && <p className="note note-error">Cannot read current cache labels: {tokenError}</p>}
          {entries.length === 0 ? (
            <p className="muted">
              Empty. Each reply stores the conversation so far so a later request can
              skip re-reading it. Open Context › Edit to save a checkpoint without
              generating anything.
            </p>
          ) : (
            <ul className="cache-entries">
              {[...entries].reverse().map((entry) => {
                const next = !historical && prediction?.reused_tokens > 0 && prediction.entry_id === entry.id
                const last = !historical && lastLookup?.reused_tokens > 0 && lastLookup.entry_id === entry.id
                const tampered = (entry.relabels || 0) > 0
                const match = !historical ? preview?.cache_entries?.find((e) => e.id === entry.id) : null
                const labels = match || (historical ? entry : liveTokenEntries?.find((e) => e.id === entry.id))
                return (
                  <li key={entry.id} className={`cache-entry ${next ? 'is-next' : ''}`}>
                    <div className="cache-entry-row">
                      <button
                        type="button"
                        className="cache-entry-open"
                        onClick={() => openEntry(entry.id)}
                      >
                        <span className="row-between">
                          <span>
                            <strong>#{entry.id}</strong> {KIND_LABEL[entry.kind] || entry.kind}
                            {entry.after_message ? <span className="muted"> · after message {entry.after_message}</span> : null}
                          </span>
                          <span className="ctx-tags">
                            {tampered && <span className="badge badge-off">tampered</span>}
                            {next && <span className="badge badge-ok">next request uses this</span>}
                            {last && <span className="badge">used last</span>}
                          </span>
                        </span>
                        {match && <>
                          <span className={`entry-match-label ${match.reusable_tokens ? 'reusable' : ''}`}>{match.shared_prefix} leading labels match · {match.reusable_tokens} tokens usable{state.config?.cache_reuse === false ? ' · reuse disabled' : ''}</span>
                          <span className="small">{match.reason}</span>
                        </>}
                        {labels?.text_excerpt && <span className="cache-excerpt">{labels.text_excerpt}{labels.excerpt_truncated ? '…' : ''}</span>}
                        <span className="entry-bar">
                          <span style={{ width: `${((entry.tokens || 0) / longest) * 100}%` }} />
                        </span>
                        <span className="muted small">
                          {entry.tokens ?? '?'} tokens · {formatBytes(entry.nbytes)} · {age(entry.age_seconds)}
                        </span>
                      </button>
                      <button
                        type="button"
                        className="mini danger cache-entry-delete"
                        disabled={busy || historical}
                        aria-label={`Delete cache entry ${entry.id}`}
                        title={historical ? 'snapshots of past turns are read-only' : 'delete this entry'}
                        onClick={() => handleDelete(entry)}
                      >
                        ✕
                      </button>
                    </div>
                  </li>
                )
              })}
            </ul>
          )}

          <div className="row-between">
            <span className="muted small">
              Open an entry for exact tokens, memory layers, and relabel/restore tools.
            </span>
            <button type="button" className="mini" disabled={busy || historical || entries.length === 0} onClick={onClearCache}>
              Clear cache
            </button>
          </div>
          </div><div className="cache-current-comparison">
          {!historical && <>
            {running ? <p className="note" role="status">Generation in progress. Cache entries can change while the model works; the next-request comparison refreshes when it finishes.</p> : previewError ? <p className="note note-error">Cache comparison unavailable: {previewError}</p> : <CacheMatch report={preview} reuseEnabled={state.config?.cache_reuse !== false} running={running} />}
          </>}
          {lastLookup && <details className="last-request"><summary>{actualRequest ? 'Last actual request' : 'Most recent cache operation'} · reused {lastLookup.reused_tokens || 0} tokens{lastPrefill ? ` · processed ${processedInput} input tokens` : ''}</summary><p>{lastLookup.reason}</p>{lastPrefill && <p>Input processing before the final token: {lastPrefill.processed_tokens} tokens in {lastPrefill.seconds}s{finalInputProcessed ? '; one final input token is processed to start the answer' : ''}. This is a recorded result; the next-request comparison is a prediction.</p>}</details>}
          </div></div>
        </>
      )}
    </div>
  )
}
