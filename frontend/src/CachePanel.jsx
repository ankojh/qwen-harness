import { formatBytes } from './ConfigPanel.jsx'

// The server-side prompt cache: what it holds, what the next request would
// reuse from it, and what the last request actually reused.

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

export default function CachePanel({ state, preview, generation, busy, onClearCache }) {
  const cache = state.cache || { entries: [] }
  const entries = cache.entries || []
  const prediction = preview?.cache_prediction
  const lastLookup = cache.last_lookup
  const lastPrefill = generation?.stats?.prefill
  const longest = Math.max(...entries.map((entry) => entry.tokens), 1)
  const planned = (preview?.checkpoints || []).filter(
    (entry) => entry.ok && entry.position > (prediction?.reused_tokens || 0),
  )

  return (
    <div className="cache-panel">
      <div className="ctx-headline">
        <span className="big-num">{entries.length}</span>
        <span className="muted">
          {entries.length === 1 ? 'entry' : 'entries'} · {formatBytes(cache.accounted_bytes)} of{' '}
          {formatBytes(cache.limit_bytes)} · {cache.evictions_total ?? 0} evicted
        </span>
      </div>
      <div className="budget-bar" title="Prompt cache memory">
        <div className="seg cached" style={{ width: `${Math.min(100, (cache.utilization || 0) * 100)}%` }} />
        <div className="seg remaining" style={{ flex: 1 }} />
      </div>

      <div className="cache-calls">
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
      </div>

      {entries.length === 0 ? (
        <p className="muted">Empty. Each reply stores the conversation so far so a later request can skip re-reading it.</p>
      ) : (
        <ul className="cache-entries">
          {[...entries].reverse().map((entry) => {
            const next = prediction?.reused_tokens > 0 && prediction.entry_id === entry.id
            const last = lastLookup?.reused_tokens > 0 && lastLookup.entry_id === entry.id
            return (
              <li key={entry.id} className={`cache-entry ${next ? 'is-next' : ''}`}>
                <div className="row-between">
                  <span>
                    <strong>#{entry.id}</strong> {KIND_LABEL[entry.kind] || entry.kind}
                    {entry.after_message ? <span className="muted"> · after message {entry.after_message}</span> : null}
                  </span>
                  <span className="ctx-tags">
                    {next && <span className="badge badge-ok">next request uses this</span>}
                    {last && <span className="badge">used last</span>}
                  </span>
                </div>
                <div className="entry-bar"><span style={{ width: `${(entry.tokens / longest) * 100}%` }} /></div>
                <div className="muted small">
                  {entry.tokens} tokens · {formatBytes(entry.nbytes)} · {age(entry.age_seconds)}
                </div>
              </li>
            )
          })}
        </ul>
      )}

      <div className="row-between">
        <span className="muted small">A request reuses an entry only if it starts with exactly that entry's tokens.</span>
        <button type="button" className="mini" disabled={busy || entries.length === 0} onClick={onClearCache}>
          Clear cache
        </button>
      </div>
    </div>
  )
}
