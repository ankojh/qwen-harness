import { useEffect, useRef, useState } from 'react'

function formatTime(seconds) {
  if (!seconds) return ''
  const date = new Date(seconds * 1000)
  return `${date.toLocaleTimeString([], { hour12: false })}.${String(date.getMilliseconds()).padStart(3, '0')}`
}

function megabytes(bytes) {
  return `${((bytes || 0) / 1e6).toFixed(1)} MB`
}

function summarize(type, data) {
  switch (type) {
    case 'token':
      return `#${data.index} id ${data.id} ${JSON.stringify(data.piece)} → ${JSON.stringify(data.text)}${data.channel ? ` [${data.channel}]` : ''}`
    case 'prefill.chunk':
      return `${data.processed_tokens}/${data.total_tokens} tokens (+${data.chunk_tokens}) ${data.tokens_per_second ?? '?'} tok/s`
    case 'generation.started':
      return `${data.generation_id} mode=${data.mode} messages=${data.messages}`
    case 'generation.finished':
      return `${data.generation_id} ${data.finish?.reason} · ${data.tokens_generated} tokens`
    case 'generation.stop_requested':
      return data.generation_id
    case 'config.changed':
      return `${data.field}: ${JSON.stringify(data.old)} → ${JSON.stringify(data.new)} (${data.applies})`
    case 'cache.lookup':
      return `reused ${data.reused_tokens} of ${data.prompt_tokens} · ${data.reason}`
    case 'cache.checkpoint':
      return `after msg ${data.after_message} · pos ${data.position} · ${data.used_existing ? 'used existing' : 'stored'} ${megabytes(data.nbytes)}`
    case 'cache.insert':
      return `#${data.entry_id} ${data.kind} · ${data.tokens} tokens · ${megabytes(data.nbytes)}${data.evicted ? ` · evicted ${data.evicted.length}` : ''}`
    case 'cache.evict':
      return `#${data.entry_id} ${data.kind} · ${megabytes(data.nbytes)} · ${data.reason}`
    case 'cache.verify':
      return `expected ${data.expected_offset} · offsets ${JSON.stringify(data.kv_offsets)} · ok=${data.ok}`
    case 'cache.clear':
      return `${data.entries} entries · ${data.reason}`
    case 'model.loading':
      return data.model_path
    case 'model.loaded':
      return `${data.model_path} · ${data.seconds}s`
    case 'model.unloaded':
      return `${data.active_bytes_before} → ${data.active_bytes_after} active bytes`
    case 'model.error':
      return data.message
    case 'resync':
      return data.reason
    case 'runtime.wired_limit':
      return data.note
    case 'runtime.mlx_cache_limit':
      return `${data.previous_bytes} → ${data.new_bytes} bytes`
    default:
      return JSON.stringify(data).slice(0, 140)
  }
}

export default function EventLog({ events, sseState, onClear }) {
  const [showTokens, setShowTokens] = useState(false)
  const [filter, setFilter] = useState('')
  const [expanded, setExpanded] = useState(() => new Set())
  const listRef = useRef(null)

  const needle = filter.trim().toLowerCase()
  const filtered = events.filter((event) => {
    if (!showTokens && event.type === 'token') return false
    if (!needle) return true
    return `${event.type} ${JSON.stringify(event.data)}`.toLowerCase().includes(needle)
  })

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight
  }, [filtered.length])

  function toggle(seq) {
    setExpanded((previous) => {
      const next = new Set(previous)
      if (next.has(seq)) next.delete(seq)
      else next.add(seq)
      return next
    })
  }

  return (
    <div className="event-log">
      <div className="log-head">
        <span className={`sse-badge ${sseState}`}>SSE {sseState}</span>
        <span className="muted">{events.length} buffered in this view</span>
        <label className="check">
          <input
            type="checkbox"
            checked={showTokens}
            onChange={(event) => setShowTokens(event.target.checked)}
          />
          show token events
        </label>
        <input
          className="log-filter"
          type="search"
          placeholder="filter type or data"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
        />
        <button type="button" className="mini" onClick={onClear}>clear view</button>
      </div>

      <div className="log-list" ref={listRef}>
        {filtered.map((event) => (
          <div key={`${event.seq}-${event.type}`} className="log-entry">
            <div
              className="log-row"
              role="button"
              tabIndex={0}
              onClick={() => toggle(event.seq)}
              onKeyDown={(keyEvent) => {
                if (keyEvent.key === 'Enter') toggle(event.seq)
              }}
            >
              <span className="seq">{event.seq}</span>
              <span className="time">{formatTime(event.time)}</span>
              <span className={`type type-${event.type.split('.')[0]}`}>{event.type}</span>
              <span className="summary">{summarize(event.type, event.data)}</span>
            </div>
            {expanded.has(event.seq) && (
              <pre className="log-json">{JSON.stringify(event.data, null, 2)}</pre>
            )}
          </div>
        ))}
        {filtered.length === 0 && <p className="muted">No events match.</p>}
      </div>
    </div>
  )
}
