// What the next request will send, message by message, and how the context
// grew over the turns so far.  Built from the tokenizer-only preview, so it
// updates as you type without running the model.

// Split the prompt tokens at every <|im_start|>: one segment per submitted
// message plus the trailing assistant opener the template adds.  Returns null
// when the split does not line up (raw mode, an unusual template).
function messageSegments(preview) {
  const pieces = preview?.token_pieces
  const messages = preview?.submitted_messages || []
  if (!pieces || preview.mode !== 'messages') return null
  const starts = []
  pieces.forEach((item, index) => {
    if (item.piece === '<|im_start|>') starts.push(index)
  })
  if (starts.length !== messages.length + 1) return null
  return starts.map((start, k) => ({
    start,
    end: k + 1 < starts.length ? starts[k + 1] : pieces.length,
    message: messages[k] || null,
  }))
}

function oneLine(text, limit = 90) {
  const flat = (text || '').replace(/\s+/g, ' ').trim()
  return flat.length > limit ? `${flat.slice(0, limit)}…` : flat
}

function PromptMap({ segments, reused, total }) {
  return (
    <div className="prompt-map" aria-label="Prompt tokens by message">
      {segments.map((segment, k) => {
        const size = segment.end - segment.start
        const cached = Math.max(0, Math.min(segment.end, reused) - segment.start)
        const label = segment.message ? `${segment.message.role} · ${size} tokens` : `assistant opener · ${size} tokens`
        return (
          <div key={k} className="map-seg" style={{ flexGrow: size }} title={`${label}${cached ? ` · ${cached} from cache` : ''}`}>
            {cached > 0 && <span className="map-cached" style={{ flexGrow: cached }} />}
            {size - cached > 0 && <span className="map-new" style={{ flexGrow: size - cached }} />}
          </div>
        )
      })}
      {total === 0 && <div className="map-seg" style={{ flexGrow: 1 }} />}
    </div>
  )
}

function TurnHistory({ turns }) {
  if (turns.length === 0) return null
  const widest = Math.max(...turns.map((turn) => turn.input + turn.generated), 1)
  const pct = (value) => `${(value / widest) * 100}%`
  return (
    <div className="turns">
      <h4>Growth by turn</h4>
      <div className="legend">
        <span><i className="sw sw-cached" />reused from cache</span>
        <span><i className="sw sw-new" />read by the model</span>
        <span><i className="sw sw-think" />thinking (dropped next turn)</span>
        <span><i className="sw sw-answer" />answer (kept)</span>
      </div>
      {turns.map((turn) => (
        <div key={turn.id} className="turn-row">
          <span className="turn-label">#{turn.n}</span>
          <div className="turn-bar">
            <span className="sw-cached" style={{ width: pct(turn.reused) }} />
            <span className="sw-new" style={{ width: pct(turn.input - turn.reused) }} />
            <span className="sw-think" style={{ width: pct(turn.thinking) }} />
            <span className="sw-answer" style={{ width: pct(turn.generated - turn.thinking) }} />
          </div>
          <span className="turn-nums muted">
            sent {turn.input} ({turn.reused} cached) → {turn.generated} out
          </span>
        </div>
      ))}
    </div>
  )
}

export default function ContextPanel({ preview, previewError, session, draft, turns, modelLoaded }) {
  if (!modelLoaded) return <p className="muted">Load the model to see the context.</p>
  if (session.messages.length === 0 && !draft.trim()) {
    return (
      <p className="muted">
        Empty. The conversation lives in this browser tab and is sent in full with every
        request; start typing to see what the first request will contain.
      </p>
    )
  }
  if (previewError) return <p className="note note-error">{previewError}</p>
  if (!preview) return <p className="muted">Measuring…</p>

  const budget = preview.budget || {}
  const total = budget.input_tokens ?? preview.token_ids?.length ?? 0
  const reused = preview.cache_prediction?.reused_tokens || 0
  const segments = messageSegments(preview)
  const draftIndex = draft.trim() ? session.messages.length + 1 : null
  const assistantTurns = []
  ;(preview.submitted_messages || []).forEach((message) => {
    if (message.role === 'assistant') assistantTurns.push(message.index)
  })
  const removed = (preview.transformations || []).join(' ')
  const pct = (value) => `${(value / (budget.context_budget || 1)) * 100}%`

  return (
    <div className="context-panel">
      <div className="ctx-headline">
        <span className="big-num">{total}</span>
        <span className="muted">
          tokens in the next request{draftIndex ? ' (including your draft)' : ''} · {reused} reusable from cache ·{' '}
          {total - reused} to read
        </span>
      </div>

      <div className="budget-bar" title="Context budget">
        <div className="seg input" style={{ width: pct(total) }} />
        <div className="seg reservation" style={{ width: pct(budget.output_reservation || 0) }} />
        <div className="seg remaining" style={{ flex: 1 }} />
      </div>
      <div className="budget-legend muted">
        <span><span className="swatch input" />prompt {total}</span>
        <span><span className="swatch reservation" />reserved for reply {budget.output_reservation}</span>
        <span><span className="swatch remaining" />free {budget.remaining}</span>
        <span>of {budget.context_budget}</span>
      </div>

      {segments && <PromptMap segments={segments} reused={reused} total={total} />}

      <ol className="ctx-messages">
        {(segments || (preview.submitted_messages || []).map((message) => ({ message }))).map((segment, k) => {
          const message = segment.message
          const size = segment.end != null ? segment.end - segment.start : null
          if (!message) {
            return (
              <li key="opener" className="ctx-msg opener">
                <span className="role-chip">template</span>
                <span className="muted">assistant turn opener{preview.starts_in_thinking ? ' + <think>' : ''}</span>
                <span className="ctx-size">{size}</span>
              </li>
            )
          }
          const cached = size != null ? Math.max(0, Math.min(segment.end, reused) - segment.start) : 0
          const turn = message.role === 'assistant' ? turns[assistantTurns.indexOf(message.index)] : null
          return (
            <li key={k} className="ctx-msg">
              <span className={`role-chip role-${message.role}`}>{message.role}</span>
              <span className="ctx-text">
                {oneLine(message.content) || <span className="muted">(empty)</span>}
                <span className="ctx-tags">
                  {message.index === draftIndex && <span className="badge badge-warn">draft</span>}
                  {message.source && message.source !== 'session' && <span className="badge">{message.source}</span>}
                  {size != null && cached === size && <span className="badge badge-ok">cached</span>}
                  {size != null && cached > 0 && cached < size && <span className="badge badge-ok">{cached} cached</span>}
                  {turn?.thinking > 0 && <span className="badge">{turn.thinking} thinking tokens dropped</span>}
                  {removed.includes(`message ${message.index} `) && <span className="badge">template removed reasoning</span>}
                </span>
              </span>
              <span className="ctx-size">{size ?? ''}</span>
            </li>
          )
        })}
      </ol>
      {(preview.dropped_messages || []).length > 0 && (
        <p className="note note-warn">{preview.dropped_messages.length} message(s) dropped to fit the budget.</p>
      )}

      <TurnHistory turns={turns} />

      <details>
        <summary>Exact prompt text</summary>
        <pre className="code-block">{preview.text}</pre>
      </details>
    </div>
  )
}
