import { useState } from 'react'
import CacheDiff from './CacheDiff.jsx'
import { highlightSpecial } from './highlight.jsx'

function HighlightedPrompt({ text, thinkStart, thinkEnd }) {
  return <pre className="prompt-text">{highlightSpecial(text, thinkStart, thinkEnd)}</pre>
}

function BudgetBar({ budget }) {
  const total = Math.max(1, budget.context_budget)
  const input = Math.min(100, (budget.input_tokens / total) * 100)
  const reservation = Math.min(100 - input, (budget.output_reservation / total) * 100)
  const remaining = Math.max(0, 100 - input - reservation)
  return (
    <div className="budget">
      <div className="budget-bar" role="img" aria-label="context budget">
        <span className="seg input" style={{ width: `${input}%` }} />
        <span className="seg reservation" style={{ width: `${reservation}%` }} />
        <span className="seg remaining" style={{ width: `${remaining}%` }} />
      </div>
      <div className="budget-legend">
        <span><i className="swatch input" /> input {budget.input_tokens}</span>
        <span><i className="swatch reservation" /> reservation {budget.output_reservation}</span>
        <span><i className="swatch remaining" /> remaining {budget.remaining}</span>
        <span className="muted">budget {budget.context_budget}</span>
      </div>
    </div>
  )
}

function MessageEditor({ messages, onChange }) {
  function update(index, patch) {
    onChange(messages.map((message, position) =>
      position === index ? { ...message, ...patch } : message,
    ))
  }

  function move(index, direction) {
    const target = index + direction
    if (target < 0 || target >= messages.length) return
    const next = [...messages]
    ;[next[index], next[target]] = [next[target], next[index]]
    onChange(next)
  }

  return (
    <div className="messages">
      {messages.map((message, index) => (
        <div className="message-card" key={index}>
          <div className="message-head">
            <span className="muted">#{index + 1}</span>
            <select
              value={message.role}
              onChange={(event) => update(index, { role: event.target.value })}
            >
              <option value="system">system</option>
              <option value="user">user</option>
              <option value="assistant">assistant</option>
            </select>
            <label className="check">
              <input
                type="checkbox"
                checked={message.enabled !== false}
                onChange={(event) => update(index, { enabled: event.target.checked })}
              />
              enabled
            </label>
            <span className="spacer" />
            <button type="button" className="mini" disabled={index === 0} onClick={() => move(index, -1)}>↑</button>
            <button type="button" className="mini" disabled={index === messages.length - 1} onClick={() => move(index, 1)}>↓</button>
            <button type="button" className="mini danger" onClick={() => onChange(messages.filter((_, position) => position !== index))}>delete</button>
          </div>
          <textarea
            rows={3}
            value={message.content}
            placeholder="content"
            onChange={(event) => update(index, { content: event.target.value })}
          />
          <textarea
            rows={2}
            className="reasoning"
            value={message.reasoning_content || ''}
            placeholder="reasoning_content (optional; the chat template may drop it from history)"
            onChange={(event) => update(index, { reasoning_content: event.target.value })}
          />
        </div>
      ))}
      <button
        type="button"
        onClick={() =>
          onChange([
            ...messages,
            { role: 'user', content: '', reasoning_content: null, enabled: true },
          ])
        }
      >
        Add message
      </button>
    </div>
  )
}

function CheckpointList({ preview }) {
  const checkpoints = preview.checkpoints || []
  return (
    <div className="inspector-block">
      <h4>Cache checkpoints</h4>
      {preview.checkpoint_reason && <p className="note">{preview.checkpoint_reason}</p>}
      {checkpoints.length > 0 && (
        <ul className="checkpoint-list">
          {checkpoints.map((entry) => (
            <li key={entry.after_message} className={entry.ok ? 'ok' : 'skipped'}>
              <span className="badge">{entry.ok ? 'accepted' : 'skipped'}</span>
              <span>after message {entry.after_message}</span>
              {entry.ok ? (
                <span className="mono">position {entry.position}</span>
              ) : (
                <span className="muted">{entry.reason}</span>
              )}
            </li>
          ))}
        </ul>
      )}
      {preview.checkpoint_estimate ? (
        <div className="estimate">
          <div className="muted">
            estimate from entry #{preview.checkpoint_estimate.derived_from.entry_id}
            {' '}({preview.checkpoint_estimate.derived_from.tokens} tokens measured)
          </div>
          <table className="data-table">
            <thead>
              <tr><th>position</th><th>recurrent</th><th>KV</th><th>total</th></tr>
            </thead>
            <tbody>
              {preview.checkpoint_estimate.rows.map((row) => (
                <tr key={row.position}>
                  <td>{row.position}</td>
                  <td>{(row.arrays_bytes / 1e6).toFixed(1)} MB</td>
                  <td>{(row.kv_bytes / 1e6).toFixed(1)} MB</td>
                  <td>{(row.estimated_bytes / 1e6).toFixed(1)} MB</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div>
            estimated total for new checkpoints:{' '}
            <strong>{(preview.checkpoint_estimate.total_bytes / 1e6).toFixed(1)} MB</strong>
          </div>
          <div className="note">{preview.checkpoint_estimate.note}</div>
        </div>
      ) : (
        <p className="muted">
          checkpoint estimate: {preview.checkpoint_estimate_reason || 'not available'}
        </p>
      )}
    </div>
  )
}

export default function InputPanel({
  session,
  onSessionChange,
  preview,
  previewError,
  onPreview,
  autoPreview,
  setAutoPreview,
  modelLoaded,
  thinkStart,
  thinkEnd,
}) {
  const [showTokens, setShowTokens] = useState(false)

  function setMode(mode) {
    onSessionChange({ ...session, mode })
  }

  return (
    <div className="input-panel">
      <div className="mode-switch">
        <button
          type="button"
          className={session.mode === 'messages' ? 'active' : ''}
          onClick={() => setMode('messages')}
        >
          Messages
        </button>
        <button
          type="button"
          className={session.mode === 'raw' ? 'active' : ''}
          onClick={() => setMode('raw')}
        >
          Raw
        </button>
      </div>

      {session.mode === 'messages' ? (
        <MessageEditor messages={session.messages} onChange={(messages) => onSessionChange({ ...session, messages })} />
      ) : (
        <div className="raw-mode">
          <textarea
            rows={10}
            value={session.raw_text}
            placeholder="Raw text. The chat template is bypassed."
            onChange={(event) => onSessionChange({ ...session, raw_text: event.target.value })}
          />
          <p className="note">Chat template bypassed. tokenizer.encode(text, add_special_tokens=False) is sent as-is.</p>
        </div>
      )}

      <div className="preview-controls">
        <button type="button" disabled={!modelLoaded} onClick={onPreview}>Preview prompt</button>
        <label className="check">
          <input
            type="checkbox"
            checked={autoPreview}
            disabled={!modelLoaded}
            onChange={(event) => setAutoPreview(event.target.checked)}
          />
          auto-refresh (tokenizer only)
        </label>
        {!modelLoaded && <span className="muted">Load the model first.</span>}
      </div>

      {previewError && <div className="note note-error">{previewError}</div>}

      {!preview ? (
        <p className="muted">Preview to see the exact prompt, tokens and cache plan.</p>
      ) : (
        <div className="inspector">
          <div className={`note ${preview.ok ? 'note-ok' : 'note-error'}`}>
            {preview.ok ? 'Prompt is within budget.' : preview.error}
          </div>

          <dl className="facts compact">
            <div><dt>tokens</dt><dd>{preview.token_ids.length}</dd></div>
            <div><dt>checkpoint</dt><dd>{preview.checkpoint_at ?? 'none'}</dd></div>
            <div><dt>starts in thinking</dt><dd>{String(preview.starts_in_thinking)}</dd></div>
            <div><dt>hash</dt><dd className="mono small">{preview.prompt_hash.slice(0, 23)}…</dd></div>
            <div><dt>mode</dt><dd>{preview.mode}</dd></div>
            <div><dt>cache reuse</dt><dd>{preview.cache_prediction?.reused_tokens ?? 0} tokens</dd></div>
          </dl>

          <BudgetBar budget={preview.budget} />

          <h4>Exact prompt</h4>
          <HighlightedPrompt
            text={preview.text}
            thinkStart={thinkStart}
            thinkEnd={thinkEnd}
          />

          <CheckpointList preview={preview} />

          <div className="inspector-block">
            <h4>Cache prediction</h4>
            <div>{preview.cache_prediction?.reason || 'no prediction'}</div>
          </div>

          <CacheDiff preview={preview} thinkStart={thinkStart} thinkEnd={thinkEnd} />

          <div className="inspector-block">
            <h4>Submitted messages</h4>
            <table className="data-table">
              <thead><tr><th>#</th><th>role</th><th>source</th><th>content</th></tr></thead>
              <tbody>
                {preview.submitted_messages.map((message) => (
                  <tr key={message.index}>
                    <td>{message.index}</td>
                    <td>{message.role}</td>
                    <td>{message.source}</td>
                    <td className="preview-cell">{message.content.slice(0, 90)}{message.content.length > 90 ? '…' : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {preview.dropped_messages.length > 0 && (
              <>
                <h4>Dropped for budget</h4>
                <ul>
                  {preview.dropped_messages.map((message) => (
                    <li key={message.message_index}>
                      #{message.message_index} {message.role}: {message.preview}
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>

          {preview.transformations.length > 0 && (
            <div className="inspector-block">
              <h4>Transformations</h4>
              <ul>{preview.transformations.map((line, index) => <li key={index}>{line}</li>)}</ul>
            </div>
          )}

          <details className="inspector-block">
            <summary>Template report</summary>
            <dl className="facts compact">
              <div><dt>source</dt><dd>{preview.template.source}</dd></div>
              <div><dt>variables</dt><dd>{preview.template.variables.join(', ') || '—'}</dd></div>
              <div><dt>passed</dt><dd>{preview.template.passed_kwargs.join(', ') || '—'}</dd></div>
              <div><dt>unused</dt><dd>{preview.template.unused_kwargs.join(', ') || '—'}</dd></div>
              <div><dt>reserved</dt><dd>{preview.template.reserved_kwargs.join(', ') || '—'}</dd></div>
              <div><dt>enable_thinking passed</dt><dd>{String(preview.template.enable_thinking_passed)}</dd></div>
            </dl>
            {preview.template.tokenize_mismatch && (
              <div className="note note-warn">
                encode(text) differs from apply_chat_template(tokenize=True):{' '}
                {JSON.stringify(preview.template.tokenize_mismatch)}
              </div>
            )}
          </details>

          <div className="inspector-block">
            <label className="check">
              <input type="checkbox" checked={showTokens} onChange={(event) => setShowTokens(event.target.checked)} />
              show token ids
            </label>
            {showTokens && (
              <div className="table-scroll tall">
                <table className="data-table token-table">
                  <thead><tr><th>#</th><th>id</th><th>piece</th></tr></thead>
                  <tbody>
                    {(preview.token_pieces || []).map((piece, index) => (
                      <tr key={index}>
                        <td>{index}</td>
                        <td>{piece.id}</td>
                        <td className="mono">{JSON.stringify(piece.piece)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
