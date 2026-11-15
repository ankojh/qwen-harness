import { chipText } from './TokenChips.jsx'
import { formatBytes } from './ConfigPanel.jsx'
import { kindLabel } from './mapModel.js'

// The inspector: details for whatever is selected on the map.  Three panes:
// a message (edit it — typing stages a what-if edit), a cache entry (memory
// bar, delete, relabel/restore from lens tokens), and a token (id, text,
// where its computation came from).  Every edit control is disabled in a
// historical view.

function InspectorShell({ title, onClose, children }) {
  return (
    <div className="inspector">
      <div className="head">
        <span className="muted-title">{title}</span>
        {onClose && (
          <button type="button" className="mini" onClick={onClose} aria-label="Close inspector">
            ✕
          </button>
        )}
      </div>
      {children}
    </div>
  )
}

// ---------------------------------------------------------------- entry pane
function EntryPane({
  entryRow,
  detail,
  detailError,
  contextLen,
  historical,
  running,
  busy,
  onDeleteEntry,
  onBack,
}) {
  const totals = detail?.totals || {}
  const recurrent = totals.recurrent_bytes || 0
  const kvUsed = totals.kv_used_bytes || 0
  const kvAllocated = totals.kv_allocated_bytes || 0
  const kvEmpty = Math.max(0, kvAllocated - kvUsed)
  const entryBytes = detail ? recurrent + kvAllocated : entryRow?.nbytes ?? 0
  const pct = (value) => (entryBytes > 0 ? `${(value / entryBytes) * 100}%` : '0%')
  const relabelCount = detail?.relabels?.length ?? entryRow?.relabelCount ?? 0

  return (
    <InspectorShell title={`cache entry #${entryRow.id}`} onClose={onBack}>
      <div>
        <strong>#{entryRow.id}</strong> {kindLabel(entryRow.kind)}
        {entryRow.afterMessage != null && <span className="muted"> · after message {entryRow.afterMessage}</span>}
        {relabelCount > 0 && (
          <span className="badge badge-off" title="tokens were relabelled; the stored state no longer matches the claimed tokens">
            label ≠ state
          </span>
        )}
        {entryRow.isNew && <span className="badge badge-ok">NEW</span>}
      </div>
      <p className="muted small">
        {entryRow.len} tokens · saved state covers positions 0–{entryRow.len - 1} ·{' '}
        {entryRow.nbytes != null ? formatBytes(entryRow.nbytes) : '?'}
        {entryRow.ageSeconds != null && ` · saved ${Math.round(entryRow.ageSeconds)}s ago`}
      </p>
      {entryRow.breakAt != null ? (
        <p className="note note-warn">
          Breaks at position {entryRow.breakAt}: the saved labels and this request diverge there.
          This model cannot rewind a state, so the entry is unusable for this request.
        </p>
      ) : entryRow.usable ? (
        <p className="note note-ok">
          Full prefix of this request — usable{entryRow.chosen ? ', and it is the one being restored' : ''}.
        </p>
      ) : (
        <p className="note">
          Not usable for this request
          {entryRow.breakAt == null && entryRow.len >= contextLen
            ? ' (it covers the whole context, so nothing would be left to compute)'
            : ''}.
        </p>
      )}

      {historical && (
        <p className="note">Historical snapshot — the stored model state is not available here; edit controls are disabled.</p>
      )}
      {detailError && <p className="note note-error">{String(detailError)} — the entry may have been evicted.</p>}
      {!historical && !running && !detail && !detailError && (
        <p className="muted small">Reading layer state… (unavailable while a generation runs)</p>
      )}
      {detail && (
        <>
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
          {(detail.layers || []).length > 0 && (
            <div className="table-scroll tall">
              <table className="data-table">
                <thead>
                  <tr><th>Layer</th><th>Class</th><th>Kind</th><th>Arrays (shape, dtype)</th><th>Used / allocated tokens</th><th>Bytes</th></tr>
                </thead>
                <tbody>
                  {detail.layers.map((layer) => (
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
          )}
        </>
      )}
      <div className="button-row wrap">
        <button
          type="button"
          className="mini danger"
          disabled={busy || historical || running}
          title={historical ? 'snapshots of past turns are read-only' : 'delete this entry'}
          onClick={onDeleteEntry}
        >
          Delete entry
        </button>
        <span className="muted small">
          {historical
            ? 'read-only for a past turn'
            : 'Relabel and Restore work from the lens: click a token on the entry row below the map.'}
        </span>
      </div>
    </InspectorShell>
  )
}

// --------------------------------------------------------------- message pane
function StagedSummary({ before, after, stops, running }) {
  if (running) return <p className="muted small">Wait for the generation to finish before editing.</p>
  const beforeReuse = before?.cache_prediction?.reused_tokens || 0
  const afterReuse = after?.cache_prediction?.reused_tokens || 0
  const beforeTokens = before?.budget?.input_tokens ?? before?.token_ids?.length ?? 0
  const afterTokens = after?.budget?.input_tokens ?? after?.token_ids?.length ?? 0
  return (
    <div className="edit-consequences">
      <strong>What would change</strong>
      <div className="muted small">
        reuse <span className={`${afterReuse < beforeReuse ? 'badge badge-off' : 'badge badge-ok'}`}>{beforeReuse} → {afterReuse}</span> ·
        tokens {beforeTokens} → {afterTokens}
        {afterReuse < beforeReuse ? ' — the bars recompute and break where the edit lands' : ''}
      </div>
      {stops.length > 0 && (
        <div>
          entries that stop matching:{' '}
          {stops.map(({ entry, divergence, extendsBeyond }) => (
            <span key={entry.id} className="badge badge-warn">
              #{entry.id} {kindLabel(entry.kind)} ·{' '}
              {extendsBeyond ? `request ends at token ${divergence}` : `diverges at token ${divergence}`}
            </span>
          ))}
        </div>
      )}
      <p className="muted small">
        Applying changes the context only. Saved cache states keep their original computation;
        incompatible entries stay stored but cannot be reused for the edited beginning.
      </p>
    </div>
  )
}

function MessagePane({
  messageIndex,
  message,
  staged,
  busy,
  historical,
  systemPrompt,
  checkpointBusy,
  onBeginEdit,
  onUpdate,
  onMove,
  onDeleteMessage,
  onInsertAfter,
  onCheckpoint,
  onApply,
  onDiscard,
  stagedReady,
  before,
  after,
  stops,
  changed,
  onBack,
}) {
  const isEditing = staged != null
  const editable = isEditing && !busy
  const disabled = !editable
  if (message == null) return <InspectorShell title="message" onClose={onBack}><p className="muted">This message is no longer in the conversation.</p></InspectorShell>

  return (
    <InspectorShell title={`message #${messageIndex + 1} · ${message.role}`} onClose={onBack}>
      {historical && (
        <p className="note">Historical snapshot — read-only. Return to the current state to edit.</p>
      )}
      {!isEditing && !historical && (
        <p className="muted small">Read-only preview. Click “Edit this message” to stage an edit — the map shows what would happen before you apply it.</p>
      )}
      <div className="edit-message-head">
        <select
          value={message.role}
          disabled={disabled || historical}
          onChange={(event) => onUpdate(messageIndex, { role: event.target.value })}
        >
          <option value="system">system</option>
          <option value="user">user</option>
          <option value="assistant">assistant</option>
        </select>
        <span className="muted">#{messageIndex + 1}</span>
        <label className="check">
          <input
            type="checkbox"
            checked={message.enabled !== false}
            disabled={disabled || historical}
            onChange={(event) => onUpdate(messageIndex, { enabled: event.target.checked })}
          />
          on
        </label>
        {changed && <span className="badge badge-warn">edited</span>}
        <span className="spacer" />
        <button type="button" className="mini" disabled={disabled || messageIndex === 0} onClick={() => onMove(messageIndex, -1)}>↑</button>
        <button type="button" className="mini" disabled={disabled || messageIndex === (staged?.length ?? 0) - 1} onClick={() => onMove(messageIndex, 1)}>↓</button>
        <button type="button" className="mini" disabled={disabled} onClick={() => onInsertAfter(messageIndex)}>insert after</button>
        <button type="button" className="mini danger" disabled={disabled} onClick={() => onDeleteMessage(messageIndex)}>✕</button>
        <button
          type="button"
          className="mini"
          disabled={disabled || checkpointBusy || message.enabled === false}
          title={
            message.enabled === false
              ? 'disabled messages are not sent, so there is no boundary after them'
              : 'snapshot the cache at the end of this message'
          }
          onClick={() => onCheckpoint(messageIndex)}
        >
          save checkpoint
        </button>
      </div>
      <textarea
        rows={3}
        value={message.content}
        aria-label={`Content for message ${messageIndex + 1}`}
        disabled={disabled || historical}
        placeholder="content"
        onChange={(event) => onUpdate(messageIndex, { content: event.target.value })}
      />
      {message.role === 'assistant' && (
        <textarea
          rows={2}
          className="reasoning"
          value={message.reasoning_content || ''}
          disabled={disabled || historical}
          placeholder="reasoning_content (the chat template drops it from history)"
          onChange={(event) => onUpdate(messageIndex, { reasoning_content: event.target.value })}
        />
      )}
      {!isEditing && !historical && (
        <div className="button-row">
          <button type="button" className="primary" disabled={busy} onClick={onBeginEdit}>
            Edit this message
          </button>
        </div>
      )}
      {isEditing && (
        <>
          <StagedSummary before={before} after={after} stops={stops} running={busy} />
          <div className="whatif-bar" role="status">
            <strong>Staged edit — the map above is a what-if</strong>
            <button type="button" className="primary" disabled={busy || !stagedReady} onClick={onApply}>
              Apply
            </button>
            <button type="button" disabled={busy} onClick={onDiscard}>Discard</button>
            {!stagedReady && <span className="muted small">measuring the staged context…</span>}
          </div>
        </>
      )}
    </InspectorShell>
  )
}

// ------------------------------------------------------------------ token pane
function TokenPane({
  position,
  token,
  spanRole,
  reused,
  containingMessageIndex,
  onSelectMessage,
  onBack,
}) {
  return (
    <InspectorShell title={`token at position ${position}`} onClose={onBack}>
      {token == null ? (
        <p className="muted">This position has no context token.</p>
      ) : (
        <>
          <div className="token-chips">
            <span className={`token-chip ${token.special ? 'tc-special' : ''}`}>{chipText(token)}</span>
          </div>
          <div className="facts compact">
            <div><dt>position</dt><dd>{position}</dd></div>
            <div><dt>token id</dt><dd className="mono">{token.id}</dd></div>
            <div><dt>piece</dt><dd className="mono break">{JSON.stringify(token.piece)}</dd></div>
            <div><dt>role</dt><dd>{spanRole ?? '—'}</dd></div>
            <div>
              <dt>computation</dt>
              <dd>
                {position < reused
                  ? <span className="badge badge-ok">restored from cache</span>
                  : <span className="badge">computed in this request</span>}
              </dd>
            </div>
          </div>
          {containingMessageIndex != null && (
            <button type="button" className="mini" onClick={() => onSelectMessage(containingMessageIndex)}>
              inspect message #{containingMessageIndex + 1}
            </button>
          )}
        </>
      )}
    </InspectorShell>
  )
}

// ------------------------------------------------------------------- switch
export default function Inspector(props) {
  const { pane, onClose } = props
  if (pane === 'entry') return <EntryPane {...props} onBack={onClose} />
  if (pane === 'message') return <MessagePane {...props} onBack={onClose} />
  if (pane === 'token') return <TokenPane {...props} onBack={onClose} />
  return (
    <InspectorShell title="nothing selected" onClose={onClose}>
      <p className="muted">
        Click a message on the tape or in the chat to edit it, a bar to inspect a saved state,
        or a token for its id and computation.
      </p>
    </InspectorShell>
  )
}
