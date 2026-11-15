import { useEffect, useMemo, useState } from 'react'
import TokenChips from './TokenChips.jsx'
import { commonPrefix, diffContexts, lcsRanges, messageOf, stopsMatching } from './turnDiffs.js'
import { formatError, getCacheTokens, previewPrompt, saveCheckpoint } from './api.js'

// The context: what the viewed request sends, message by message.  Four
// views: Overview (the summary as before), Tokens (every token as a chip,
// grouped by message, plus a per-message word/token table), Changes (what
// this request changed versus the previous turn, plus the edits applied
// before it), and Edit (a staged copy of the conversation with per-message
// operations and a consequences card; only for "Next").  A historical turn
// shows its recorded snapshot and stays read-only.

function oneLine(text, limit = 90) {
  const flat = (text || '').replace(/\s+/g, ' ').trim()
  return flat.length > limit ? `${flat.slice(0, limit)}…` : flat
}

function wordCount(text) {
  return (text || '').trim().split(/\s+/).filter(Boolean).length
}

// Template tokens inside one message span: the <|im_start|> and role token,
// special tokens, and whitespace.  The opener span is entirely template.
function templateTokens(span, pieces) {
  if (span.kind === 'opener') return span.end - span.start
  let count = 0
  for (let position = span.start; position < span.end; position += 1) {
    const token = pieces[position]
    const relative = position - span.start
    if (relative <= 1) count += 1
    else if (token?.special) count += 1
    else if (/^\s*$/.test(token?.text || '')) count += 1
  }
  return count
}

function spanLabel(span) {
  if (span.kind === 'opener') return `assistant opener · ${span.end - span.start} tokens`
  return `${span.role} · ${span.end - span.start} tokens`
}

// One edit-log line per message-level change when the staged copy is applied.
function describeChanges(before, after) {
  const lines = []
  const length = Math.max(before.length, after.length)
  for (let index = 0; index < length; index += 1) {
    const prev = before[index]
    const next = after[index]
    const label = `message #${index + 1}`
    if (prev && next && prev !== next) {
      const changes = []
      if (prev.content !== next.content) {
        changes.push(`${JSON.stringify(oneLine(prev.content, 30))} → ${JSON.stringify(oneLine(next.content, 30))}`)
      }
      if (prev.role !== next.role) changes.push(`role ${prev.role} → ${next.role}`)
      if ((prev.enabled !== false) !== (next.enabled !== false)) {
        changes.push(next.enabled === false ? 'switched off' : 'switched on')
      }
      if ((prev.reasoning_content || '') !== (next.reasoning_content || '')) {
        changes.push(next.reasoning_content ? 'reasoning set' : 'reasoning removed')
      }
      if (changes.length) lines.push(`edited ${label} (${next.role}): ${changes.join(', ')}`)
    } else if (prev && !next) {
      lines.push(`deleted ${label} (${prev.role}) “${oneLine(prev.content, 40)}”`)
    } else if (!prev && next) {
      lines.push(`inserted ${label} (${next.role}) “${oneLine(next.content, 40)}”`)
    }
  }
  return lines
}

function PromptMap({ spans, reused, total }) {
  return (
    <div className="prompt-map" aria-label="Prompt tokens by message">
      {spans.map((span, key) => {
        const size = span.end - span.start
        const cached = Math.max(0, Math.min(span.end, reused) - span.start)
        return (
          <div
            key={key}
            className="map-seg"
            style={{ flexGrow: size }}
            title={`${spanLabel(span)}${cached ? ` · ${cached} from cache` : ''}`}
          >
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

function TokensView({ report, spans, pieces, reused }) {
  const submitted = report.submitted_messages || []
  const totals = { userWords: 0, assistantWords: 0, systemWords: 0, template: 0, tokens: 0 }
  return (
    <div className="context-tokens">
      {spans.length === 0 && (
        <p className="muted">
          No message spans could be computed (raw mode or an unusual template), so the
          tokens are shown as one run.
        </p>
      )}
      {spans.length === 0 && (
        <TokenChips
          tokens={pieces}
          marks={reused > 0 ? [{ start: 0, end: reused, kind: 'cached' }] : undefined}
        />
      )}
      {spans.map((span, key) => {
        const message = messageOf(report, span)
        const content = message?.content || ''
        const words = wordCount(content)
        const chars = content.length
        const size = span.end - span.start
        const template = templateTokens(span, pieces)
        if (message?.role === 'user') totals.userWords += words
        else if (message?.role === 'assistant') totals.assistantWords += words
        else if (message?.role === 'system') totals.systemWords += words
        totals.template += template
        totals.tokens += size
        const cachedStart = 0
        const cachedEnd = Math.min(size, Math.max(0, Math.min(reused, span.end) - span.start))
        return (
          <div key={key} className="tok-group">
            <div className="tok-group-head row-between">
              <span>
                <span className={`role-chip role-${span.role}`}>{span.role}</span>{' '}
                {span.kind === 'opener' ? (
                  <span className="muted">assistant turn opener{report.starts_in_thinking ? ' + <think>' : ''}</span>
                ) : (
                  <span>
                    message {span.index} · “{oneLine(content, 60) || <span className="muted">(empty)</span>}”
                  </span>
                )}
              </span>
              <span className="ctx-size">{size}</span>
            </div>
            <TokenChips
              tokens={pieces.slice(span.start, span.end)}
              offset={span.start}
              marks={cachedEnd > cachedStart ? [{ start: cachedStart, end: cachedEnd, kind: 'cached' }] : undefined}
            />
          </div>
        )
      })}
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr><th>Message</th><th>Words</th><th>Chars</th><th>Tokens</th><th>Template tokens</th><th>Tok/word</th></tr>
          </thead>
          <tbody>
            {spans.map((span, key) => {
              const message = messageOf(report, span)
              const content = message?.content || ''
              const size = span.end - span.start
              const words = wordCount(content)
              return (
                <tr key={key}>
                  <td>
                    {span.kind === 'opener' ? (
                      <span className="muted">assistant opener</span>
                    ) : (
                      <>#{span.index} {span.role}</>
                    )}
                  </td>
                  <td>{span.kind === 'opener' ? '—' : words}</td>
                  <td>{span.kind === 'opener' ? '—' : content.length}</td>
                  <td>{size}</td>
                  <td>{templateTokens(span, pieces)}</td>
                  <td>{words > 0 ? (size / words).toFixed(1) : '—'}</td>
                </tr>
              )
            })}
            <tr>
              <td><strong>total</strong></td>
              <td title="your words">{totals.userWords} you</td>
              <td />
              <td>{totals.tokens}</td>
              <td>{totals.template} template</td>
              <td>
                {totals.assistantWords > 0 || totals.systemWords > 0
                  ? `model answers ${totals.assistantWords}${totals.systemWords ? ` · system ${totals.systemWords}` : ''}`
                  : ''}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

function DiffRow({ row, report, prevReport }) {
  const prevPieces = prevReport.token_pieces || []
  const nextPieces = report.token_pieces || []
  const spanTitle = (span) =>
    span?.kind === 'opener' ? 'assistant opener' : `#${span?.index ?? '?'} ${span?.role ?? '?'}`
  switch (row.kind) {
    case 'unchanged': {
      const size = row.next.end - row.next.start
      return (
        <div className="change-row">
          <div>{spanTitle(row.next)}: <span className="muted">unchanged ({size} tokens)</span></div>
        </div>
      )
    }
    case 'removed': {
      const size = row.prev.end - row.prev.start
      const message = messageOf(prevReport, row.prev)
      return (
        <div className="change-row">
          <div>
            {spanTitle(row.prev)}: <span className="badge badge-off">removed (−{size} tokens)</span>
            {message && <> “{oneLine(message.content, 60)}”</>}
          </div>
          <TokenChips
            tokens={prevPieces.slice(row.prev.start, row.prev.end)}
            offset={row.prev.start}
            marks={[{ start: 0, end: size, kind: 'removed' }]}
          />
        </div>
      )
    }
    case 'added': {
      const size = row.next.end - row.next.start
      const message = messageOf(report, row.next)
      return (
        <div className="change-row">
          <div>
            {spanTitle(row.next)}: <span className="badge badge-ok">added (+{size} tokens)</span>
            {message && <> “{oneLine(message.content, 60)}”</>}
          </div>
          <TokenChips
            tokens={nextPieces.slice(row.next.start, row.next.end)}
            offset={row.next.start}
            marks={[{ start: 0, end: size, kind: 'added' }]}
          />
        </div>
      )
    }
    case 're-rendered':
    case 'changed': {
      const prevSize = row.prev.end - row.prev.start
      const nextSize = row.next.end - row.next.start
      const label = row.kind === 're-rendered' ? 're-rendered' : 'changed'
      return (
        <div className="change-row">
          <div>
            {spanTitle(row.next)}:{' '}
            <span className="badge badge-warn">
              {label} {prevSize} → {nextSize}
            </span>
          </div>
          {row.diff?.approximate && <p className="muted small">Large changed region: highlighting is approximate inside this region.</p>}
          <div className="chips-wrap">
            <span className="muted small">removed</span>
            <TokenChips
              tokens={prevPieces.slice(row.prev.start, row.prev.end)}
              offset={row.prev.start}
              marks={row.diff.removed.map((range) => ({ ...range, kind: 'removed' }))}
            />
          </div>
          <div className="chips-wrap">
            <span className="muted small">added</span>
            <TokenChips
              tokens={nextPieces.slice(row.next.start, row.next.end)}
              offset={row.next.start}
              marks={row.diff.added.map((range) => ({ ...range, kind: 'added' }))}
            />
          </div>
        </div>
      )
    }
    default:
      return null
  }
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

function ChangesView({ report, prevTurn, editLog, historical }) {
  return (
    <div className="change-list">
      <EditLog edits={editLog} historical={historical} />
      {!prevTurn || !prevTurn.context ? (
        <p className="muted">First turn — nothing to compare.</p>
      ) : (
        <>
          {(() => {
            const diff = diffContexts(prevTurn.context, report, prevTurn.thinking || 0)
            return (
              <>
                <p className="muted">
                  Common prefix {diff.commonPrefix} tokens ·{' '}
                  {diff.prevTokens} → {diff.nextTokens} tokens (net {diff.netChange >= 0 ? '+' : ''}{diff.netChange}) ·{' '}
                  thinking from turn {prevTurn.n} not carried: {diff.thinkingNotCarried} tokens
                </p>
                {diff.rows.map((row, key) => (
                  <DiffRow key={key} row={row} report={report} prevReport={prevTurn.context} />
                ))}
              </>
            )
          })()}
        </>
      )}
    </div>
  )
}

// ---------------------------------------------------------------- Edit view
function InsertRow({ position, onInsert, disabled }) {
  return (
    <div className="edit-insert">
      <button
        type="button"
        className="mini"
        disabled={disabled}
        onClick={() => onInsert(position)}
      >
        + insert message here
      </button>
    </div>
  )
}

function ConsequencesCard({
  preview,
  stagedSession,
  stagedPreview,
  stagedPreviewError,
  cacheTokens,
  busy,
  onApply,
  onDiscard,
}) {
  const nowReport = preview
  const nowIds = nowReport?.token_ids || []
  const afterIds = stagedPreview?.token_ids || []
  const nowTokens = nowReport?.budget?.input_tokens ?? nowIds.length
  const afterTokens = stagedPreview?.budget?.input_tokens ?? afterIds.length
  const nowReuse = nowReport?.cache_prediction?.reused_tokens || 0
  const nowEntry = nowReport?.cache_prediction?.entry_id
  const afterReuse = stagedPreview?.cache_prediction?.reused_tokens || 0
  const afterEntry = stagedPreview?.cache_prediction?.entry_id
  const firstDiff = commonPrefix(nowIds, afterIds)
  const changed = firstDiff < Math.max(nowIds.length, afterIds.length)
  const stops = stopsMatching(cacheTokens, nowIds, afterIds)
  const reuse = (tokens, entry) => (tokens > 0 ? `${tokens} (#${entry})` : '0')
  const tokenDiff = lcsRanges(nowIds, afterIds)
  const ready = Boolean(nowReport && stagedPreview && !stagedPreviewError)

  return (
    <div className="edit-consequences">
      <strong>If you apply these edits</strong>
      {stagedPreviewError && <p className="note note-error">{stagedPreviewError}</p>}
      {!ready ? <p role="status">{stagedPreviewError ? 'Fix the preview error before applying.' : 'Measuring staged context and cache consequences…'}</p> : <>
      <div className="staged-diff">
        <strong>Context diff · current → staged (draft included on both sides)</strong>
        <p className="muted small">{tokenDiff.approximate ? 'Large changed region: red and green mark the entire changed region, including any unchanged IDs inside it. Matching beginning and ending remain uncoloured.' : 'Red = removed token IDs. Green = added token IDs. Uncoloured tokens are unchanged.'}</p>
        <div className="chips-wrap"><b>Before</b><TokenChips tokens={nowReport?.token_pieces || []} marks={tokenDiff.removed.map((r) => ({ ...r, kind: 'removed' }))} /></div>
        <div className="chips-wrap"><b>After</b><TokenChips tokens={stagedPreview?.token_pieces || []} marks={tokenDiff.added.map((r) => ({ ...r, kind: 'added' }))} /></div>
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr><th></th><th>Now</th><th>After edits</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Tokens in next request</td>
              <td>{nowTokens}</td>
              <td>{afterTokens}</td>
            </tr>
            <tr>
              <td>Reused from cache</td>
              <td>{reuse(nowReuse, nowEntry)}</td>
              <td>{reuse(afterReuse, afterEntry)}</td>
            </tr>
            <tr>
              <td>First changed token</td>
              <td>—</td>
              <td>
                {!nowIds.length && !afterIds.length
                  ? '—'
                  : changed
                    ? `position ${firstDiff}`
                    : 'none (identical tokens)'}
              </td>
            </tr>
            <tr>
              <td>Entries that stop matching</td>
              <td>—</td>
              <td>
                {stops.length === 0 ? (
                  'none'
                ) : (
                  stops.map(({ entry, divergence, extendsBeyond }) => (
                    <div key={entry.id} className="badge badge-warn">
                      #{entry.id} {entry.kind}{entry.after_message ? ` after message ${entry.after_message}` : ''} ·{' '}
                      {extendsBeyond
                        ? `the request ends at token ${divergence}`
                        : `diverges at token ${divergence}`}
                    </div>
                  ))
                )}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="muted small">Applying changes the context only. Saved cache states keep their original computation; incompatible entries remain stored but cannot be reused for the edited beginning.</p>
      </>}
      <div className="chips-wrap">
        <button type="button" className="primary" disabled={busy || !ready || stagedPreview?.ok === false} onClick={onApply}>Apply</button>
        <button type="button" disabled={busy} onClick={onDiscard}>Discard</button>
        {busy && <span className="muted small">wait for the generation to finish</span>}
      </div>
    </div>
  )
}

function EditView({
  session,
  draft,
  staged,
  setStaged,
  preview,
  cacheState,
  config,
  systemPrompt,
  modelLoaded,
  busy,
  onApplyStaged,
  onCacheAction,
  onLeave,
}) {
  const [beforePreview, setBeforePreview] = useState(null)
  const [stagedPreview, setStagedPreview] = useState(null)
  const [stagedPreviewError, setStagedPreviewError] = useState(null)
  const [cacheTokens, setCacheTokens] = useState([])
  const [checkpointBusy, setCheckpointBusy] = useState(false)

  const stagedSession = useMemo(
    () => (staged ? { mode: 'messages', messages: draft.trim() ? [...staged, { role: 'user', content: draft.trim(), enabled: true, reasoning_content: null }] : staged } : null),
    [staged, draft],
  )

  // Consequences card: preview the staged session (tokenizer only) and the
  // live cache token lists, debounced like the main preview.
  useEffect(() => {
    if (!stagedSession || !modelLoaded) {
      setStagedPreview(null)
      setStagedPreviewError(null)
      setCacheTokens([])
      return undefined
    }
    setBeforePreview(null)
    setStagedPreview(null)
    setStagedPreviewError(null)
    let cancelled = false
    const timer = setTimeout(async () => {
      try {
        const currentSession = { ...session, messages: draft.trim() ? [...session.messages, { role: 'user', content: draft.trim(), enabled: true, reasoning_content: null }] : session.messages }
        const [report, before, tokenLists] = await Promise.all([
          previewPrompt(stagedSession),
          previewPrompt(currentSession),
          getCacheTokens(),
        ])
        if (!cancelled) {
          setBeforePreview(before)
          setStagedPreview(report)
          setStagedPreviewError(report.ok === false ? report.error || 'Invalid staged context' : null)
          setCacheTokens(tokenLists.entries || [])
        }
      } catch (error) {
        if (!cancelled) setStagedPreviewError(formatError(error.detail || error))
      }
    }, 300)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [stagedSession, modelLoaded, cacheState, config, session, draft])

  function update(index, patch) {
    setStaged(staged.map((message, position) =>
      position === index ? { ...message, ...patch } : message,
    ))
  }

  function move(index, direction) {
    const target = index + direction
    if (target < 0 || target >= staged.length) return
    const next = [...staged]
    ;[next[index], next[target]] = [next[target], next[index]]
    setStaged(next)
  }

  function removeAt(index) {
    setStaged(staged.filter((_, position) => position !== index))
  }

  function insert(position) {
    setStaged([
      ...staged.slice(0, position),
      { role: 'user', content: '', reasoning_content: null, enabled: true },
      ...staged.slice(position),
    ])
  }

  function addSystem() {
    setStaged([
      { role: 'system', content: '', reasoning_content: null, enabled: true },
      ...staged,
    ])
  }

  // The checkpoint boundary number uses the same numbering the server's
  // checkpoint table uses: submitted messages (enabled only), with the
  // configured system prompt as message 1 when set.
  function submittedIndexOf(index) {
    const before = staged.slice(0, index).filter((message) => message.enabled !== false).length
    return before + 1 + (systemPrompt ? 1 : 0)
  }

  function checkpointAt(index) {
    const afterMessage = submittedIndexOf(index)
    setCheckpointBusy(true)
    onCacheAction(
      () => saveCheckpoint(stagedSession, afterMessage),
      (result) =>
        result.used_existing
          ? `saved a checkpoint after message ${afterMessage}: entry #${result.entry_id} already held it (marked used)`
          : `saved a checkpoint after message ${afterMessage}: entry #${result.entry_id}, ${result.tokens} tokens`,
    )
      .catch(() => {})
      .finally(() => setCheckpointBusy(false))
  }

  const changedAt = (index) => {
    const prev = session.messages[index]
    const next = staged[index]
    if (!prev || !next) return true
    return (
      prev.content !== next.content ||
      prev.role !== next.role ||
      (prev.enabled !== false) !== (next.enabled !== false) ||
      (prev.reasoning_content || '') !== (next.reasoning_content || '')
    )
  }

  function handleApply() {
    const lines = describeChanges(session.messages, staged)
    onApplyStaged({ mode: 'messages', messages: staged }, lines)
    onLeave('overview')
  }

  function handleDiscard() {
    onLeave('overview')
  }

  return (
    <div className="edit-view">
      <div className="row-between">
        <span className="muted small">
          Edits are staged: the live conversation does not change until you Apply.
        </span>
        <button type="button" className="mini" disabled={busy} onClick={addSystem}>
          + system message
        </button>
      </div>
      {systemPrompt && (
        <p className="muted small">
          Note: the configured system prompt (“{oneLine(systemPrompt, 40)}”) is inserted as
          message 1 by the server, before these messages.
        </p>
      )}
      <ol className="edit-messages">
        {staged.map((message, index) => (
          <li key={index}>
            <InsertRow position={index} onInsert={insert} disabled={busy} />
            <div className={`edit-message ${changedAt(index) ? 'changed' : ''}`}>
              <div className="edit-message-head">
                <select
                  value={message.role}
                  disabled={busy}
                  onChange={(event) => update(index, { role: event.target.value })}
                >
                  <option value="system">system</option>
                  <option value="user">user</option>
                  <option value="assistant">assistant</option>
                </select>
                <span className="muted">#{index + 1}</span>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={message.enabled !== false}
                    disabled={busy}
                    onChange={(event) => update(index, { enabled: event.target.checked })}
                  />
                  on
                </label>
                {changedAt(index) && <span className="badge badge-warn">edited</span>}
                <span className="spacer" />
                <button type="button" className="mini" disabled={busy || index === 0} onClick={() => move(index, -1)}>↑</button>
                <button type="button" className="mini" disabled={busy || index === staged.length - 1} onClick={() => move(index, 1)}>↓</button>
                <button type="button" className="mini danger" disabled={busy} onClick={() => removeAt(index)}>✕</button>
                <button
                  type="button"
                  className="mini"
                  disabled={busy || checkpointBusy || message.enabled === false}
                  title={
                    message.enabled === false
                      ? 'disabled messages are not sent, so there is no boundary after them'
                      : `snapshot the cache at the end of this message (server message ${submittedIndexOf(index)})`
                  }
                  onClick={() => checkpointAt(index)}
                >
                  save checkpoint
                </button>
              </div>
              <textarea
                rows={2}
                value={message.content}
                aria-label={`Content for message ${index + 1}`}
                disabled={busy}
                placeholder="content"
                onChange={(event) => update(index, { content: event.target.value })}
              />
              {message.role === 'assistant' && (
                <textarea
                  rows={2}
                  className="reasoning"
                  value={message.reasoning_content || ''}
                  disabled={busy}
                  placeholder="reasoning_content (the chat template drops it from history)"
                  onChange={(event) => update(index, { reasoning_content: event.target.value })}
                />
              )}
            </div>
          </li>
        ))}
        <InsertRow position={staged.length} onInsert={insert} disabled={busy} />
      </ol>
      <ConsequencesCard
        preview={beforePreview}
        stagedSession={stagedSession}
        stagedPreview={stagedPreview}
        stagedPreviewError={stagedPreviewError}
        cacheTokens={cacheTokens}
        busy={busy}
        onApply={handleApply}
        onDiscard={handleDiscard}
      />
    </div>
  )
}

export default function ContextPanel({
  preview,
  previewError,
  session,
  draft,
  turns,
  modelLoaded,
  running,
  viewTurn,
  prevTurn,
  pendingEdits,
  systemPrompt,
  cacheState,
  config,
  onApplyStaged,
  onCacheAction,
  onEditingChange,
}) {
  const [view, setView] = useState('overview')
  const [staged, setStaged] = useState(null)
  const historical = Boolean(viewTurn)
  const report = historical ? viewTurn.context : preview
  // Editing is only for "Next", and never while a generation runs.
  const effectiveView = view === 'edit' && (historical || running) ? 'overview' : view
  const editLog = historical ? viewTurn.edits || [] : pendingEdits || []

  useEffect(() => {
    onEditingChange?.(effectiveView === 'edit')
    return () => onEditingChange?.(false)
  }, [effectiveView, onEditingChange])

  function enterEdit() {
    if (view === 'edit' && staged) return
    setStaged(session.messages.map((message) => ({ ...message })))
    setView('edit')
  }

  function leaveEdit(nextView) {
    setStaged(null)
    setView(nextView)
  }

  const tabs = (
    <div className="tabs" role="tablist">
      {[
        ['overview', 'Current'],
        ['tokens', 'Tokens'],
        ['changes', 'History diff'],
        ...(historical ? [] : [['edit', 'Edit context']]),
      ].map(([id, label]) => (
        <button
          key={id}
          type="button"
          role="tab"
          className={effectiveView === id ? 'active' : ''}
          disabled={(id === 'edit' && running) || (effectiveView === 'edit' && id !== 'edit')}
          title={id === 'edit' && running ? 'wait for the generation to finish' : undefined}
          onClick={() => (id === 'edit' ? enterEdit() : setView(id))}
        >
          {label}
        </button>
      ))}
    </div>
  )

  if (!modelLoaded) return <div>{tabs}<p className="muted">Load the model to measure context tokens. The conversation lives in this tab.</p></div>
  if (historical && !report) return <p className="muted">This turn recorded no context.</p>

  if (effectiveView === 'edit') {
    return (
      <div className="context-panel">
        <div className="ctx-headline">
          <span className="big-num">{session.messages.length}</span>
          <span className="muted">
            messages in the conversation · editing a staged copy
          </span>
        </div>
        <div className="tabs" role="tablist">
          {[
            ['overview', 'Current'],
            ['tokens', 'Tokens'],
            ['changes', 'History diff'],
            ['edit', 'Edit context'],
          ].map(([id, label]) => (
            <button
              key={id}
              type="button"
              role="tab"
              className={effectiveView === id ? 'active' : ''}
              disabled={(id === 'edit' && running) || (effectiveView === 'edit' && id !== 'edit')}
              title={id === 'edit' && running ? 'wait for the generation to finish' : undefined}
              onClick={() => (id === 'edit' ? enterEdit() : setView(id))}
            >
              {label}
            </button>
          ))}
        </div>
        <EditView
          session={session}
          draft={draft}
          staged={staged}
          setStaged={setStaged}
          preview={preview}
          cacheState={cacheState}
          config={config}
          systemPrompt={systemPrompt}
          modelLoaded={modelLoaded}
          busy={running}
          onApplyStaged={onApplyStaged}
          onCacheAction={onCacheAction}
          onLeave={leaveEdit}
        />
      </div>
    )
  }

  if (!report) {
    return <div className="context-panel">{tabs}
      {previewError ? <p className="note note-error">{previewError}</p> : <p className="muted" role="status">{session.messages.length || draft.trim() ? running ? 'Generation in progress · current messages below; measurements update when it finishes.' : 'Measuring current context…' : 'No messages yet. Type in Input or use Edit context to add one.'}</p>}
      <ol className="ctx-messages">{session.messages.map((message, index) => <li className="ctx-msg" key={index}><b>{message.role}{message.enabled === false ? ' · disabled' : ''}</b><span className="ctx-text full-text">{message.content || '(empty)'}</span></li>)}
      {draft.trim() && <li className="ctx-msg"><b>user · draft</b><span className="ctx-text full-text">{draft.trim()}</span></li>}</ol>
      <p className="muted small">Context is message text, not saved model state. The full rendered token sequence is sent each request. Reloading this tab clears the conversation; server cache entries remain.</p>
    </div>
  }
  if (previewError && !historical) return <div>{tabs}<p className="note note-error">{previewError}</p></div>

  const pieces = report.token_pieces || []
  const spans = report.spans || []
  const budget = report.budget || {}
  const total = budget.input_tokens ?? report.token_ids?.length ?? 0
  const reused = historical
    ? viewTurn.reused || 0
    : report.cache_prediction?.reused_tokens || 0
  const draftIndex = !historical && draft.trim() ? (report.submitted_messages || []).findLast((message) => message.role === 'user' && message.source === 'session')?.index : null
  const assistantTurns = []
  ;(report.submitted_messages || []).forEach((message) => {
    if (message.role === 'assistant') assistantTurns.push(message.index)
  })
  const removed = (report.transformations || []).join(' ')
  const pct = (value) => `${(value / (budget.context_budget || 1)) * 100}%`
  const subject = historical ? `turn ${viewTurn.n}'s request` : 'current context (next request)'

  if (effectiveView === 'tokens') {
    return (
      <div className="context-panel">
        <div className="ctx-headline">
          <span className="big-num">{total}</span>
          <span className="muted">
            tokens in {subject} · {reused} reusable from cache · {total - reused} to read
          </span>
        </div>
        {tabs}
        <TokensView report={report} spans={spans} pieces={pieces} reused={reused} />
      </div>
    )
  }

  if (effectiveView === 'changes') {
    return (
      <div className="context-panel">
        <div className="ctx-headline">
          <span className="big-num">{total}</span>
          <span className="muted">
            tokens in {subject}
            {historical ? '' : draftIndex ? ' (including your draft)' : ''}
          </span>
        </div>
        {tabs}
        <ChangesView
          report={report}
          prevTurn={prevTurn}
          editLog={editLog}
          historical={historical}
        />
      </div>
    )
  }

  return (
    <div className="context-panel">
      <div className="ctx-headline">
        <span className="big-num">{total}</span>
        <span className="muted">
          tokens in {subject}{!historical && draftIndex ? ' (including your draft)' : ''} ·{' '}
          {reused} reusable from cache · {total - reused} to read
        </span>
      </div>

      {tabs}

      <p className="context-explanation">These are the messages the model will read{draftIndex ? ', including your unsent draft' : ''}. <strong>{reused} tokens of input computation can be restored from cache; {total - reused} still need processing.</strong> Matching words later in the context do not count as reuse.</p>
      {report.ok === false && <p className="note note-error">{report.error}</p>}
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

      {spans.length > 0 && <PromptMap spans={spans} reused={reused} total={total} />}

      <ol className="ctx-messages">
        {(spans.length > 0 ? spans : (report.submitted_messages || []).map((message) => ({ start: null, end: null, message }))).map(
          (span, key) => {
            const message = spans.length > 0 ? messageOf(report, span) : span.message
            const start = span.start ?? null
            const end = span.end ?? null
            const size = start != null && end != null ? end - start : null
            if (!message) {
              return (
                <li key="opener" className="ctx-msg opener">
                  <span className="role-chip">template</span>
                  <span className="muted">
                    assistant turn opener{report.starts_in_thinking ? ' + <think>' : ''}
                  </span>
                  <span className="ctx-size">{size}</span>
                </li>
              )
            }
            const cached = size != null ? Math.max(0, Math.min(end, reused) - start) : 0
            const turn =
              message.role === 'assistant' ? turns[assistantTurns.indexOf(message.index)] : null
            return (
              <li key={key} className="ctx-msg">
                <span className={`role-chip role-${message.role}`}>{message.role}</span>
                <span className="ctx-text">
                  <span className="full-text">{message.content || <span className="muted">(empty)</span>}</span>
                  <span className="ctx-tags">
                    {message.index === draftIndex && <span className="badge badge-warn">draft</span>}
                    {message.source && message.source !== 'session' && (
                      <span className="badge">{message.source}</span>
                    )}
                    {size != null && cached === size && (
                      <span className="badge badge-ok">cached</span>
                    )}
                    {size != null && cached > 0 && cached < size && (
                      <span className="badge badge-ok">{cached} cached</span>
                    )}
                    {turn?.thinking > 0 && (
                      <span className="badge">{turn.thinking} thinking tokens dropped</span>
                    )}
                    {removed.includes(`message ${message.index} `) && (
                      <span className="badge">template removed reasoning</span>
                    )}
                  </span>
                </span>
                <span className="ctx-size">{size ?? ''}</span>
              </li>
            )
          },
        )}
      </ol>
      {(report.dropped_messages || []).length > 0 && (
        <p className="note note-warn">
          {report.dropped_messages.length} message(s) dropped to fit the budget.
        </p>
      )}

      <details><summary>Growth by turn · history</summary><TurnHistory turns={turns} /></details>

      <details><summary>Exact tokens · green marks reused computation</summary><TokensView report={report} spans={spans} pieces={pieces} reused={reused} /></details>
      <details>
        <summary>Exact prompt text</summary>
        <pre className="code-block">{report.text}</pre>
      </details>
    </div>
  )
}
