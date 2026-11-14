import { useState } from 'react'
import TokenChips from './TokenChips.jsx'
import { diffContexts, messageOf } from './turnDiffs.js'

// The context: what the viewed request sends, message by message.  Three
// views: Overview (the summary as before), Tokens (every token as a chip,
// grouped by message, plus a per-message word/token table), Changes (what
// this request changed versus the previous turn).  A historical turn shows
// its recorded snapshot; "Next" shows the live tokenizer-only preview.

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
        const cachedStart = Math.max(0, Math.min(reused, span.end) - span.start)
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

function ChangesView({ report, prevTurn }) {
  if (!prevTurn || !prevTurn.context) {
    return <p className="muted">First turn — nothing to compare.</p>
  }
  const diff = diffContexts(prevTurn.context, report, prevTurn.thinking || 0)
  return (
    <div className="change-list">
      <p className="muted">
        Common prefix {diff.commonPrefix} tokens ·{' '}
        {diff.prevTokens} → {diff.nextTokens} tokens (net {diff.netChange >= 0 ? '+' : ''}{diff.netChange}) ·{' '}
        thinking from turn {prevTurn.n} not carried: {diff.thinkingNotCarried} tokens
      </p>
      {diff.rows.map((row, key) => (
        <DiffRow key={key} row={row} report={report} prevReport={prevTurn.context} />
      ))}
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
  viewTurn,
  prevTurn,
}) {
  const [view, setView] = useState('overview')
  const historical = Boolean(viewTurn)
  const report = historical ? viewTurn.context : preview

  if (!modelLoaded) return <p className="muted">Load the model to see the context.</p>
  if (historical && !report) return <p className="muted">This turn recorded no context.</p>
  if (!historical && session.messages.length === 0 && !draft.trim()) {
    return (
      <p className="muted">
        Empty. The conversation lives in this browser tab and is sent in full with every
        request; start typing to see what the first request will contain.
      </p>
    )
  }
  if (previewError && !historical) return <p className="note note-error">{previewError}</p>
  if (!report) return <p className="muted">Measuring…</p>

  const pieces = report.token_pieces || []
  const spans = report.spans || []
  const budget = report.budget || {}
  const total = budget.input_tokens ?? report.token_ids?.length ?? 0
  const reused = historical
    ? viewTurn.reused || 0
    : report.cache_prediction?.reused_tokens || 0
  const draftIndex = !historical && draft.trim() ? session.messages.length + 1 : null
  const assistantTurns = []
  ;(report.submitted_messages || []).forEach((message) => {
    if (message.role === 'assistant') assistantTurns.push(message.index)
  })
  const removed = (report.transformations || []).join(' ')
  const pct = (value) => `${(value / (budget.context_budget || 1)) * 100}%`
  const subject = historical ? `turn ${viewTurn.n}'s request` : 'the next request'

  const tabs = (
    <div className="tabs" role="tablist">
      {[
        ['overview', 'Overview'],
        ['tokens', 'Tokens'],
        ['changes', 'Changes'],
      ].map(([id, label]) => (
        <button
          key={id}
          type="button"
          role="tab"
          className={view === id ? 'active' : ''}
          onClick={() => setView(id)}
        >
          {label}
        </button>
      ))}
    </div>
  )

  if (view === 'tokens') {
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

  if (view === 'changes') {
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
        <ChangesView report={report} prevTurn={prevTurn} />
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
                  {oneLine(message.content) || <span className="muted">(empty)</span>}
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

      <TurnHistory turns={turns} />

      <details>
        <summary>Exact prompt text</summary>
        <pre className="code-block">{report.text}</pre>
      </details>
    </div>
  )
}
