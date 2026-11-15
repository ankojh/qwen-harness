// The six-node strip: Messages → Tokens → Cache lookup → Compute → Generate
// → Saved.  Driven by the mapModel facts plus live SSE progress; the current
// stage lights up while a request runs.

function shortReason(reason) {
  if (!reason) return 'no match'
  const flat = reason.replace(/\s+/g, ' ').trim()
  if (flat.length > 46) return `${flat.slice(0, 45)}…`
  return flat
}

export default function Pipeline({ model, lookupKnown, running, savedSummary }) {
  const { total, reused, computed, totalProcessed, thinkCount, answerCount, messageCount, phase, historyThinking, historyAnswer } = model

  const lookupValue = lookupKnown === false && running
    ? 'looking…'
    : reused > 0
      ? `${reused} from #${model.chosenEntryId}`
      : reused === 0 && total === 0
        ? '—'
        : 'no match'

  const savedActive = savedSummary && (savedSummary.added > 0 || savedSummary.evicted > 0)
  const nodes = [
    {
      key: 'messages',
      label: 'Messages',
      value: total === 0 ? '0' : `${messageCount || '?'}`,
      sub: running ? 'this request' : 'in the conversation',
    },
    {
      key: 'tokens',
      label: 'Tokens',
      value: `${total}`,
      sub: 'full context sent',
    },
    {
      key: 'lookup',
      label: 'Cache lookup',
      value: lookupValue,
      sub: running && reused === 0 && model.lookup?.reason ? shortReason(model.lookup.reason) : 'reused prefix',
      tone: reused > 0 ? 'ok' : (running || total > 0 ? 'bad' : ''),
    },
    {
      key: 'compute',
      label: 'Compute',
      value: running ? `${totalProcessed} / ${computed}` : `${computed} tokens`,
      sub: running ? 'prefill' : 'predicted',
    },
    {
      key: 'generate',
      label: 'Generate',
      value: running ? `${thinkCount} thinking · ${answerCount} answer` : (phase === 'turn' ? `${historyThinking} thinking · ${historyAnswer} answer` : '—'),
      sub: 'thinking · answer',
    },
    {
      key: 'saved',
      label: 'Saved',
      value: savedActive
        ? `+${savedSummary.added}${savedSummary.evicted ? `, −${savedSummary.evicted} evicted` : ''}`
        : '—',
      sub: 'new cache entries',
      tone: savedActive ? 'ok' : '',
    },
  ]

  return (
    <div className="pipe" role="group" aria-label="What happens to a request">
      {nodes.map((node) => (
        <div key={node.key} className={`node ${phase === node.key ? 'hot' : ''} ${node.tone || ''}`}>
          <small>{node.label}</small>
          <b>{node.value}</b>
          <span>{node.sub}</span>
        </div>
      ))}
    </div>
  )
}
