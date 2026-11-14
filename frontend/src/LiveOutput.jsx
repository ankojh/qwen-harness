import { useEffect, useRef } from 'react'
import { deriveChannels } from './OutputPanel.jsx'

// The latest reply only: the message that produced it, the answer, and the
// thinking folded underneath.  Full token/raw/stats views live in Details.

function Scrolling({ text, className }) {
  const ref = useRef(null)
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [text])
  return <pre ref={ref} className={className}>{text}</pre>
}

export default function LiveOutput({ generation, tokens, running, lastInput }) {
  const channels = generation?.channels ?? deriveChannels(tokens)
  const count = tokens.length || generation?.tokens_generated || 0
  const thinkingTokens = (tokens.length ? tokens : generation?.tokens || []).filter(
    (token) => token.channel === 'thinking',
  ).length
  const decode = generation?.stats?.decode
  const finish = generation?.finish
  const thinkingOnly = running && !channels.answer && channels.thinking

  if (!running && !generation && tokens.length === 0) {
    return <p className="muted">The reply to your next message appears here.</p>
  }

  return (
    <div className="live-output">
      {lastInput && (
        <div className="you-said">
          <span className="role-chip role-user">you</span>
          <span>{lastInput}</span>
        </div>
      )}
      <div className="output-status">
        {running ? (
          <span className="badge badge-warn">{thinkingOnly ? 'thinking' : 'writing'} · {count} tokens</span>
        ) : finish ? (
          <span className={`badge ${finish.reason === 'eos' || finish.reason === 'stop_sequence' ? 'badge-ok' : 'badge-warn'}`}>
            {finish.reason}
          </span>
        ) : null}
        {!running && decode && (
          <span className="muted">
            {decode.generated_tokens} tokens · {decode.tokens_per_second} tok/s · {decode.seconds}s
          </span>
        )}
      </div>
      <Scrolling
        className="text-pane answer-pane"
        text={channels.answer?.trim() || (running ? '' : '(no answer text)')}
      />
      {channels.thinking && (
        <details open={thinkingOnly || undefined}>
          <summary>
            Thinking · {thinkingTokens} tokens{' '}
            <span className="muted">(not carried into the next turn's context)</span>
          </summary>
          <Scrolling className="text-pane thinking-pane" text={channels.thinking} />
        </details>
      )}
    </div>
  )
}
