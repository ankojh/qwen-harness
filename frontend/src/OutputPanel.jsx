import { useEffect, useRef, useState } from 'react'

function AutoPre({ text, children, className }) {
  const ref = useRef(null)
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [text, children])
  return <pre ref={ref} className={className}>{children ?? text ?? ''}</pre>
}

function renderRaw(text, thinkStart, thinkEnd) {
  const markers = [thinkStart, thinkEnd].filter(Boolean)
  if (!text) return ''
  if (markers.length === 0) return text
  const pattern = new RegExp(`(${markers.join('|')})`, 'g')
  return text.split(pattern).map((part, index) =>
    markers.includes(part) ? (
      <span key={index} className="tok-special">{part}</span>
    ) : (
      <span key={index}>{part}</span>
    ),
  )
}

function formatBytes(value) {
  if (value === null || value === undefined) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let number = Number(value)
  let unit = 0
  while (number >= 1000 && unit < units.length - 1) {
    number /= 1000
    unit += 1
  }
  return `${number.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`
}

const FINISH_CLASS = {
  eos: 'ok',
  stop_sequence: 'ok',
  max_tokens: 'warn',
  cancelled: 'warn',
  context_overflow: 'error',
  error: 'error',
}

function deriveChannels(tokens) {
  const channels = { thinking: '', answer: '', raw: '' }
  for (const token of tokens) {
    channels.raw += token.text || ''
    const segments = token.segments?.length
      ? token.segments
      : token.channel && token.text
        ? [{ channel: token.channel, text: token.text }]
        : []
    for (const segment of segments) {
      channels[segment.channel] = (channels[segment.channel] || '') + segment.text
    }
  }
  return channels
}

export default function OutputPanel({
  generation,
  tokens,
  running,
  previewHash,
  modelLoaded,
  thinkStart,
  thinkEnd,
  onGenerate,
  onStop,
  onClearOutput,
  onClearConversation,
  onAppend,
}) {
  const [tab, setTab] = useState('answer')
  const [keepReasoning, setKeepReasoning] = useState(false)
  const channels = generation?.channels ?? deriveChannels(tokens)
  const tokenRows = tokens.length ? tokens : generation?.tokens || []
  const stats = generation?.stats
  const prefill = stats?.prefill
  const decode = stats?.decode
  const finish = generation?.finish
  const finishReason = finish?.reason
  const hashMatch =
    previewHash && generation?.prompt_hash
      ? previewHash === generation.prompt_hash
      : null
  const checkpoints = generation?.prompt?.checkpoints || []

  return (
    <div className="output-panel">
      <div className="output-head">
        <div className="button-row wrap">
          <button type="button" className="primary" disabled={!modelLoaded || running} onClick={onGenerate}>
            Generate
          </button>
          <button type="button" disabled={!running} onClick={onStop}>Stop</button>
          <button type="button" disabled={running || (!generation && tokenRows.length === 0)} onClick={onClearOutput}>
            Clear output
          </button>
          <button type="button" disabled={running} onClick={onClearConversation}>
            Clear conversation
          </button>
        </div>
        <div className="output-status">
          {generation ? (
            <>
              <span className={`badge finish-${FINISH_CLASS[finishReason] || 'warn'}`}>
                {finishReason || generation.status}
              </span>
              <span className="mono small">{generation.id}</span>
              <span className="muted">{tokenRows.length} tokens</span>
            </>
          ) : (
            <span className="muted">
              {running ? `streaming · ${tokenRows.length} tokens` : 'no output yet'}
            </span>
          )}
          {hashMatch !== null && (
            <span className={`badge ${hashMatch ? 'badge-ok' : 'badge-warn'}`}>
              {hashMatch ? 'inspector matches submitted prompt' : 'inspector differs from submitted prompt'}
            </span>
          )}
        </div>
      </div>

      {finish && (
        <div className="finish-line">
          <strong>{finish.reason}</strong>
          {finish.detail ? <span> · {finish.detail}</span> : null}
          {generation?.error ? <pre className="code-block error-block">{generation.error}</pre> : null}
        </div>
      )}

      <div className="append-row">
        <label className="check">
          <input
            type="checkbox"
            checked={keepReasoning}
            onChange={(event) => setKeepReasoning(event.target.checked)}
          />
          keep reasoning as reasoning_content when appending
        </label>
        <button type="button" disabled={!generation || running} onClick={() => onAppend(keepReasoning)}>
          Append to conversation
        </button>
      </div>

      <div className="tabs" role="tablist">
        {['thinking', 'answer', 'raw', 'tokens', 'stats'].map((name) => (
          <button
            key={name}
            type="button"
            role="tab"
            aria-selected={tab === name}
            className={tab === name ? 'active' : ''}
            onClick={() => setTab(name)}
          >
            {name}
          </button>
        ))}
      </div>

      <div className="tab-body">
        {tab === 'thinking' && (
          <>
            <p className="note">
              This is text the model generated between the think markers — ordinary sampled
              tokens. It is not a view of the model's internal computation (activations,
              attention), which is not observable here.
            </p>
            <AutoPre className="text-pane" text={channels.thinking} />
          </>
        )}

        {tab === 'answer' && (
          <>
            {!channels.answer && channels.thinking && (
              <p className="note note-warn">
                No answer text: the model did not emit the closing think marker within the
                token limit.
              </p>
            )}
            <AutoPre className="text-pane" text={channels.answer} />
          </>
        )}

        {tab === 'raw' && (
          <>
            <p className="note">Raw decoded stream. Think markers are shown and highlighted here; they are removed from the thinking and answer channels.</p>
            <AutoPre className="text-pane raw" text={channels.raw}>
              {renderRaw(channels.raw, thinkStart, thinkEnd)}
            </AutoPre>
          </>
        )}

        {tab === 'tokens' && (
          <div className="table-scroll tall">
            <table className="data-table token-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>id</th>
                  <th>piece</th>
                  <th>emitted</th>
                  <th>channel</th>
                  <th>logprob</th>
                  <th>top alternatives</th>
                </tr>
              </thead>
              <tbody>
                {tokenRows.map((token) => (
                  <tr key={token.index}>
                    <td>{token.index}</td>
                    <td>{token.id}</td>
                    <td className="mono">{JSON.stringify(token.piece)}</td>
                    <td className="mono">{JSON.stringify(token.text || '')}</td>
                    <td>{token.channel || '—'}</td>
                    <td>{token.logprob != null ? token.logprob.toFixed(3) : '—'}</td>
                    <td className="alternatives" title={(token.top_logprobs || []).map((item) => `${item.id} ${JSON.stringify(item.piece)} ${item.logprob}`).join('\n')}>
                      {(token.top_logprobs || []).slice(0, 3).map((item) => (
                        <span key={item.id} className="alt">
                          {item.id} {item.piece} <span className="muted">{item.logprob.toFixed(2)}</span>
                        </span>
                      ))}
                      {(token.top_logprobs || []).length > 3 ? <span className="muted">…</span> : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {tab === 'stats' && (
          <div className="stats">
            {!stats ? (
              <p className="muted">No stats yet.</p>
            ) : (
              <>
                <h4>Prefill</h4>
                <dl className="facts compact">
                  <div><dt>reused from cache</dt><dd>{prefill.reused_tokens} tokens</dd></div>
                  <div><dt>processed</dt><dd>{prefill.processed_tokens} tokens in {prefill.chunks} chunks</dd></div>
                  <div><dt>seconds</dt><dd>{prefill.seconds}s</dd></div>
                  <div><dt>rate</dt><dd>{prefill.tokens_per_second ?? '—'} tok/s</dd></div>
                </dl>
                <h4>Decode</h4>
                <dl className="facts compact">
                  <div><dt>generated</dt><dd>{decode.generated_tokens} tokens</dd></div>
                  <div><dt>seconds</dt><dd>{decode.seconds}s</dd></div>
                  <div><dt>rate</dt><dd>{decode.tokens_per_second ?? '—'} tok/s (first token excluded)</dd></div>
                  <div><dt>time to first token</dt><dd>{decode.time_to_first_token_seconds ?? '—'}s</dd></div>
                </dl>
                <h4>Memory</h4>
                <dl className="facts compact">
                  <div><dt>active</dt><dd>{formatBytes(stats.memory?.active_bytes)}</dd></div>
                  <div><dt>peak</dt><dd>{formatBytes(stats.memory?.peak_bytes)}</dd></div>
                  <div><dt>MLX buffer cache</dt><dd>{formatBytes(stats.memory?.cache_bytes)}</dd></div>
                  <div><dt>prompt cache</dt><dd>{formatBytes(stats.prompt_cache_bytes)}</dd></div>
                  <div><dt>forward calls (total)</dt><dd>{stats.forward_calls}</dd></div>
                  <div><dt>forward calls (this run)</dt><dd>{stats.forward_calls_this_run}</dd></div>
                </dl>
                <h4>Cache verification</h4>
                {generation.cache_verify ? (
                  <div className={`note ${generation.cache_verify.ok ? 'note-ok' : 'note-error'}`}>
                    expected offset {generation.cache_verify.expected_offset}; KV offsets{' '}
                    {JSON.stringify(generation.cache_verify.kv_offsets)}; ok ={' '}
                    {String(generation.cache_verify.ok)}
                  </div>
                ) : (
                  <p className="muted">No cache entry was stored for this run.</p>
                )}
                <h4>Prompt</h4>
                <dl className="facts compact">
                  <div><dt>hash</dt><dd className="mono small">{generation.prompt_hash || '—'}</dd></div>
                  <div><dt>checkpoint at</dt><dd>{generation.prompt?.checkpoint_at ?? '—'}</dd></div>
                  <div><dt>accepted checkpoints</dt><dd>{checkpoints.filter((entry) => entry.ok).length} of {checkpoints.length}</dd></div>
                  <div><dt>starts in thinking</dt><dd>{String(generation.prompt?.starts_in_thinking)}</dd></div>
                  <div><dt>budget</dt><dd>{JSON.stringify(generation.prompt?.budget)}</dd></div>
                  <div><dt>transformations</dt><dd>{generation.prompt?.transformations?.length ?? 0}</dd></div>
                </dl>
                <details>
                  <summary>Effective settings</summary>
                  <pre className="code-block">{JSON.stringify(generation.effective_settings, null, 2)}</pre>
                </details>
                <details>
                  <summary>Prompt report</summary>
                  <pre className="code-block">{JSON.stringify(generation.prompt, null, 2)}</pre>
                </details>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
