import { useState } from 'react'
import { chipText } from './TokenChips.jsx'

export function ReuseExplainer() {
  const [example, setExample] = useState('prefix')
  const examples = {
    prefix: { request: ['My', ' name', ' is', ' Alice', '.', ' Hello'], saved: ['My', ' name', ' is', ' Alice', '.'], note: 'A saved state for the entire beginning can be restored. Only the new suffix needs input processing.' },
    edit: { request: ['My', ' name', ' is', ' Bob', '.', ' Hello'], saved: ['My', ' name', ' is', ' Alice', '.'], note: 'The first difference breaks this saved prefix. This model cannot rewind the longer state; a shorter checkpoint may still work.' },
    repeat: { request: ['Hello', ' Alice', ',', ' Alice'], saved: ['My', ' name', ' is', ' Alice', '.'], note: 'Alice appears in both, but at different positions after different beginnings. Repeated words do not create a reusable prefix.' },
  }
  const chosen = examples[example]
  let common = 0
  while (common < chosen.saved.length && chosen.saved[common] === chosen.request[common]) common += 1
  return (
    <div className="reuse-explainer">
      <p><strong>Reuse means skipping input computation.</strong> The full context is still sent. The server restores model state already computed for its matching beginning, then processes the remaining tokens and generates a fresh answer.</p>
      <details>
        <summary>How matching works · try an example</summary>
        <p>Token IDs must match in order, starting at position 0, with compatible model and cache settings. The same token elsewhere is insufficient. At least one input token is always processed.</p>
        <div className="button-row wrap">
          {Object.entries({ prefix: 'Append text', edit: 'Edit an early word', repeat: 'Repeat a word' }).map(([id, label]) => <button key={id} type="button" className={example === id ? 'active' : ''} onClick={() => setExample(id)}>{label}</button>)}
        </div>
        <p className="muted small">Illustrative pieces; live token IDs are shown in the comparison below.</p>
        <div className="example-line"><b>Context</b><span>{chosen.request.map((t, i) => <span key={i} className={`token-chip ${i < common ? 'tc-cached' : 'tc-added'}`}>{t.replace(/ /g, '␣')}</span>)}</span></div>
        <div className="example-line"><b>Cache labels</b><span>{chosen.saved.map((t, i) => <span key={i} className={`token-chip ${i < common ? 'tc-cached' : 'tc-removed'}`}>{t.replace(/ /g, '␣')}</span>)}</span></div>
        <p>{chosen.note}</p>
        <p>This Qwen3.5 model has recurrent state as well as attention keys and values. Its saved state cannot rewind to an earlier position. Checkpoints save usable stopping points. Cached state is separate from model weights and from the editable message text.</p>
        <p>Ordinary reuse saves work; it does not replay a stored answer. Sampling may still change the wording. Relabelling is a deliberate experiment: it changes token labels while leaving computed state unchanged, which can change the answer. Restoring one entry does not undo states later derived from it.</p>
        <a href="https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py" target="_blank" rel="noreferrer">MLX prompt-cache implementation</a>
      </details>
    </div>
  )
}

export default function CacheMatch({ report, reuseEnabled, running }) {
  const [selectedId, setSelectedId] = useState(null)
  const [wider, setWider] = useState(false)
  const entries = report?.cache_entries || []
  const prediction = report?.cache_prediction
  const selected = entries.find((e) => e.id === selectedId) || entries.find((e) => e.id === prediction?.entry_id) || entries[0]
  if (!report) return <p className="muted">Type a message to compare the current context with saved cache labels.</p>
  const total = report.token_ids?.length || 0
  const reused = prediction?.reused_tokens || 0
  return (
    <section className="cache-match" aria-label="Current context and cache comparison">
      <div className={`reuse-verdict ${reused ? 'note-ok' : 'note-warn'}`} aria-live="polite">
        <strong>{running ? 'Submitted request' : 'Next request'}: {reused} tokens of computation reusable · {total - reused} to process</strong>
        <div>{!reuseEnabled ? 'Reuse is OFF. Matching entries are still shown for comparison.' : prediction?.reason}</div>
        {reused > 0 && <div>Restore entry #{prediction.entry_id} for positions 0–{reused - 1}. Process positions {reused}–{total - 1}.</div>}
        {prediction?.relabeled && reused > 0 && <p className="note note-warn">This entry has changed token labels. Its computed state still represents the original text.</p>}
      </div>
      <div className="reuse-flow"><span>Full context sent <b>{total}</b></span><span>→ saved state restored <b>{reused}</b></span><span>→ input processed <b>{total - reused}</b></span><span>→ fresh answer</span></div>
      {selected ? <>
        <label className="match-select">Compare context ↔ cache entry
          <select aria-label="Cache entry to compare" value={selected.id} onChange={(e) => { setSelectedId(Number(e.target.value)); setWider(false) }}>
            {entries.map((entry) => <option key={entry.id} value={entry.id}>#{entry.id} · {entry.kind} · {entry.shared_prefix} matching · {entry.reusable_tokens} usable{prediction?.entry_id === entry.id && reused > 0 ? ' · chosen' : ''}</option>)}
          </select>
        </label>
        <p><strong>{selected.shared_prefix} leading token labels match</strong> · {selected.reusable_tokens} tokens of usable saved state{!reuseEnabled ? ' (disabled)' : ''}. {selected.reason}</p>
        <div className="table-scroll"><table className="data-table match-table">
          <caption>Exact IDs at the match boundary · positions start at 0</caption>
          <thead><tr><th>Position</th><th>Current context</th><th>Cache #{selected.id} labels</th><th>Comparison</th></tr></thead>
          <tbody>{(() => {
            const boundary = selected.shared_prefix
            const start = Math.max(0, boundary - (wider ? 5 : 2))
            const end = Math.min(Math.max(total, selected.tokens), boundary + (wider ? 8 : 3))
            const saved = new Map(selected.token_window.map((t) => [t.position, t]))
            return Array.from({ length: Math.max(0, end - start) }, (_, n) => {
              const pos = start + n
              const request = report.token_pieces?.[pos]
              const cache = saved.get(pos)
              const same = request && cache && request.id === cache.id
              const first = pos === boundary && boundary < Math.min(total, selected.tokens)
              const label = first ? '≠ first difference' : pos < boundary ? '✓ matching prefix' : same ? 'same ID, after break' : !cache ? 'new input' : !request ? 'cache continues' : 'different'
              const token = (t, empty) => t ? <><code>{chipText(t)}</code><small>ID {t.id}</small></> : <span className="muted">{empty}</span>
              return <tr key={pos} className={first ? 'match-break' : pos < boundary ? 'match-same' : ''}><td>{pos}</td><td>{token(request, 'request ends')}</td><td>{token(cache, 'saved state ends')}</td><td>{label}</td></tr>
            })
          })()}</tbody>
        </table></div>
        <button type="button" className="mini" onClick={() => setWider(!wider)}>{wider ? 'Show fewer tokens' : 'Show more around boundary'}</button>
        <p className="muted small">Matching labels describe the lookup key. Reuse restores the whole saved computation up to a usable boundary; later matching tokens cannot bridge a difference.</p>
      </> : <p className="muted">No saved states yet. Send a message or save a checkpoint from Edit context.</p>}
    </section>
  )
}
