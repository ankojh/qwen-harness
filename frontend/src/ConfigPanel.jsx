import { useEffect, useState } from 'react'
import Field from './Field.jsx'

const GROUPS = [
  { id: 'model', label: 'Model (load-time)' },
  { id: 'prompt', label: 'Prompt & template' },
  { id: 'context', label: 'Context & stopping' },
  { id: 'sampling', label: 'Sampling' },
  { id: 'runtime', label: 'Runtime, KV cache & prompt cache' },
]

export function formatBytes(bytes) {
  if (bytes === null || bytes === undefined) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let value = Number(bytes)
  let unit = 0
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000
    unit += 1
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`
}

function megabytes(bytes) {
  if (bytes === null || bytes === undefined) return '—'
  return (Number(bytes) / 1e6).toFixed(1)
}

function ModelSection({ state, busy, onModelAction }) {
  const info = state.model
  const loaded = state.model_loaded
  const mix = info?.layer_mix || {}
  const layerMix = Object.entries(mix)
    .map(([name, count]) => `${count} × ${name}`)
    .join(', ')

  return (
    <section className="model-section">
      <div className="model-head">
        <div>
          <span className={`status-dot ${loaded ? 'ok' : 'off'}`} />
          <strong>{loaded ? 'Model loaded' : 'No model loaded'}</strong>
          {loaded && info && (
            <span className="muted">
              {' '}
              · {info.path} · load {info.load_seconds}s
            </span>
          )}
        </div>
        <div className="button-row">
          <button type="button" disabled={busy || loaded} onClick={() => onModelAction('load')}>
            Load
          </button>
          <button type="button" disabled={busy || !loaded} onClick={() => onModelAction('unload')}>
            Unload
          </button>
          <button type="button" disabled={busy || !loaded} onClick={() => onModelAction('reload')}>
            Reload
          </button>
        </div>
      </div>

      {state.model_stale && (
        <div className="note note-warn">
          Model settings changed since this model was loaded. Use Reload to apply them.
        </div>
      )}

      {loaded && info && (
        <>
          <dl className="facts">
            <div><dt>Architecture</dt><dd>{info.architecture || '—'}</dd></div>
            <div><dt>Model type</dt><dd>{info.model_type || '—'}</dd></div>
            <div><dt>Layers</dt><dd>{info.num_layers} ({layerMix || '—'})</dd></div>
            <div><dt>Hidden / vocab</dt><dd>{info.hidden_size} / {info.vocab_size}</dd></div>
            <div><dt>Context length</dt><dd>{info.max_position_embeddings}</dd></div>
            <div><dt>Weight bytes</dt><dd>{formatBytes(info.weight_bytes)}</dd></div>
            <div><dt>Weight quantization</dt><dd>{info.quantization ? JSON.stringify(info.quantization) : 'none'}</dd></div>
            <div><dt>Cache layers</dt><dd>{JSON.stringify(info.cache_types)}</dd></div>
            <div><dt>Cache trimmable</dt><dd>{String(info.cache_trimmable)}</dd></div>
            <div><dt>EOS ids</dt><dd>{JSON.stringify(info.eos_token_ids)}</dd></div>
            <div><dt>Thinking markers</dt><dd>{info.has_thinking ? `${info.think_start} … ${info.think_end}` : 'none'}</dd></div>
            <div><dt>Detokenizer</dt><dd>{info.detokenizer}</dd></div>
            <div className="wide"><dt>Resolved path</dt><dd className="break">{info.resolved_path || '—'}</dd></div>
            <div className="wide"><dt>Chat template reads</dt><dd>{info.chat_template_variables?.join(', ') || '—'}</dd></div>
          </dl>
          <p className="note">
            Weight quantization is baked into the downloaded weights. To change it, load a
            different repo (for example an 8-bit or bf16 variant).
          </p>
          <details>
            <summary>Model chat template (read-only)</summary>
            <pre className="code-block">{info.chat_template || '(none)'}</pre>
          </details>
        </>
      )}
    </section>
  )
}

function CacheSection({ state, busy, onClearCache }) {
  const cache = state.cache || { entries: [] }
  const entries = cache.entries || []
  const limit = cache.limit_bytes

  return (
    <details className="group" open>
      <summary>
        Prompt cache (live) <span className="muted">({entries.length} entries)</span>
      </summary>
      <div className="cache-totals">
        <span>accounted <strong>{formatBytes(cache.accounted_bytes)}</strong></span>
        <span>recurrent {formatBytes(cache.arrays_bytes)}</span>
        <span>KV {formatBytes(cache.kv_bytes)}</span>
        <span>limit {limit ? formatBytes(limit) : '—'}</span>
        <span>used {cache.utilization != null ? `${(cache.utilization * 100).toFixed(1)}%` : '—'}</span>
        <span>evictions {cache.evictions_total ?? 0}</span>
        <span>MLX active {formatBytes(cache.active_bytes)}</span>
        <span>
          MLX buffer-cache limit {formatBytes(state.memory?.mlx_cache_limit_bytes)}
          {state.memory?.mlx_cache_limit_bytes === state.memory?.mlx_default_cache_limit_bytes
            ? ' (MLX default)'
            : ''}
        </span>
      </div>

      {entries.length === 0 ? (
        <p className="muted">No cache entries. Generate with cache reuse on to store one.</p>
      ) : (
        <div className="table-scroll">
          <table className="data-table cache-table">
            <thead>
              <tr>
                <th>#</th>
                <th>kind</th>
                <th>after msg</th>
                <th>tokens</th>
                <th>recurrent</th>
                <th>KV</th>
                <th>total</th>
                <th>age</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={entry.id}>
                  <td>{entry.id}</td>
                  <td>{entry.kind}</td>
                  <td>{entry.after_message ?? '—'}</td>
                  <td>{entry.tokens}</td>
                  <td>{megabytes(entry.arrays_bytes)} MB</td>
                  <td>{megabytes(entry.kv_bytes)} MB</td>
                  <td>{megabytes(entry.nbytes)} MB</td>
                  <td>{entry.age_seconds}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="note">{cache.note}</p>
      <div className="cache-foot">
        <span className="muted">layer types: {JSON.stringify(cache.layer_types || {})}</span>
        <button type="button" className="mini" disabled={busy || entries.length === 0} onClick={onClearCache}>
          Clear prompt cache
        </button>
      </div>
      {cache.last_lookup && (
        <details>
          <summary>Last lookup</summary>
          <pre className="code-block">{JSON.stringify(cache.last_lookup, null, 2)}</pre>
        </details>
      )}
    </details>
  )
}

export default function ConfigPanel({
  state,
  busy,
  error,
  onApply,
  onReset,
  onPreset,
  onModelAction,
  onClearCache,
}) {
  const [advanced, setAdvanced] = useState(() => JSON.stringify(state.config, null, 2))
  const [advancedError, setAdvancedError] = useState(null)

  useEffect(() => {
    setAdvanced(JSON.stringify(state.config, null, 2))
  }, [state.config])

  function exportConfig() {
    const blob = new Blob([JSON.stringify(state.config, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'qwen-harness-config.json'
    link.click()
    URL.revokeObjectURL(url)
  }

  async function importConfig(event) {
    const file = event.target.files?.[0]
    if (!file) return
    try {
      const parsed = JSON.parse(await file.text())
      setAdvancedError(null)
      onApply(parsed)
    } catch (importError) {
      setAdvancedError(`import failed: ${importError.message}`)
    } finally {
      event.target.value = ''
    }
  }

  function saveAdvanced() {
    try {
      const parsed = JSON.parse(advanced)
      setAdvancedError(null)
      onApply(parsed)
    } catch (parseError) {
      setAdvancedError(`invalid JSON: ${parseError.message}`)
    }
  }

  return (
    <div className="config-panel">
      <ModelSection state={state} busy={busy} onModelAction={onModelAction} />

      {error && <div className="note note-error">{error}</div>}

      {GROUPS.map((group) => {
        const fields = state.fields.filter((field) => field.group === group.id)
        return (
          <details className="group" open key={group.id}>
            <summary>
              {group.label} <span className="muted">({fields.length})</span>
            </summary>
            <div className="fields">
              {fields.map((field) => (
                <Field
                  key={field.name}
                  field={field}
                  value={state.config[field.name]}
                  defaultValue={state.defaults[field.name]}
                  capability={state.capabilities[field.name]}
                  busy={busy}
                  onApply={(value) => onApply({ [field.name]: value })}
                />
              ))}
            </div>
          </details>
        )
      })}

      <CacheSection state={state} busy={busy} onClearCache={onClearCache} />

      <section className="tools">
        <h3>Presets</h3>
        <div className="button-row wrap">
          {state.presets.map((preset) => (
            <button
              key={preset.id}
              type="button"
              title={preset.description}
              disabled={busy}
              onClick={() => onPreset(preset)}
            >
              {preset.name}
            </button>
          ))}
        </div>
        <div className="button-row wrap">
          <button type="button" disabled={busy} onClick={exportConfig}>Export JSON</button>
          <label className={`file-button ${busy ? 'disabled' : ''}`}>
            Import JSON
            <input type="file" accept="application/json,.json" disabled={busy} onChange={importConfig} />
          </label>
          <button type="button" disabled={busy} onClick={onReset}>Reset to defaults</button>
        </div>
      </section>

      <details className="group">
        <summary>Advanced JSON editor</summary>
        <p className="note">
          Validated by the server through the same PUT /api/config path as the controls.
        </p>
        <textarea
          className="json-editor"
          rows={16}
          spellCheck={false}
          value={advanced}
          onChange={(event) => setAdvanced(event.target.value)}
        />
        <div className="button-row">
          <button type="button" disabled={busy} onClick={saveAdvanced}>Apply JSON</button>
          <button
            type="button"
            className="mini"
            onClick={() => {
              setAdvanced(JSON.stringify(state.config, null, 2))
              setAdvancedError(null)
            }}
          >
            Revert edits
          </button>
        </div>
        {advancedError && <div className="note note-error">{advancedError}</div>}
      </details>
    </div>
  )
}
