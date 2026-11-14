import { useEffect, useState } from 'react'

const APPLIES_LABEL = {
  immediate: 'immediate',
  next_gen: 'next generation',
  reload: 'reload',
  unsupported: 'unsupported',
}

function display(value) {
  if (value === null) return 'null'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function toDraft(value, field) {
  if (value === null || value === undefined) return ''
  if (field.kind === 'json' || field.type === 'object' || field.type === 'array') {
    return JSON.stringify(value, null, 2)
  }
  if (field.kind === 'lines') return (value || []).join('\n')
  if (field.kind === 'int_list') return (value || []).join(', ')
  return String(value)
}

// Parse a text draft into the JSON type the server expects.  Errors stay local
// until the user fixes them; the server validates everything again.
function parseDraft(draft, field) {
  const kind = field.kind
  const trimmed = typeof draft === 'string' ? draft.trim() : draft
  if (kind === 'json' || field.type === 'object' || field.type === 'array') {
    if (trimmed === '') {
      if (field.nullable) return null
      throw new Error('JSON is required')
    }
    return JSON.parse(draft)
  }
  if (kind === 'lines') {
    return draft
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
  }
  if (kind === 'int_list') {
    if (trimmed === '') return []
    if (trimmed.startsWith('[')) return JSON.parse(trimmed).map(Number)
    return trimmed
      .split(/[,\s]+/)
      .filter(Boolean)
      .map((item) => {
        const number = Number(item)
        if (!Number.isInteger(number)) throw new Error(`"${item}" is not an integer`)
        return number
      })
  }
  if (field.type === 'integer' || field.type === 'number') {
    if (trimmed === '') {
      if (field.nullable) return null
      throw new Error('a number is required')
    }
    const number = Number(trimmed)
    if (!Number.isFinite(number)) throw new Error('not a number')
    if (field.type === 'integer' && !Number.isInteger(number)) {
      throw new Error('must be an integer')
    }
    return number
  }
  if (field.nullable && trimmed === '') return null
  return draft
}

function isTextarea(field) {
  return (
    field.kind === 'json' ||
    field.kind === 'lines' ||
    field.kind === 'int_list' ||
    field.kind === 'textarea' ||
    field.kind === 'textarea_nullable'
  )
}

export default function Field({
  field,
  value,
  defaultValue,
  capability,
  busy,
  onApply,
}) {
  const [draft, setDraft] = useState(() => toDraft(value, field))
  const [localError, setLocalError] = useState(null)

  // Re-sync the draft only when the effective value really changes.  /api/state
  // returns fresh objects on every refresh, so depending on `value`/`field`
  // identity would wipe a half-typed edit.
  const valueKey = JSON.stringify(value)
  useEffect(() => {
    setDraft(toDraft(value, field))
    setLocalError(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [valueKey, field.name])

  const unsupported = capability && capability.supported === false
  const inactive = capability && capability.inactive
  const warning = capability && capability.warning
  const disabled = Boolean(busy || unsupported)
  const dirty = display(value) !== display(defaultValue)

  function commit() {
    try {
      const parsed = parseDraft(draft, field)
      setLocalError(null)
      onApply(parsed)
    } catch (error) {
      setLocalError(error.message)
    }
  }

  const common = {
    disabled,
    onBlur: commit,
    onKeyDown: (event) => {
      if (event.key === 'Enter' && !isTextarea(field)) {
        event.currentTarget.blur()
      }
    },
  }

  let control
  if (field.type === 'boolean') {
    control = (
      <label className="check">
        <input
          type="checkbox"
          checked={Boolean(value)}
          disabled={disabled}
          onChange={(event) => onApply(event.target.checked)}
        />
        <span>{value ? 'on' : 'off'}</span>
      </label>
    )
  } else if (field.kind === 'select') {
    const options = field.options || []
    const index = options.findIndex((option) => option === value)
    control = (
      <select
        value={index >= 0 ? String(index) : ''}
        disabled={disabled}
        onChange={(event) => onApply(options[Number(event.target.value)])}
      >
        {options.map((option, position) => (
          <option key={position} value={String(position)}>
            {option === null ? 'None (off)' : String(option)}
          </option>
        ))}
      </select>
    )
  } else if (isTextarea(field)) {
    control = (
      <textarea
        {...common}
        rows={field.kind === 'json' ? 5 : 3}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        placeholder={field.nullable ? 'null' : ''}
        spellCheck={false}
      />
    )
  } else if (field.type === 'integer' || field.type === 'number') {
    control = (
      <input
        {...common}
        type="number"
        step={field.type === 'integer' ? '1' : 'any'}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        placeholder={field.nullable ? 'null' : ''}
      />
    )
  } else {
    control = (
      <input
        {...common}
        type="text"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        placeholder={field.nullable ? 'null' : ''}
      />
    )
  }

  return (
    <div className={`field ${unsupported ? 'field-disabled' : ''}`}>
      <div className="field-head">
        <span className="field-label">{field.label}</span>
        <span className={`badge applies-${field.applies}`}>
          {APPLIES_LABEL[field.applies] || field.applies}
        </span>
        {inactive && <span className="badge badge-warn">inactive</span>}
        {unsupported && <span className="badge badge-off">disabled</span>}
      </div>

      <div className="field-control">
        {control}
        <button
          type="button"
          className="mini"
          disabled={disabled || !dirty}
          title={`Reset to default: ${display(defaultValue)}`}
          onClick={() => onApply(defaultValue)}
        >
          reset
        </button>
      </div>

      <div className="field-purpose">{field.purpose}</div>
      <div className="field-tradeoff">{field.tradeoff}</div>
      <div className="field-meta">
        <span>default: {display(defaultValue)}</span>
        {field.library_default && <span>mlx-lm: {field.library_default}</span>}
      </div>

      {unsupported && capability.reason && (
        <div className="note note-off">{capability.reason}</div>
      )}
      {!unsupported && inactive && capability.reason && (
        <div className="note note-warn">inactive now: {capability.reason}</div>
      )}
      {warning && <div className="note note-warn">{warning}</div>}
      {localError && <div className="note note-error">{localError}</div>}
    </div>
  )
}
