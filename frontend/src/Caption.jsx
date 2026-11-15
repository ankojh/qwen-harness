// The one-line caption: what happens to this request, in words, with the
// live numbers.  Parts come from mapModel; each kind maps to a colour class
// in the same grammar as the map (green = restored, blue = computed, red =
// broken, warn = tampered).

const KIND_CLASS = { ok: 'g', info: 'b', bad: 'r', warn: 'w' }

export default function Caption({ model, extra = [], ariaLabel }) {
  const parts = [...model.caption, ...extra]
  return (
    <p className="caption" aria-live="polite" aria-label={ariaLabel}>
      {parts.map((part, index) => {
        if (part.kind === 'code') {
          return <code key={index}>{part.text}</code>
        }
        const cls = KIND_CLASS[part.kind]
        return cls ? (
          <span key={index} className={cls}>{part.text}</span>
        ) : (
          <span key={index}>{part.text}</span>
        )
      })}
    </p>
  )
}
