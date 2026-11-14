import { useState } from 'react'

// One token chip strip shared by the Context and Cache panels.  Shows each
// token's decoded text (␣ for spaces, ⏎ for newlines), styles special tokens
// apart, colours by span role, and marks ranges (cached, divergence, added,
// removed).  Hover/focus shows position, id and the raw piece.  Very long
// sequences show the first and last 200 tokens with a "show all" control.

export function chipText(token) {
  return (token?.text || token?.piece || '').replace(/ /g, '␣').replace(/\n/g, '⏎')
}

export default function TokenChips({ tokens, spans, marks, offset = 0, limit = 400 }) {
  const [showAll, setShowAll] = useState(false)
  if (!tokens || tokens.length === 0) return null

  const total = tokens.length
  const truncated = !showAll && total > limit
  const half = Math.floor(limit / 2)

  // Position -> span / mark lookup tables (built once per render).
  const spanMap = new Map()
  if (spans) {
    for (const span of spans) {
      for (let position = span.start; position < span.end; position += 1) {
        spanMap.set(position, span)
      }
    }
  }
  const markMap = new Map()
  if (marks) {
    for (const mark of marks) {
      for (let position = mark.start; position < mark.end; position += 1) {
        if (!markMap.has(position)) markMap.set(position, mark)
      }
    }
  }

  const items = truncated
    ? [
        ...tokens.slice(0, half).map((token, index) => ({ token, position: index })),
        { gap: total - limit },
        ...tokens
          .slice(total - half)
          .map((token, index) => ({ token, position: total - half + index })),
      ]
    : tokens.map((token, index) => ({ token, position: index }))

  return (
    <span className="token-chips">
      {items.map((item, key) => {
        if (item.gap) {
          return (
            <button
              key={key}
              type="button"
              className="token-chip tc-gap"
              onClick={() => setShowAll(true)}
            >
              ⋯ {item.gap} more
            </button>
          )
        }
        const { token, position } = item
        const span = spanMap.get(position) || null
        const mark = markMap.get(position) || null
        const classes = ['token-chip']
        if (token.special) classes.push('tc-special')
        if (span) {
          if (span.kind === 'opener') classes.push('tc-template')
          else if (span.role) classes.push(`tc-${span.role}`)
        }
        if (mark) classes.push(`tc-${mark.kind}`)
        return (
          <span
            key={key}
            className={classes.join(' ')}
            tabIndex={0}
            title={`position ${position + offset} · id ${token.id} · piece ${JSON.stringify(token.piece)}`}
          >
            {chipText(token)}
          </span>
        )
      })}
    </span>
  )
}
