import { useState } from 'react'

// One token chip strip shared by the Context and Cache panels.  Shows each
// token's decoded text (␣ for spaces, ⏎ for newlines), styles special tokens
// apart, colours by span role, and marks ranges (cached, divergence, added,
// removed).  Hover/focus shows position, id and the raw piece.  Very long
// sequences show the first and last 200 tokens with a "show all" control.
//
// ``tampered`` lists positions whose token was relabelled: the chip shows in
// red and its tooltip names the original piece.  ``onTokenClick`` turns the
// chips into buttons (used by the Cache panel to open the relabel form).

export function chipText(token) {
  return (token?.text || token?.piece || '').replace(/ /g, '␣').replace(/\n/g, '⏎')
}

export default function TokenChips({
  tokens,
  spans,
  marks,
  tampered,
  onTokenClick,
  offset = 0,
  limit = 400,
}) {
  const [showAll, setShowAll] = useState(false)
  if (!tokens || tokens.length === 0) return null

  const total = tokens.length
  const truncated = !showAll && total > limit
  const half = Math.floor(limit / 2)

  // Position -> span / mark / tamper lookup tables (built once per render).
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
  const tamperedMap = new Map()
  if (tampered) {
    for (const item of tampered) tamperedMap.set(item.position, item)
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
        const absolute = position + offset
        const span = spanMap.get(position) || null
        const mark = markMap.get(position) || null
        const relabel = tamperedMap.get(position) || null
        const classes = ['token-chip']
        if (token.special) classes.push('tc-special')
        if (span) {
          if (span.kind === 'opener') classes.push('tc-template')
          else if (span.role) classes.push(`tc-${span.role}`)
        }
        if (mark) classes.push(`tc-${mark.kind}`)
        if (relabel) classes.push('tc-tampered')
        if (onTokenClick) classes.push('tc-clickable')
        let title = `position ${absolute} · id ${token.id} · piece ${JSON.stringify(token.piece)}`
        if (relabel) {
          title += ` · relabelled: was ${JSON.stringify(relabel.original_piece)} (id ${relabel.original_id})`
          if (onTokenClick) title += ' · click to relabel again'
        } else if (onTokenClick) {
          title += ' · click to relabel'
        }
        const inner = <>{chipText(token)}</>
        return onTokenClick ? (
          <button
            key={key}
            type="button"
            className={classes.join(' ')}
            title={title}
            onClick={() => onTokenClick(absolute)}
          >
            {inner}
          </button>
        ) : (
          <span key={key} className={classes.join(' ')} tabIndex={0} title={title}>
            {inner}
          </span>
        )
      })}
    </span>
  )
}
