// Split decoded prompt text into spans, styling special tokens (<|...|>) and
// the model's think markers apart.  Shared by the prompt inspector and the
// cache diff.

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

export function highlightSpecial(text, thinkStart, thinkEnd) {
  const markers = [thinkStart, thinkEnd].filter(Boolean).map(escapeRegExp)
  const pattern = new RegExp(
    `(<\\|[^|]*\\|>${markers.length ? `|${markers.join('|')}` : ''})`,
    'g',
  )
  return text
    .split(pattern)
    .filter((part) => part !== '')
    .map((part, index) => {
      const special =
        part.startsWith('<|') || part === thinkStart || part === thinkEnd
      return special ? (
        <span key={index} className="tok-special">{part}</span>
      ) : (
        <span key={index}>{part}</span>
      )
    })
}
