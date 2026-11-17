import { chipText } from './TokenChips.jsx'
import { kindLabel } from './mapModel.js'

// The lens: aligned chip rows (positions, context, one entry's labels)
// around the break.  The window centres on the selected entry's break — or
// the reuse boundary when nothing is selected — and holds at most about 60
// chips.  Click a token on the entry row to relabel it; click a context
// token to inspect it.  ``entryPieces`` carries the entry's decoded labels
// (from the entry detail when live and idle, or the preview's token window);
// where they are unknown the chip shows the raw token id.
//
// ``relabelPositions`` marks tampered positions.

const SPECIAL = new Set(['<|im_start|>', '<|im_end|>', '<think>', '</think>'])
const HALF = 26 // chips each side of the centre

export default function Lens({
  model,
  selectedEntryId,
  entryPieces = [],
  relabelPositions = [],
  onContextTokenClick,
  onEntryTokenClick,
}) {
  const entries = model.rows.filter((row) => !row.ghost)
  const entry = entries.find((row) => row.id === selectedEntryId) || null
  const centre = entry ? entry.breakAt ?? entry.len : model.lensCenter
  const contextLen = model.total
  const maxLen = Math.max(contextLen, entry ? entry.len : 0, 1)
  const start = Math.max(0, centre - HALF)
  const end = Math.min(maxLen + 1, centre + HALF)

  const pieceById = new Map()
  for (const token of entryPieces || []) {
    if (!pieceById.has(token.position)) pieceById.set(token.position, token)
  }

  const heading = entry
    ? entry.breakAt != null
      ? `Lens: entry #${entry.id} breaks at position ${entry.breakAt}`
      : `Lens: entry #${entry.id} matches all ${entry.len} of its tokens`
    : entries.length > 0
      ? 'Lens: no bar selected — click a bar in the map'
      : 'Lens'

  const positionChips = []
  const contextChips = []
  const entryChips = []
  for (let position = start; position < end; position += 1) {
    positionChips.push(
      <span key={position} className="chip pos">{position}</span>,
    )
    const contextToken = model.pieces[position]
    const entryToken = entry ? entry.ids[position] : undefined
    const pieceToken = entry ? pieceById.get(position) : null
    const relabeled = relabelPositions.includes(position)
    const base = (tokenPresent) => {
      if (!entry || !tokenPresent) return 'none'
      if (position < entry.match) return 'same'
      if (position === entry.breakAt) return 'diff'
      return 'after'
    }
    contextChips.push(
      <span
        key={position}
        className={`chip ${base(contextToken != null)} ${SPECIAL.has(contextToken?.piece) ? 'spec' : ''} ${onContextTokenClick ? 'clickable' : ''}`}
        title={`context position ${position}${contextToken ? ` · id ${contextToken.id} · piece ${JSON.stringify(contextToken.piece)}` : ''}`}
        onClick={contextToken && onContextTokenClick ? () => onContextTokenClick(position) : undefined}
      >
        {contextToken == null ? 'end' : chipText(contextToken)}
      </span>,
    )
    let entryLabel = 'end'
    if (pieceToken) entryLabel = chipText(pieceToken)
    else if (entryToken != null) entryLabel = `⟨${entryToken}⟩`
    entryChips.push(
      <span
        key={position}
        className={`chip ${base(entryToken != null)} ${SPECIAL.has(pieceToken?.piece) ? 'spec' : ''} ${relabeled ? 'relab' : ''} ${entry && entryToken != null && onEntryTokenClick ? 'clickable' : ''}`}
        title={relabeled
          ? `relabelled at position ${position} · click to relabel again`
          : entry && entryToken != null && onEntryTokenClick
            ? `entry #${entry.id} position ${position} · click to relabel this token`
            : undefined}
        onClick={entry && entryToken != null && onEntryTokenClick ? () => onEntryTokenClick(position) : undefined}
      >
        {entryLabel}
      </span>,
    )
  }

  return (
    <div className="lens" aria-label="Lens: tokens around the break">
      <span className="lab">Position</span>
      <div className="chips">{positionChips}</div>
      <span className="lab">Context</span>
      <div className="chips">{contextChips}</div>
      {entry && (
        <>
          <span className="lab">#{entry.id} {kindLabel(entry.kind)}</span>
          <div className="chips">{entryChips}</div>
        </>
      )}
      {!entry && (
        <>
          <span className="lab">cache</span>
          <div className="chips">
            <span className="chip none">{entries.length > 0 ? 'select a bar to compare' : 'nothing saved yet'}</span>
          </div>
        </>
      )}
      <span className="lab" style={{ gridColumn: '1 / -1' }}>
        {heading} · drag the map to move the window
      </span>
    </div>
  )
}
