import { useLayoutEffect, useRef, useState } from 'react'
import { kindLabel } from './mapModel.js'

// The centrepiece: one token-position ruler with the context tape, the
// restored/computed band and one bar per cache entry.  Pure SVG, sized by a
// ResizeObserver; one rect per span (not per token) when the tokens outnumber
// the pixels, and tiny breaks stay at least 2 px wide.  Bars are sorted by
// length (done in mapModel); at most ``maxRows`` are drawn and the rest fold
// into a "+N more" row.

const GUTTER = 118      // left label column
const PAD = 14          // right padding
const CELL_MIN = 2      // below this many px per token, draw spans not tokens
const LABEL_MIN = 15    // below this, no token text inside cells
const SPECIAL = new Set(['<|im_start|>', '<|im_end|>', '<think>', '</think>'])

function shown(text) {
  return (text || '').replace(/\n/g, '⏎').replace(/ /g, '␣')
}

function abbr(text, width) {
  const value = shown(text)
    .replace('<|im_start|>', '⟨s⟩')
    .replace('<|im_end|>', '⟨/s⟩')
    .replace('assistant', 'asst')
  const max = Math.max(1, Math.floor(width / 6))
  return value.length > max ? `${value.slice(0, max - 1)}…` : value
}

const ROLE_CLASS = { user: 'm-user', assistant: 'm-asst', system: 'm-sys', template: 'm-tpl' }

function cellTitle(position, piece, id) {
  return `position ${position} · ${shown(piece || '')}${id != null ? ` · id ${id}` : ''}`
}

export default function TapeMap({
  model,
  selectedEntryId,
  selectedToken,
  highlightMessage,
  onSelectEntry,
  onSelectToken,
  onSelectMessage,
  onDragLens,
  maxRows = 8,
  pulseEntryId = null,
}) {
  const wrapRef = useRef(null)
  const [width, setWidth] = useState(900)

  useLayoutEffect(() => {
    const el = wrapRef.current
    if (!el) return undefined
    const measure = () => setWidth(Math.max(el.clientWidth || 900, 560))
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const drag = useRef(null)

  const { tape, generatedTail, rows, ghosts, ticks, domain, rulerStep, total, reused, computed, totalProcessed, pieces, lensCenter } = model
  const plot = Math.max(40, width - GUTTER - PAD)
  const x = (position) => GUTTER + (position / domain) * plot
  const cellWidth = plot / domain
  const perToken = cellWidth >= CELL_MIN

  // Tape rows
  const tapeY = 30
  const tapeH = 26
  const bandY = tapeY + tapeH + 8
  const bandH = 10
  const bandBottom = bandY + bandH
  let y = bandBottom + 34

  const rowsShown = rows.slice(0, maxRows)
  const folded = rows.slice(maxRows)
  const chosenFolded = folded.some((row) => row.chosen)
  const barH = 18

  // Which span a message index highlights (chat bubble hover).
  const hlSpan = tape.find((seg) => seg.messageIndex === highlightMessage) || null

  // The lens window marker (a faint guide line at the centre).
  const lensInDomain = lensCenter >= 0 && lensCenter <= domain

  function beginDrag(event, fallback) {
    const startX = event.clientX
    drag.current = { startX, moved: false, fallback }
    const move = (ev) => {
      if (!drag.current) return
      if (Math.abs(ev.clientX - startX) > 4) drag.current.moved = true
    }
    const up = (ev) => {
      if (!drag.current) return
      const wasDrag = drag.current.moved
      const startPosition = fallback
      const endX = ev.clientX
      drag.current = null
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
      if (wasDrag) {
        // Drag the lens window to the released position.
        const rect = wrapRef.current.getBoundingClientRect()
        const rel = endX - rect.left
        const position = Math.max(0, Math.round(((rel - GUTTER) / plot) * domain))
        onDragLens?.(position)
      } else {
        onSelectMessage?.(startPosition)
      }
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }

  const tapeCells = []
  if (perToken) {
    for (const seg of tape) {
      for (let position = seg.start; position < seg.end; position += 1) {
        const piece = pieces[position]
        const isSelected = selectedToken === position
        tapeCells.push(
          <g
            key={position}
            className="strip"
            onClick={() => onSelectToken?.(position)}
            style={{ cursor: 'pointer' }}
          >
            <rect
              className={`${ROLE_CLASS[seg.role] || 'm-tpl'} ${isSelected ? 'm-sel' : ''}`}
              x={x(position) + 0.5}
              y={tapeY}
              width={Math.max(cellWidth - 1, 1)}
              height={tapeH}
              rx={2}
            >
              <title>{cellTitle(position, piece?.piece, piece?.id)}</title>
            </rect>
            {cellWidth > LABEL_MIN && (
              <text
                x={x(position) + cellWidth / 2}
                y={tapeY + 17}
                textAnchor="middle"
                fontSize={cellWidth > 26 ? 10 : 8}
                className={SPECIAL.has(piece?.piece) ? 'm-mut' : undefined}
                pointerEvents="none"
              >
                {abbr(piece?.piece || '', cellWidth)}
              </text>
            )}
          </g>,
        )
      }
    }
  } else {
    for (const seg of tape) {
      tapeCells.push(
        <g
          key={`seg-${seg.start}`}
          className="strip"
          onMouseDown={(event) => beginDrag(event, seg.messageIndex ?? seg.start)}
          onClick={() => onSelectMessage?.(seg.messageIndex ?? null)}
        >
          <rect
            className={ROLE_CLASS[seg.role] || 'm-tpl'}
            x={x(seg.start) + 0.5}
            y={tapeY}
            width={Math.max(x(seg.end) - x(seg.start) - 1, 2)}
            height={tapeH}
            rx={2}
          >
            <title>{`${seg.role} · positions ${seg.start}–${seg.end - 1} · drag to move the lens, click for details`}</title>
          </rect>
          {seg.role === 'template' && seg.end - seg.start < 8 && cellWidth * (seg.end - seg.start) > LABEL_MIN && (
            <text x={x(seg.start) + 3} y={tapeY + 17} className="m-mut" pointerEvents="none">
              ⟨s⟩
            </text>
          )}
        </g>,
      )
    }
  }

  const generatedCells = generatedTail.map((token) => (
    <g key={`gen-${token.position}`}>
      <rect
        className={token.channel === 'thinking' ? 'm-think' : 'm-asst'}
        x={x(token.position) + 0.5}
        y={tapeY}
        width={Math.max(cellWidth - 1, 1)}
        height={tapeH}
        rx={2}
      >
        <title>{cellTitle(token.position, token.piece, token.id)}</title>
      </rect>
    </g>
  ))

  const tickMarks = rulerStep > 0 ? (() => {
    const marks = []
    for (let position = 0; position <= domain; position += rulerStep) {
      marks.push(
        <g key={`tick-${position}`}>
          <line className="m-tick" x1={x(position)} y1={16} x2={x(position)} y2={21} />
          <text className="m-mut" x={x(position)} y={13} textAnchor="middle" fontSize={10}>
            {position}
          </text>
        </g>,
      )
    }
    return marks
  })() : null

  const plusTicks = ticks.map((position) => (
    <text key={`plus-${position}`} className="m-infotext" x={x(position) + 0.5} y={tapeY - 5} textAnchor="middle" fontSize={11} fontWeight={700}>
      +
    </text>
  ))

  // Band: restored / computed / waiting (running only) / generated
  const bandSegments = []
  if (reused > 0) {
    bandSegments.push(<rect key="ok" className="m-band-ok" x={x(0)} y={bandY} width={Math.max(x(reused) - x(0), 2)} height={bandH} rx={2} />)
  }
  if (totalProcessed > reused) {
    bandSegments.push(<rect key="info" className="m-band-info" x={x(reused)} y={bandY} width={Math.max(x(totalProcessed) - x(reused), 2)} height={bandH} rx={2} />)
  }
  if (totalProcessed < total) {
    bandSegments.push(<rect key="wait" className="m-band-wait" x={x(totalProcessed)} y={bandY} width={Math.max(x(total) - x(totalProcessed), 2)} height={bandH} rx={2} />)
  }
  if (generatedTail.length > 0) {
    bandSegments.push(<rect key="think" className="m-band-think" x={x(total)} y={bandY} width={Math.max(x(total + generatedTail.length) - x(total), 2)} height={bandH} rx={2} />)
  }

  const entryRows = []
  for (const row of rowsShown) {
    const isSelected = row.id === selectedEntryId
    const isChosen = row.chosen
    const hasArrow = isChosen
    entryRows.push(
      <g key={row.id} className="strip" tabIndex={0} role="button" aria-label={`Cache entry ${row.id}`}
        onClick={() => onSelectEntry?.(row.id)}
        onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelectEntry?.(row.id) } }}>
        <rect className="m-hit" x={0} y={y - 4} width={width} height={barH + 8} />
        <text x={4} y={y + 13} fontWeight={isSelected ? 700 : 500} className={row.ghost ? 'm-mut' : undefined}>
          {row.isNew ? <tspan className="m-infotext" fontWeight={700} fontSize={9}>NEW </tspan> : null}
          #{row.id} {row.ghost ? 'evicted' : kindLabel(row.kind)} · {row.len}
        </text>
        {row.ghost && <text x={GUTTER - 4} y={y - 3} textAnchor="end" className="m-mut" fontSize={9}>evicted</text>}
        <rect
          className={`m-bar-ok ${row.ghost ? 'm-ghost' : ''}`}
          x={x(0)}
          y={y}
          width={Math.max(x(row.match) - x(0), 2)}
          height={barH}
          rx={3}
        />
        {row.breakAt != null && !row.ghost && (
          <>
            <rect
              className="m-hatch"
              x={x(row.breakAt)}
              y={y}
              width={Math.max(x(row.len) - x(row.breakAt), 2)}
              height={barH}
              rx={3}
            >
              <title>{`saved labels continue past the first difference at position ${row.breakAt}`}</title>
            </rect>
            <text className="m-badtext" x={x(row.breakAt)} y={y + 14} textAnchor="middle" fontWeight={700}>✕</text>
            <text className="m-mut" x={x(row.breakAt) + 10} y={y - 3} fontSize={10}>
              breaks at {row.breakAt}
            </text>
          </>
        )}
        {row.breakAt == null && row.usable && !row.ghost && (
          <text className="m-oktext" x={x(row.len) + 6} y={y + 13} fontWeight={700}>
            ✓ {isChosen ? 'restored' : 'usable'}{row.relabeled ? '  ⚠ label ≠ state' : ''}
          </text>
        )}
        {row.relabeled && (
          <text className="m-warntext" x={x(row.len) + 6} y={row.breakAt == null ? y - 3 : y + 13} fontSize={10}>
            ⚠ label ≠ state
          </text>
        )}
        {isSelected && (
          <rect className="m-sel" x={x(0) - 2} y={y - 3} width={x(row.len) - x(0) + 4} height={barH + 6} rx={4} pointerEvents="none" />
        )}
        {isSelected && row.breakAt != null && (
          <line className="m-guide" x1={x(row.breakAt)} y1={tapeY} x2={x(row.breakAt)} y2={y + barH} pointerEvents="none" />
        )}
      </g>,
    )
    if (hasArrow) {
      entryRows.push(
        <g key={`arrow-${row.id}`} pointerEvents="none">
          <line className="m-arrow" x1={x(row.len) - 3} y1={y - 3} x2={x(row.len) - 3} y2={bandBottom + 2} markerEnd="url(#m-arr)" />
          {pulseEntryId === row.id && <rect className="m-pulse" x={x(0) - 2} y={y - 3} width={x(row.len) - x(0) + 4} height={barH + 6} rx={4} />}
        </g>,
      )
    }
    y += barH + 15
  }
  if (folded.length > 0) {
    entryRows.push(
      <g key="folded">
        <text className="m-mut" x={GUTTER} y={y + 12} fontSize={10}>
          +{folded.length} more saved state{folded.length > 1 ? 's' : ''} (longest first)
          {chosenFolded ? <tspan className="m-oktext"> · includes the restored entry</tspan> : null}
        </text>
      </g>,
    )
    y += 20
  } else if (rows.length === 0) {
    entryRows.push(
      <text key="empty" className="m-mut" x={GUTTER} y={y + 12} fontSize={10}>
        no saved states yet — this request computes everything
      </text>,
    )
    y += 20
  }

  const height = y + 8
  const hlOverlay = hlSpan ? (
    <rect className="m-hl" x={x(hlSpan.start)} y={tapeY - 2} width={Math.max(x(hlSpan.end) - x(hlSpan.start), 3)} height={tapeH + 4} rx={3} pointerEvents="none" />
  ) : null
  const lensGuide = lensInDomain ? (
    <line className="m-guide" x1={x(lensCenter)} y1={tapeY - 8} x2={x(lensCenter)} y2={bandBottom} pointerEvents="none" />
  ) : null

  return (
    <div className="map-wrap" ref={wrapRef} aria-label="Context tape and cache entry bars on one token-position ruler">
      <svg className="map" role="img" viewBox={`0 0 ${width} ${height}`} height={height}>
        <defs>
          <pattern id="m-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect className="m-band-wait" width="6" height="6" />
            <line className="m-hatch-line" x1="0" y1="0" x2="0" y2="6" />
          </pattern>
          <marker id="m-arr" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path className="m-band-ok" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>
        <text className="m-mut" x="4" y={13}>position</text>
        {tickMarks}
        {plusTicks}
        <text x="4" y={tapeY + 17} fontWeight={600}>Context</text>
        {tapeCells}
        {generatedCells}
        {hlOverlay}
        {lensGuide}
        {generatedTail.length > 0 && (
          <text className="m-mut" x={x(total) + 2} y={tapeY - 6} fontSize={10}>
            generated → (thinking dashed: saved in the after-reply entry, never sent again)
          </text>
        )}
        <text className="m-mut" x="4" y={bandY + 9} fontSize={10}>restored | computed</text>
        {bandSegments}
        <text className="m-mut" x="4" y={bandBottom + 16} fontSize={10} fontWeight={600}>
          CACHE ENTRIES (saved states)
        </text>
        {entryRows}
      </svg>
    </div>
  )
}
