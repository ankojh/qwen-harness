import { buildMapModel } from './mapModel.js'

// "How reuse works" — the three classic examples (append / edit early /
// repeat a word) drawn as three mini maps in the same ruler grammar as the
// main map, folded out of the way.  Illustrative token pieces; the live map
// above uses the real ids.

const EXAMPLES = {
  append: {
    title: 'Append text',
    request: ['My', ' name', ' is', ' Alice', '.', ' Hello'],
    saved: ['My', ' name', ' is', ' Alice', '.'],
    note: 'A saved state for the entire beginning can be restored. Only the new suffix needs input processing.',
  },
  edit: {
    title: 'Edit an early word',
    request: ['My', ' name', ' is', ' Bob', '.', ' Hello'],
    saved: ['My', ' name', ' is', ' Alice', '.'],
    note: 'The first difference breaks this saved prefix. This model cannot rewind the longer state, so the whole entry is unusable.',
  },
  repeat: {
    title: 'Repeat a word',
    request: ['Hello', ' Alice', ',', ' Alice'],
    saved: ['My', ' name', ' is', ' Alice', '.'],
    note: '“ Alice” appears in both, but at different positions after different beginnings. Repeated words do not create a reusable prefix.',
  },
}

function MiniMap({ request, saved }) {
  const ids = Array.from({ length: request.length }, (_, index) => index + 1)
  const report = {
    token_ids: ids,
    token_pieces: request.map((piece, index) => ({ id: index + 1, piece, text: piece, special: false })),
    spans: [{ role: 'user', index: 1, start: 0, end: request.length, kind: 'message' }],
    cache_prediction: null,
    cache_entries: [],
  }
  const model = buildMapModel({
    report,
    entries: [{
      id: 1,
      kind: 'checkpoint',
      token_ids: saved.map((piece) => {
        const found = request.indexOf(piece)
        return found >= 0 ? found + 1 : 900 + saved.indexOf(piece)
      }),
    }],
    reuseEnabled: true,
  })
  const entry = model.rows[0]
  const W = 300
  const L = 4
  const plot = W - L - 4
  const x = (position) => L + (position / model.domain) * plot
  const cw = plot / model.domain
  const tapeY = 14
  const tapeH = 14
  const bandY = tapeY + tapeH + 4
  const bandH = 6
  const barY = bandY + bandH + 8
  const barH = 10
  return (
    <svg viewBox={`0 0 ${W} ${barY + barH + 6}`} role="img" aria-label={report.spans[0].role}>
      {model.tape.map((seg) => (
        <rect key={seg.start} className={`m-user`} x={x(seg.start)} y={tapeY} width={Math.max(x(seg.end) - x(seg.start), 2)} height={tapeH} rx={2} />
      ))}
      {model.tape[0].end - model.tape[0].start <= 12 &&
        request.map((piece, position) => (
          <text key={position} x={x(position) + cw / 2} y={tapeY + 10} textAnchor="middle" fontSize={8} pointerEvents="none">
            {piece.replace(/ /g, '␣').slice(0, 3)}
          </text>
        ))}
      <rect className="m-band-ok" x={x(0)} y={bandY} width={Math.max(x(entry.match) - x(0), 2)} height={bandH} rx={2} />
      <rect className="m-band-info" x={x(entry.match)} y={bandY} width={Math.max(x(request.length) - x(entry.match), 2)} height={bandH} rx={2} />
      <rect className="m-bar-ok" x={x(0)} y={barY} width={Math.max(x(entry.match) - x(0), 2)} height={barH} rx={2} />
      {entry.breakAt != null && (
        <>
          <rect className="m-hatch" x={x(entry.breakAt)} y={barY} width={Math.max(x(entry.len) - x(entry.breakAt), 2)} height={barH} rx={2} />
          <text className="m-badtext" x={x(entry.breakAt)} y={barY + 9} textAnchor="middle" fontSize={9} fontWeight={700}>✕</text>
        </>
      )}
      <line className="m-guide" x1={x(entry.breakAt ?? entry.match)} y1={tapeY} x2={x(entry.breakAt ?? entry.match)} y2={barY + barH} />
    </svg>
  )
}

export default function ReuseExplainer() {
  return (
    <div className="reuse-explainer">
      <details>
        <summary>How reuse works</summary>
        <p>
          <strong>Reuse means skipping input computation.</strong> The full context is still sent.
          The server restores the model state already computed for its matching beginning, then
          processes the remaining tokens and generates a fresh answer. Token ids must match in
          order, starting at position 0 — the same word elsewhere is insufficient, and at least
          one input token is always processed.
        </p>
        <div className="mini-maps">
          {Object.values(EXAMPLES).map((example) => (
            <div key={example.title} className="mini-map">
              <h5>{example.title}</h5>
              <MiniMap request={example.request} saved={example.saved} />
              <p className="muted small">{example.note}</p>
            </div>
          ))}
        </div>
        <p className="muted small">
          This Qwen3.5 model has recurrent state as well as attention keys and values; its saved
          state cannot rewind to an earlier position. Checkpoints save usable stopping points.
          Relabelling changes token labels while leaving computed state unchanged, which can change
          the answer. Ordinary reuse saves work; it does not replay a stored answer.
        </p>
      </details>
    </div>
  )
}
