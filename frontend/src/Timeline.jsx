// The folded timeline: one dot per recorded turn with a tiny sent/reused
// bar.  Selecting a turn renders its snapshot into the Stage read-only;
// "current" returns to the live view.

export default function Timeline({ turns, viewTurnId, onSelect, disabled }) {
  return (
    <details className="timeline">
      <summary>
        Timeline · {turns.length} recorded turn{turns.length === 1 ? '' : 's'}
        {viewTurnId != null ? ' · viewing a past turn (read-only)' : ''}
      </summary>
      <div className="dots">
        <button
          type="button"
          className={viewTurnId == null ? 'active' : ''}
          disabled={disabled}
          onClick={() => onSelect(null)}
        >
          current state
        </button>
        {turns.map((turn) => {
          const total = Math.max(turn.input + turn.generated, 1)
          const reusedPct = ((turn.reused / total) * 100).toFixed(1)
          const inputPct = (((turn.input - turn.reused) / total) * 100).toFixed(1)
          const edits = turn.edits && turn.edits.length > 0 ? `\nedits: ${turn.edits.join('; ')}` : ''
          return (
            <button
              key={turn.id}
              type="button"
              className={`dot ${viewTurnId === turn.id ? 'active' : ''}`}
              disabled={disabled}
              title={`turn ${turn.n}: sent ${turn.input} (${turn.reused} restored) → generated ${turn.generated}${edits}`}
              onClick={() => onSelect(viewTurnId === turn.id ? null : turn.id)}
            >
              <span className="muted">#{turn.n}</span>
              <span className="mini-bar" aria-hidden="true">
                <i className="ok" style={{ width: `${reusedPct}%` }} />
                <i className="new" style={{ width: `${inputPct}%` }} />
              </span>
              <span className="muted small">
                {turn.reused}/{turn.input} → {turn.generated}
              </span>
            </button>
          )
        })}
        {turns.length === 0 && (
          <span className="muted small">Send a message to record a turn.</span>
        )}
      </div>
    </details>
  )
}
