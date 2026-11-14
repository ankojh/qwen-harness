// One message box.  Send appends the text to the session as a user message
// and starts a generation; App appends the answer when the run finishes.

export default function ChatInput({
  draft,
  onDraftChange,
  turns,
  modelLoaded,
  running,
  busy,
  onSend,
  onStop,
  onNewConversation,
  onLoadModel,
}) {
  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      onSend()
    }
  }

  return (
    <div className="chat-input">
      {!modelLoaded && (
        <div className="note note-warn row-between">
          <span>No model loaded.</span>
          <button type="button" className="mini" disabled={busy} onClick={onLoadModel}>Load model</button>
        </div>
      )}
      <textarea
        rows={3}
        value={draft}
        placeholder={turns === 0 ? 'Say something to the model…' : 'Next message…'}
        onChange={(event) => onDraftChange(event.target.value)}
        onKeyDown={handleKeyDown}
      />
      <div className="button-row wrap">
        <button
          type="button"
          className="primary"
          disabled={!modelLoaded || running || !draft.trim()}
          onClick={onSend}
        >
          Send
        </button>
        <button type="button" disabled={!running} onClick={onStop}>Stop</button>
        <span className="muted small">Enter to send · Shift+Enter for a new line</span>
        <span className="spacer" />
        <span className="muted small">{turns} {turns === 1 ? 'turn' : 'turns'}</span>
        <button type="button" disabled={running} onClick={onNewConversation}>New conversation</button>
      </div>
    </div>
  )
}
