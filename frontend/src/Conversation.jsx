import { useEffect, useRef } from 'react'
import { deriveChannels } from './OutputPanel.jsx'

// The left column: chat bubbles (newest at the bottom) plus the input box.
// Hovering a bubble lights that message's tokens in the map; clicking it
// opens the message in the Inspector.  Thinking folds into a dashed
// "not sent again" chip.  Replaces ChatInput and LiveOutput.

function oneLine(text, limit = 80) {
  const flat = (text || '').replace(/\s+/g, ' ').trim()
  return flat.length > limit ? `${flat.slice(0, limit)}…` : flat
}

export default function Conversation({
  session,
  draft,
  onDraftChange,
  turns,
  modelLoaded,
  running,
  busy,
  editing,
  generation,
  tokens,
  hoveredMessage,
  onHoverMessage,
  onSelectMessage,
  onSend,
  onStop,
  onNewConversation,
  onLoadModel,
}) {
  const bubblesRef = useRef(null)

  useEffect(() => {
    if (bubblesRef.current) bubblesRef.current.scrollTop = bubblesRef.current.scrollHeight
  }, [session.messages, draft, tokens.length, running])

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      onSend()
    }
  }

  // Assistant messages map 1:1 to turns in order.
  let assistantCount = 0
  const thinkingByIndex = new Map()
  session.messages.forEach((message, index) => {
    if (message.role === 'assistant') {
      thinkingByIndex.set(index, turns[assistantCount]?.thinking || 0)
      assistantCount += 1
    }
  })
  const channels = generation?.channels ?? deriveChannels(tokens)
  const liveThinking = (tokens.length ? tokens : generation?.tokens || []).filter(
    (token) => token.channel === 'thinking',
  ).length
  const liveAnswer = channels.answer || ''

  return (
    <div className="conversation">
      {!modelLoaded && (
        <div className="note note-warn row-between">
          <span>No model loaded.</span>
          <button type="button" className="mini" disabled={busy} onClick={onLoadModel}>
            Load model
          </button>
        </div>
      )}
      <div className="bubbles" ref={bubblesRef}>
        {session.messages.length === 0 && !running && (
          <p className="muted">Say something to the model. Everything it reads appears on the ruler to the right.</p>
        )}
        {session.messages.map((message, index) => (
          <div
            key={index}
            className={`bubble ${message.role} ${message.enabled === false ? 'off' : ''}`}
            onMouseEnter={() => onHoverMessage(index)}
            onMouseLeave={() => onHoverMessage(null)}
            onClick={() => onSelectMessage(index)}
            title="click to inspect this message"
          >
            <div className="bubble-head">
              <span>{message.role}{message.enabled === false ? ' · disabled' : ''}</span>
              <span>#{index + 1}</span>
            </div>
            {message.content || '(empty)'}
            {message.role === 'assistant' && thinkingByIndex.get(index) > 0 && (
              <span className="thinkchip">
                thinking · {thinkingByIndex.get(index)} tokens · not sent again
              </span>
            )}
          </div>
        ))}
        {running && (
          <div className="bubble assistant" aria-live="polite">
            <div className="bubble-head"><span>assistant · writing</span></div>
            {liveAnswer || '…'}<span className="caret" />
            {liveThinking > 0 && (
              <span className="thinkchip">
                thinking · {liveThinking} tokens · not sent again
              </span>
            )}
          </div>
        )}
      </div>
      <div className="chat-input">
        <textarea
          rows={3}
          value={draft}
          placeholder={turns.length === 0 ? 'Say something to the model…' : 'Next message…'}
          onChange={(event) => onDraftChange(event.target.value)}
          onKeyDown={handleKeyDown}
        />
        {editing && (
          <p className="note note-warn">
            You are staging an edit on the map — apply or discard it before sending.
          </p>
        )}
        <div className="button-row wrap">
          <button
            type="button"
            className="primary"
            disabled={!modelLoaded || busy || !draft.trim()}
            onClick={onSend}
          >
            Send
          </button>
          <button type="button" disabled={!running} onClick={onStop}>Stop</button>
          <span className="muted small">Enter to send · Shift+Enter for a new line</span>
          <span className="spacer" />
          <span className="muted small">{turns.length} {turns.length === 1 ? 'turn' : 'turns'}</span>
          <button type="button" disabled={busy} onClick={onNewConversation}>New conversation</button>
        </div>
      </div>
    </div>
  )
}

export { oneLine }
