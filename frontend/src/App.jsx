import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import ConfigPanel from './ConfigPanel.jsx'
import InputPanel from './InputPanel.jsx'
import OutputPanel from './OutputPanel.jsx'
import EventLog from './EventLog.jsx'
import Conversation from './Conversation.jsx'
import Stage from './Stage.jsx'
import {
  clearCache,
  formatError,
  getCacheTokens,
  getGeneration,
  getState,
  modelAction,
  previewPrompt,
  putConfig,
  resetConfig,
  startGeneration,
  stopGeneration,
  subscribeEvents,
} from './api.js'

const EMPTY_MESSAGE = {
  role: 'user',
  content: '',
  reasoning_content: null,
  enabled: true,
}

// Everything that is not input, output, context or cache lives in the
// Details drawer.
const DETAILS_TABS = [
  ['settings', 'Settings & model'],
  ['prompt', 'Prompt editor'],
  ['output', 'Output details'],
  ['log', 'Event log'],
]

function newRequestId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID()
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

// Turns 1..N (their recorded snapshots) plus Next (the request being typed).
// Rendered by the Stage's folded timeline; kept here for the record.

export default function App() {
  const [state, setState] = useState(null)
  const [session, setSession] = useState({
    mode: 'messages',
    messages: [],
    raw_text: '',
  })
  const [draft, setDraft] = useState('')
  const [editing, setEditing] = useState(false)
  const [lastInput, setLastInput] = useState('')
  const [turns, setTurns] = useState([])
  const [detailsTab, setDetailsTab] = useState(null)
  const [viewTurnId, setViewTurnId] = useState(null)
  const [preview, setPreview] = useState(null)
  const [previewError, setPreviewError] = useState(null)
  const [autoPreview, setAutoPreview] = useState(true)
  const [generation, setGeneration] = useState(null)
  const [tokens, setTokens] = useState([])
  const [activeGenId, setActiveGenId] = useState(null)
  const [events, setEvents] = useState([])
  const [sseState, setSseState] = useState('connecting')
  const [configError, setConfigError] = useState(null)
  const [actionError, setActionError] = useState(null)
  const [saving, setSaving] = useState(false)
  const activeGenRef = useRef(null)
  // Generation started by Send whose answer still has to join the session.
  const pendingTurnRef = useRef(null)
  // One-line descriptions of the edits applied since the last turn.  The
  // next recorded turn takes them and clears the list.
  const [pendingEdits, setPendingEdits] = useState([])
  const pendingEditsRef = useRef([])
  // SSE progress of the running generation: the lookup event, the latest
  // prefill chunk, and the cache changes, for the pipeline/map live view.
  const [runProgress, setRunProgress] = useState(null)
  // Which message the chat bubbles are hovering (lights its tape segment).
  const [hoveredMessage, setHoveredMessage] = useState(null)
  // A bubble was clicked: {index, seq}; the Stage opens it in the Inspector.
  const [bubbleClick, setBubbleClick] = useState(null)

  const logEdit = useCallback((line) => {
    pendingEditsRef.current = [...pendingEditsRef.current, line]
    setPendingEdits(pendingEditsRef.current)
  }, [])

  const refreshState = useCallback(async () => {
    const data = await getState()
    setState(data)
    return data
  }, [])

  // A cache action (delete, relabel, restore, save checkpoint) with its
  // edit-log line: run it, log it, refresh state, surface errors.  ``line``
  // may be a function of the result (for labels that need the entry id).
  const runCacheAction = useCallback(
    async (action, line, options = {}) => {
      if (!options.silent) setActionError(null)
      try {
        const result = await action()
        logEdit(typeof line === 'function' ? line(result) : line)
        await refreshState()
        return result
      } catch (error) {
        if (!options.silent) setActionError(formatError(error.detail || error))
        throw error
      }
    },
    [logEdit, refreshState],
  )

  // The Edit view applies its staged copy: it replaces the conversation and
  // each change becomes an edit-log line.
  const applyStagedSession = useCallback(
    (nextSession, lines) => {
      setSession(nextSession)
      setPreview(null)
      setViewTurnId(null)
      lines.forEach(logEdit)
    },
    [logEdit],
  )

  const refreshGeneration = useCallback(async (id) => {
    const record = await getGeneration(id)
    setGeneration(record)
    setTokens(record.tokens || [])
    return record
  }, [])

  // Pick up a generation that was already running when the page loaded: fetch
  // the tokens produced so far once; SSE token events carry the rest.
  const adoptRunningGeneration = useCallback(
    async (data) => {
      if (!data.active_generation) return
      activeGenRef.current = data.active_generation.id
      setActiveGenId(data.active_generation.id)
      const record = await getGeneration(data.active_generation.id)
      setTokens((previous) => {
        const seen = new Set(previous.map((token) => token.index))
        return [...(record.tokens || []).filter((token) => !seen.has(token.index)), ...previous]
          .sort((a, b) => a.index - b.index)
      })
    },
    [],
  )

  // Initial state.
  useEffect(() => {
    getState()
      .then((data) => {
        setState(data)
        return adoptRunningGeneration(data)
      })
      .catch((error) => setActionError(formatError(error.detail || error)))
  }, [adoptRunningGeneration])

  // One SSE subscription for the whole app.  There is deliberately no polling:
  // the live output is built from `token` events, and the full generation
  // record is fetched once when the run finishes (or after a reconnect).
  // Polling the record would make the server deep-copy every token under the
  // lock the generation thread needs, slowing down long generations.
  useEffect(() => {
    let hadError = false
    const close = subscribeEvents({
      onOpen: () => {
        setSseState('connected')
        if (hadError) {
          // EventSource resent Last-Event-ID and the server replays missed
          // events; refresh state in case the replay could not cover the gap.
          hadError = false
          refreshState()
            .then((data) => {
              if (data.active_generation) return adoptRunningGeneration(data)
              if (activeGenRef.current) {
                const finishedId = activeGenRef.current
                activeGenRef.current = null
                setActiveGenId(null)
                return refreshGeneration(finishedId)
              }
              return undefined
            })
            .catch(() => {})
        }
      },
      onError: () => {
        hadError = true
        setSseState('reconnecting')
      },
      onEvent: (type, data, seq) => {
        setEvents((previous) => {
          if (
            Number.isFinite(seq) &&
            previous.some((event) => event.seq === seq && event.type === type)
          ) {
            return previous
          }
          const next = [...previous, { seq, time: Date.now() / 1000, type, data }]
          return next.length > 800 ? next.slice(-800) : next
        })

        if (type === 'token') {
          if (activeGenRef.current === data.generation_id) {
            setTokens((previous) =>
              previous.some((token) => token.index === data.index)
                ? previous
                : [...previous, data],
            )
          }
        } else if (type === 'generation.started') {
          activeGenRef.current = data.generation_id
          setActiveGenId(data.generation_id)
          setGeneration(null)
          setTokens([])
          setActionError(null)
          setRunProgress({})
        } else if (type === 'cache.lookup') {
          setRunProgress((previous) => (previous ? { ...previous, lookup: data } : previous))
        } else if (type === 'prefill.chunk') {
          setRunProgress((previous) => previous ? {
            ...previous,
            prefill: { processed_tokens: data.processed_tokens, total_tokens: data.total_tokens },
          } : previous)
        } else if (type === 'cache.insert') {
          setRunProgress((previous) => previous ? {
            ...previous,
            saved: { ...(previous.saved || { added: 0, evicted: 0 }), added: (previous.saved?.added || 0) + 1 },
          } : previous)
        } else if (type === 'cache.evict') {
          setRunProgress((previous) => previous ? {
            ...previous,
            saved: { ...(previous.saved || { added: 0, evicted: 0 }), evicted: (previous.saved?.evicted || 0) + 1 },
          } : previous)
        } else if (type === 'generation.finished') {
          if (activeGenRef.current === data.generation_id) activeGenRef.current = null
          setActiveGenId(null)
          refreshGeneration(data.generation_id).catch(() => {})
          refreshState().catch(() => {})
        } else if (type === 'resync') {
          refreshState().catch(() => {})
          if (activeGenRef.current) refreshGeneration(activeGenRef.current).catch(() => {})
        } else if (
          type === 'model.loaded' ||
          type === 'model.unloaded' ||
          type === 'model.error' ||
          type === 'cache.clear' ||
          type === 'cache.delete' ||
          type === 'cache.relabel' ||
          type === 'cache.restore' ||
          type === 'cache.insert' ||
          type === 'cache.checkpoint' ||
          type === 'config.changed' ||
          type === 'cache.evict'
        ) {
          refreshState().catch(() => {})
        }
      },
    })
    return close
  }, [adoptRunningGeneration, refreshGeneration, refreshState])

  // What the next request would contain: the session plus the draft as a
  // pending user message.  Context and Cache read the preview of this.
  const nextSession = useMemo(() => {
    const text = draft.trim()
    if (!text || session.mode !== 'messages') return session
    return { ...session, messages: [...session.messages, { ...EMPTY_MESSAGE, content: text }] }
  }, [session, draft])
  const hasContext = nextSession.mode === 'raw' ? Boolean(nextSession.raw_text) : nextSession.messages.length > 0

  const runPreview = useCallback(async () => {
    try {
      const report = await previewPrompt(nextSession)
      setPreview(report)
      setPreviewError(null)
    } catch (error) {
      setPreviewError(formatError(error.detail || error))
    }
  }, [nextSession])

  // Auto-refresh calls the tokenizer only, never the model.  It also reruns
  // when the cache changes, so the reuse prediction stays current.  The
  // previous report stays on screen until the new one arrives: the tape
  // redraws in place instead of flashing empty (the moment after a run is
  // the animated switch the map teaches with).
  useEffect(() => {
    if (activeGenId) return undefined
    if (!autoPreview || !state?.model_loaded) {
      setPreview(null)
      return undefined
    }
    if (!hasContext) {
      setPreview(null)
      setPreviewError(null)
      return undefined
    }
    let cancelled = false
    const timer = setTimeout(async () => {
      try {
        const report = await previewPrompt(nextSession)
        if (!cancelled) {
          setPreview(report)
          setPreviewError(null)
        }
      } catch (error) {
        if (!cancelled) setPreviewError(formatError(error.detail || error))
      }
    }, 300)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [autoPreview, nextSession, hasContext, state?.model_loaded, state?.config, state?.cache, activeGenId])

  // When a run started by Send finishes, its answer joins the conversation
  // (without the thinking, which the chat template would drop anyway) and the
  // turn is recorded with its context report and a cache snapshot: the entry
  // token lists plus the summary rows, fetched right after generation.finished.
  useEffect(() => {
    if (!generation || !generation.finish || generation.id !== pendingTurnRef.current) return
    pendingTurnRef.current = null
    const answer = (generation.channels?.answer || '').trim()
    const submitted = generation.session
    // The edits applied since the last turn belong to this one.
    const edits = pendingEditsRef.current
    pendingEditsRef.current = []
    setPendingEdits([])
    setSession({
      ...submitted,
      messages: answer
        ? [...submitted.messages, { role: 'assistant', content: answer, reasoning_content: null, enabled: true }]
        : submitted.messages,
    })
    const generated = generation.stats?.decode?.generated_tokens ?? generation.tokens_generated ?? 0
    setTurns((previous) => [
      ...previous,
      {
        id: generation.id,
        n: previous.length + 1,
        input: generation.prompt?.budget?.input_tokens ?? 0,
        reused: generation.stats?.prefill?.reused_tokens ?? 0,
        generated,
        thinking: (generation.tokens || []).filter((token) => token.channel === 'thinking').length,
        context: generation.prompt,
        prefill: generation.stats?.prefill || null,
        finish: generation.finish,
        cache: null,
        edits,
      },
    ])
    ;(async () => {
      try {
        const [tokenLists, freshState] = await Promise.all([getCacheTokens(), getState()])
        const rows = new Map((freshState.cache?.entries || []).map((entry) => [entry.id, entry]))
        const entries = (tokenLists.entries || []).map((row) => ({ ...rows.get(row.id), ...row }))
        const snapshot = {
          entries,
          total_bytes: freshState.cache?.total_bytes ?? 0,
          last_lookup: generation.cache_lookup ?? freshState.cache?.last_lookup ?? null,
        }
        setTurns((previous) =>
          previous.map((turn) => (turn.id === generation.id ? { ...turn, cache: snapshot } : turn)),
        )
      } catch {
        // The snapshot is best-effort; the live panels keep working without it.
      }
    })()
  }, [generation])

  useEffect(() => {
    if (!detailsTab) return undefined
    const onKey = (event) => {
      if (event.key === 'Escape') setDetailsTab(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [detailsTab])

  async function updateConfig(patch) {
    setSaving(true)
    setConfigError(null)
    try {
      const data = await putConfig(patch)
      setState(data)
      if ('cache_reuse' in patch) {
        logEdit(`turned cache reuse ${patch.cache_reuse ? 'on' : 'off'}`)
      }
      if ('cache_checkpoints' in patch) {
        logEdit(`changed cache checkpoints to ${patch.cache_checkpoints}`)
      }
    } catch (error) {
      setConfigError(formatError(error.detail || error))
    } finally {
      setSaving(false)
    }
  }

  async function handleReset() {
    setSaving(true)
    setConfigError(null)
    try {
      setState(await resetConfig())
    } catch (error) {
      setConfigError(formatError(error.detail || error))
    } finally {
      setSaving(false)
    }
  }

  async function handleModel(action) {
    setSaving(true)
    setActionError(null)
    try {
      setState(await modelAction(action))
    } catch (error) {
      setActionError(formatError(error.detail || error))
    } finally {
      setSaving(false)
    }
  }

  async function handleClearCache() {
    setSaving(true)
    try {
      const before = state.cache?.entries_count ?? 0
      const data = await clearCache()
      setState(data)
      logEdit(`cleared the cache (${before} ${before === 1 ? 'entry' : 'entries'})`)
    } catch (error) {
      setActionError(formatError(error.detail || error))
    } finally {
      setSaving(false)
    }
  }

  async function handleGenerate() {
    setActionError(null)
    try {
      const response = await startGeneration(newRequestId(), session)
      activeGenRef.current = response.generation_id
      setActiveGenId(response.generation_id)
      setGeneration(null)
      setTokens([])
      refreshState().catch(() => {})
    } catch (error) {
      setActionError(formatError(error.detail || error))
    }
  }

  async function handleSend() {
    const text = draft.trim()
    if (editing) { setActionError('Apply or discard the staged context edits before sending.'); return }
    if (!text || activeGenId || !state?.model_loaded) return
    const previous = session
    const next = {
      ...session,
      mode: 'messages',
      messages: [...session.messages, { ...EMPTY_MESSAGE, content: text }],
    }
    setViewTurnId(null)
    setSession(next)
    // Fetch the submitted request's report immediately (the debounced preview
    // may not have landed yet, and auto-preview pauses during the run): it
    // freezes the tape on the prompt while the run is in flight.
    setDraft('')
    setLastInput(text)
    setActionError(null)
    try {
      const [response, submittedReport] = await Promise.all([
        startGeneration(newRequestId(), next),
        previewPrompt(next).catch(() => null),
      ])
      if (submittedReport) setPreview(submittedReport)
      pendingTurnRef.current = response.generation_id
      activeGenRef.current = response.generation_id
      setActiveGenId(response.generation_id)
      setGeneration(null)
      setTokens([])
      refreshState().catch(() => {})
    } catch (error) {
      setSession(previous)
      setDraft(text)
      setActionError(formatError(error.detail || error))
    }
  }

  function handleNewConversation() {
    pendingTurnRef.current = null
    pendingEditsRef.current = []
    setPendingEdits([])
    setSession((previous) => ({ ...previous, mode: 'messages', messages: [], raw_text: '' }))
    setTurns([])
    setViewTurnId(null)
    setDraft('')
    setLastInput('')
    setGeneration(null)
    setTokens([])
    setPreview(null)
    setRunProgress(null)
  }

  async function handleStop() {
    try {
      await stopGeneration()
    } catch (error) {
      setActionError(formatError(error.detail || error))
    }
  }

  function handleClearOutput() {
    setGeneration(null)
    setTokens([])
  }

  // Explicit, user-triggered: the answer channel becomes the assistant message.
  // Reasoning is only kept (as reasoning_content) when asked; raw text with
  // think markers is never pasted into the conversation.
  function handleAppend(keepReasoning) {
    if (!generation) return
    const channels = generation.channels || {}
    const content = channels.answer || ''
    const reasoning = keepReasoning && channels.thinking ? channels.thinking : null
    setSession((previous) => ({
      ...previous,
      messages: [
        ...previous.messages,
        {
          role: 'assistant',
          content,
          reasoning_content: reasoning,
          enabled: true,
        },
      ],
    }))
  }

  if (!state) {
    return (
      <div className="boot">
        <strong>Qwen Harness</strong>
        <span className="muted">connecting to the local backend…</span>
        {actionError && <><span className="note note-error">{actionError}</span><button type="button" onClick={() => refreshState().catch((error) => setActionError(error.message))}>Retry connection</button></>}
      </div>
    )
  }

  const running = Boolean(activeGenId)
  const busy = saving || running
  const modelLoaded = Boolean(state.model_loaded)
  // Which snapshot the right column shows: a historical turn, or Next (null).
  const viewTurn = viewTurnId ? turns.find((turn) => turn.id === viewTurnId) || null : null
  const viewIndex = viewTurn ? turns.findIndex((turn) => turn.id === viewTurn.id) : -1
  const prevTurn = viewTurn ? turns[viewIndex - 1] || null : turns[turns.length - 1] || null

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <strong>Qwen Harness</strong>
          <span className="muted">Context & cache lab · {window.location.port || 'local'}</span>
        </div>
        <div className="topbar-meta">
          <span className={`pill ${modelLoaded ? 'pill-ok' : 'pill-off'}`}>
            {modelLoaded ? 'model loaded' : 'model idle'}
          </span>
          <span>{sseState !== 'connected' ? 'reconnecting · live state may be stale' : running ? 'generating' : 'idle'}</span>
          <button
            type="button"
            className="menu-button"
            aria-label="Details"
            aria-expanded={Boolean(detailsTab)}
            onClick={() => setDetailsTab(detailsTab ? null : 'settings')}
          >
            ☰ Details
          </button>
        </div>
      </header>

      {actionError && (
        <div className="banner">
          <span>{actionError}</span>
          <button type="button" className="mini" onClick={() => setActionError(null)}>dismiss</button>
        </div>
      )}

      <div className="main">
        <PanelGroup direction="horizontal" style={{ height: '100%' }}>
          <Panel defaultSize={30} minSize={22}>
            <div className="panel-scroll">
              <div className="panel-title">Conversation</div>
              <Conversation
                session={session}
                draft={draft}
                onDraftChange={setDraft}
                turns={turns}
                modelLoaded={modelLoaded}
                running={running}
                busy={busy || editing}
                editing={editing}
                generation={generation}
                tokens={tokens}
                hoveredMessage={hoveredMessage}
                onHoverMessage={setHoveredMessage}
                onSelectMessage={(index) => setBubbleClick((previous) => ({ index, seq: (previous?.seq || 0) + 1 }))}
                onSend={handleSend}
                onStop={handleStop}
                onNewConversation={handleNewConversation}
                onLoadModel={() => handleModel('load')}
              />
            </div>
          </Panel>

          <PanelResizeHandle className="resize vertical" />

          <Panel defaultSize={70} minSize={35}>
            <div className="panel-scroll">
              <div className="panel-title">
                What the model sees <span className="title-note">one ruler: the context tape, the restored/computed band, and every saved state</span>
              </div>
              <Stage
                state={state}
                session={session}
                draft={draft}
                preview={preview}
                previewError={previewError}
                generation={generation}
                tokens={tokens}
                running={running}
                busy={busy}
                modelLoaded={modelLoaded}
                turns={turns}
                viewTurn={viewTurn}
                viewTurnId={viewTurnId}
                prevTurn={prevTurn}
                pendingEdits={pendingEdits}
                systemPrompt={state.config?.system_prompt || ''}
                runProgress={runProgress}
                hoveredMessage={hoveredMessage}
                bubbleClick={bubbleClick}
                onViewTurnChange={setViewTurnId}
                onApplyStaged={applyStagedSession}
                onCacheAction={runCacheAction}
                onConfigChange={updateConfig}
                onClearCache={handleClearCache}
                onEditingChange={setEditing}
                onLoadModel={() => handleModel('load')}
              />
            </div>
          </Panel>
        </PanelGroup>
      </div>

      {detailsTab && (
        <div className="drawer-backdrop" onClick={() => setDetailsTab(null)}>
          <aside className="drawer" aria-label="Details" onClick={(event) => event.stopPropagation()}>
            <div className="drawer-head">
              <div className="tabs" role="tablist">
                {DETAILS_TABS.map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    role="tab"
                    aria-selected={detailsTab === id}
                    className={detailsTab === id ? 'active' : ''}
                    onClick={() => setDetailsTab(id)}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <button type="button" className="mini" aria-label="Close details" onClick={() => setDetailsTab(null)}>✕</button>
            </div>
            <div className={`drawer-body ${detailsTab === 'log' ? 'log-panel' : ''}`}>
              {detailsTab === 'settings' && (
                <ConfigPanel
                  state={state}
                  busy={busy}
                  error={configError}
                  onApply={updateConfig}
                  onReset={handleReset}
                  onPreset={(preset) => updateConfig(preset.values)}
                  onModelAction={handleModel}
                  onClearCache={handleClearCache}
                />
              )}
              {detailsTab === 'prompt' && (
                <InputPanel
                  session={session}
                  onSessionChange={setSession}
                  preview={preview}
                  previewError={previewError}
                  onPreview={runPreview}
                  autoPreview={autoPreview}
                  setAutoPreview={setAutoPreview}
                  modelLoaded={modelLoaded}
                  thinkStart={state.model?.think_start}
                  thinkEnd={state.model?.think_end}
                />
              )}
              {detailsTab === 'output' && (
                <OutputPanel
                  generation={generation}
                  tokens={tokens}
                  running={running}
                  previewHash={null}
                  modelLoaded={modelLoaded}
                  thinkStart={state.model?.think_start}
                  thinkEnd={state.model?.think_end}
                  onGenerate={handleGenerate}
                  onStop={handleStop}
                  onClearOutput={handleClearOutput}
                  onClearConversation={handleNewConversation}
                  onAppend={handleAppend}
                />
              )}
              {detailsTab === 'log' && (
                <EventLog events={events} sseState={sseState} onClear={() => setEvents([])} />
              )}
            </div>
          </aside>
        </div>
      )}
    </div>
  )
}
