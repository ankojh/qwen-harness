import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import ConfigPanel from './ConfigPanel.jsx'
import InputPanel from './InputPanel.jsx'
import OutputPanel from './OutputPanel.jsx'
import EventLog from './EventLog.jsx'
import ChatInput from './ChatInput.jsx'
import LiveOutput from './LiveOutput.jsx'
import ContextPanel from './ContextPanel.jsx'
import CachePanel from './CachePanel.jsx'
import {
  clearCache,
  formatError,
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

export default function App() {
  const [state, setState] = useState(null)
  const [session, setSession] = useState({
    mode: 'messages',
    messages: [],
    raw_text: '',
  })
  const [draft, setDraft] = useState('')
  const [lastInput, setLastInput] = useState('')
  const [turns, setTurns] = useState([])
  const [detailsTab, setDetailsTab] = useState(null)
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

  const refreshState = useCallback(async () => {
    const data = await getState()
    setState(data)
    return data
  }, [])

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
  // when the cache changes, so the reuse prediction stays current.
  useEffect(() => {
    if (!autoPreview || !state?.model_loaded) return undefined
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
  }, [autoPreview, nextSession, hasContext, state?.model_loaded, state?.config, state?.cache?.accounted_bytes, state?.cache?.entries_count])

  // When a run started by Send finishes, its answer joins the conversation
  // (without the thinking, which the chat template would drop anyway) and the
  // turn is recorded for the context growth chart.
  useEffect(() => {
    if (!generation || !generation.finish || generation.id !== pendingTurnRef.current) return
    pendingTurnRef.current = null
    const answer = (generation.channels?.answer || '').trim()
    const submitted = generation.session
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
      },
    ])
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
      setState(await clearCache())
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
    if (!text || activeGenId || !state?.model_loaded) return
    const previous = session
    const next = {
      ...session,
      mode: 'messages',
      messages: [...session.messages, { ...EMPTY_MESSAGE, content: text }],
    }
    setSession(next)
    setDraft('')
    setLastInput(text)
    setActionError(null)
    try {
      const response = await startGeneration(newRequestId(), next)
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
    setSession((previous) => ({ ...previous, mode: 'messages', messages: [], raw_text: '' }))
    setTurns([])
    setDraft('')
    setLastInput('')
    setGeneration(null)
    setTokens([])
    setPreview(null)
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
        {actionError && <span className="note note-error">{actionError}</span>}
      </div>
    )
  }

  const running = Boolean(activeGenId)
  const busy = saving || running
  const modelLoaded = Boolean(state.model_loaded)

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <strong>Qwen Harness</strong>
          <span className="muted">{state.config?.model_path}</span>
        </div>
        <div className="topbar-meta">
          <span className={`pill ${modelLoaded ? 'pill-ok' : 'pill-off'}`}>
            {modelLoaded ? 'model loaded' : 'model idle'}
          </span>
          <span>{running ? 'generating' : 'idle'}</span>
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
          <Panel defaultSize={50} minSize={25}>
            <PanelGroup direction="vertical" style={{ height: '100%' }}>
              <Panel defaultSize={28} minSize={15}>
                <div className="panel-scroll">
                  <div className="panel-title">Input</div>
                  <ChatInput
                    draft={draft}
                    onDraftChange={setDraft}
                    turns={turns.length}
                    modelLoaded={modelLoaded}
                    running={running}
                    busy={busy}
                    onSend={handleSend}
                    onStop={handleStop}
                    onNewConversation={handleNewConversation}
                    onLoadModel={() => handleModel('load')}
                  />
                </div>
              </Panel>
              <PanelResizeHandle className="resize horizontal" />
              <Panel defaultSize={72} minSize={20}>
                <div className="panel-scroll">
                  <div className="panel-title">Output</div>
                  <LiveOutput generation={generation} tokens={tokens} running={running} lastInput={lastInput} />
                </div>
              </Panel>
            </PanelGroup>
          </Panel>

          <PanelResizeHandle className="resize vertical" />

          <Panel defaultSize={50} minSize={25}>
            <PanelGroup direction="vertical" style={{ height: '100%' }}>
              <Panel defaultSize={55} minSize={15}>
                <div className="panel-scroll">
                  <div className="panel-title">Context <span className="title-note">lives in this browser tab, sent in full every request</span></div>
                  <ContextPanel
                    preview={preview}
                    previewError={previewError}
                    session={session}
                    draft={draft}
                    turns={turns}
                    modelLoaded={modelLoaded}
                  />
                </div>
              </Panel>
              <PanelResizeHandle className="resize horizontal" />
              <Panel defaultSize={45} minSize={15}>
                <div className="panel-scroll">
                  <div className="panel-title">Cache <span className="title-note">lives on the server, reused on an exact prefix match</span></div>
                  <CachePanel
                    state={state}
                    preview={preview}
                    generation={generation}
                    busy={busy}
                    onClearCache={handleClearCache}
                  />
                </div>
              </Panel>
            </PanelGroup>
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
