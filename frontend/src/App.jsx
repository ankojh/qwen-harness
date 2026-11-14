import { useCallback, useEffect, useRef, useState } from 'react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import ConfigPanel from './ConfigPanel.jsx'
import InputPanel from './InputPanel.jsx'
import OutputPanel from './OutputPanel.jsx'
import EventLog from './EventLog.jsx'
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

function newRequestId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID()
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export default function App() {
  const [state, setState] = useState(null)
  const [session, setSession] = useState({
    mode: 'messages',
    messages: [{ ...EMPTY_MESSAGE }],
    raw_text: '',
  })
  const [preview, setPreview] = useState(null)
  const [previewError, setPreviewError] = useState(null)
  const [autoPreview, setAutoPreview] = useState(false)
  const [generation, setGeneration] = useState(null)
  const [tokens, setTokens] = useState([])
  const [activeGenId, setActiveGenId] = useState(null)
  const [events, setEvents] = useState([])
  const [sseState, setSseState] = useState('connecting')
  const [configError, setConfigError] = useState(null)
  const [actionError, setActionError] = useState(null)
  const [saving, setSaving] = useState(false)
  const activeGenRef = useRef(null)

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
          type === 'cache.clear'
        ) {
          refreshState().catch(() => {})
        }
      },
    })
    return close
  }, [adoptRunningGeneration, refreshGeneration, refreshState])

  const runPreview = useCallback(async () => {
    try {
      const report = await previewPrompt(session)
      setPreview(report)
      setPreviewError(null)
    } catch (error) {
      setPreviewError(formatError(error.detail || error))
    }
  }, [session])

  // Auto-refresh calls the tokenizer only, never the model.
  useEffect(() => {
    if (!autoPreview || !state?.model_loaded) return undefined
    let cancelled = false
    const timer = setTimeout(async () => {
      try {
        const report = await previewPrompt(session)
        if (!cancelled) {
          setPreview(report)
          setPreviewError(null)
        }
      } catch (error) {
        if (!cancelled) setPreviewError(formatError(error.detail || error))
      }
    }, 600)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [autoPreview, session, state?.model_loaded, state?.config])

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

  function handleClearConversation() {
    setSession((previous) => ({ ...previous, messages: [], raw_text: '' }))
    setPreview(null)
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
          <span className="muted">local MLX inspector</span>
        </div>
        <div className="topbar-meta">
          <span className={`pill ${modelLoaded ? 'pill-ok' : 'pill-off'}`}>
            {modelLoaded ? 'model loaded' : 'model idle'}
          </span>
          <span>forward calls <strong>{state.counters?.forward_calls_total ?? 0}</strong></span>
          <span>{state.active_generation ? `generating ${state.active_generation.id}` : 'idle'}</span>
          <span className={`pill sse-${sseState}`}>SSE {sseState}</span>
        </div>
      </header>

      {actionError && (
        <div className="banner">
          <span>{actionError}</span>
          <button type="button" className="mini" onClick={() => setActionError(null)}>dismiss</button>
        </div>
      )}

      <div className="main">
        <PanelGroup direction="vertical" style={{ height: '100%' }}>
          <Panel defaultSize={76} minSize={30}>
            <PanelGroup direction="horizontal" style={{ height: '100%' }}>
              <Panel defaultSize={30} minSize={18}>
                <div className="panel-scroll">
                  <div className="panel-title">Configuration</div>
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
                </div>
              </Panel>

              <PanelResizeHandle className="resize vertical" />

              <Panel defaultSize={35} minSize={20}>
                <div className="panel-scroll">
                  <div className="panel-title">Input &amp; prompt</div>
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
                </div>
              </Panel>

              <PanelResizeHandle className="resize vertical" />

              <Panel defaultSize={35} minSize={20}>
                <div className="panel-scroll">
                  <div className="panel-title">Output</div>
                  <OutputPanel
                    generation={generation}
                    tokens={tokens}
                    running={running}
                    previewHash={preview?.prompt_hash}
                    modelLoaded={modelLoaded}
                    thinkStart={state.model?.think_start}
                    thinkEnd={state.model?.think_end}
                    onGenerate={handleGenerate}
                    onStop={handleStop}
                    onClearOutput={handleClearOutput}
                    onClearConversation={handleClearConversation}
                    onAppend={handleAppend}
                  />
                </div>
              </Panel>
            </PanelGroup>
          </Panel>

          <PanelResizeHandle className="resize horizontal" />

          <Panel defaultSize={24} minSize={10}>
            <div className="panel-scroll log-panel">
              <div className="panel-title">Event log</div>
              <EventLog events={events} sseState={sseState} onClear={() => setEvents([])} />
            </div>
          </Panel>
        </PanelGroup>
      </div>
    </div>
  )
}
