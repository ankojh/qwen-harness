import { useEffect, useMemo, useState } from 'react'
import Pipeline from './Pipeline.jsx'
import Caption from './Caption.jsx'
import TapeMap from './TapeMap.jsx'
import Lens from './Lens.jsx'
import Inspector from './Inspector.jsx'
import Timeline from './Timeline.jsx'
import { buildMapModel } from './mapModel.js'
import { stopsMatching } from './turnDiffs.js'
import { formatBytes } from './ConfigPanel.jsx'
import {
  deleteCacheEntry,
  formatError,
  getCacheEntry,
  getCacheTokens,
  previewPrompt,
  relabelEntry,
  restoreEntry,
  saveCheckpoint,
} from './api.js'

// The right column: Pipeline → caption → cache header → map → Lens →
// Inspector → folded Timeline.  Holds the selection (message / entry /
// token) and the staged message edit, and composes the view model for the
// current mode: the next request (typing), the submitted request (running),
// or a historical turn (read-only).

// One edit-log line per message-level change when the staged copy is applied
// (moved from ContextPanel).
function describeChanges(before, after) {
  const oneLine = (text, limit) => {
    const flat = (text || '').replace(/\s+/g, ' ').trim()
    return flat.length > limit ? `${flat.slice(0, limit)}…` : flat
  }
  const lines = []
  const length = Math.max(before.length, after.length)
  for (let index = 0; index < length; index += 1) {
    const prev = before[index]
    const next = after[index]
    const label = `message #${index + 1}`
    if (prev && next && prev !== next) {
      const changes = []
      if (prev.content !== next.content) {
        changes.push(`${JSON.stringify(oneLine(prev.content, 30))} → ${JSON.stringify(oneLine(next.content, 30))}`)
      }
      if (prev.role !== next.role) changes.push(`role ${prev.role} → ${next.role}`)
      if ((prev.enabled !== false) !== (next.enabled !== false)) {
        changes.push(next.enabled === false ? 'switched off' : 'switched on')
      }
      if ((prev.reasoning_content || '') !== (next.reasoning_content || '')) {
        changes.push(next.reasoning_content ? 'reasoning set' : 'reasoning removed')
      }
      if (changes.length) lines.push(`edited ${label} (${next.role}): ${changes.join(', ')}`)
    } else if (prev && !next) {
      lines.push(`deleted ${label} (${prev.role}) “${oneLine(prev.content, 40)}”`)
    } else if (!prev && next) {
      lines.push(`inserted ${label} (${next.role}) “${oneLine(next.content, 40)}”`)
    }
  }
  return lines
}

export default function Stage({
  state,
  session,
  draft,
  preview,
  previewError,
  generation,
  tokens,
  running,
  busy,
  modelLoaded,
  turns,
  viewTurn,
  viewTurnId,
  prevTurn,
  pendingEdits,
  systemPrompt,
  runProgress,
  hoveredMessage,
  bubbleClick,
  onViewTurnChange,
  onApplyStaged,
  onCacheAction,
  onConfigChange,
  onClearCache,
  onEditingChange,
  onLoadModel,
}) {
  const [selectedEntryId, setSelectedEntryId] = useState(null)
  const [selectedMessageIndex, setSelectedMessageIndex] = useState(null)
  const [selectedToken, setSelectedToken] = useState(null)
  const [lensOverride, setLensOverride] = useState(null)
  const [comparePrev, setComparePrev] = useState(false)
  const [cacheTokens, setCacheTokens] = useState(null)
  const [entryDetail, setEntryDetail] = useState(null)
  const [entryDetailError, setEntryDetailError] = useState(null)
  const [detailVersion, setDetailVersion] = useState(0)
  const [relabelPosition, setRelabelPosition] = useState(null)
  const [relabelText, setRelabelText] = useState('')
  const [relabelError, setRelabelError] = useState(null)
  const [relabelBusy, setRelabelBusy] = useState(false)
  const [restoreBusy, setRestoreBusy] = useState(false)
  const [checkpointBusy, setCheckpointBusy] = useState(false)
  const [staged, setStaged] = useState(null)
  const [beforePreview, setBeforePreview] = useState(null)
  const [stagedPreview, setStagedPreview] = useState(null)
  const [stagedPreviewError, setStagedPreviewError] = useState(null)
  const [pulseEntryId, setPulseEntryId] = useState(null)

  const historical = Boolean(viewTurn)
  const config = state?.config || {}
  const reuseEnabled = config.cache_reuse !== false
  const editing = staged != null

  // A bubble click in the chat opens that message here.
  useEffect(() => {
    if (bubbleClick?.index != null && bubbleClick.index >= 0 && bubbleClick.index < session.messages.length) {
      selectMessage(bubbleClick.index)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bubbleClick?.seq])

  // Pulse the chosen bar once when the lookup lands.
  useEffect(() => {
    if (runProgress?.lookup?.reused_tokens > 0) {
      setPulseEntryId(runProgress.lookup.entry_id)
      const timer = setTimeout(() => setPulseEntryId(null), 900)
      return () => clearTimeout(timer)
    }
    return undefined
  }, [runProgress?.lookup])

  // Tell App whether a staged edit is open (blocks Send).
  useEffect(() => {
    onEditingChange?.(editing)
    return () => onEditingChange?.(false)
  }, [editing, onEditingChange])

  // Live cache token lists (safe during generation).  Historical snapshots
  // carry their own.
  useEffect(() => {
    if (historical || !modelLoaded) {
      setCacheTokens(null)
      return undefined
    }
    let cancelled = false
    getCacheTokens()
      .then((data) => { if (!cancelled) setCacheTokens(data.entries || []) })
      .catch(() => { if (!cancelled) setCacheTokens([]) })
    return () => { cancelled = true }
  }, [historical, modelLoaded, state?.cache])

  // Entry details for the inspector (409 while a generation runs).
  useEffect(() => {
    if (selectedEntryId == null || historical || running) {
      setEntryDetail(null)
      setEntryDetailError(null)
      return undefined
    }
    let cancelled = false
    getCacheEntry(selectedEntryId)
      .then((data) => { if (!cancelled) { setEntryDetail(data); setEntryDetailError(null) } })
      .catch((error) => { if (!cancelled) { setEntryDetail(null); setEntryDetailError(error.detail || String(error)) } })
    return () => { cancelled = true }
  }, [selectedEntryId, historical, running, detailVersion, state?.cache])

  // ---- what the map shows --------------------------------------------------
  // While a run is in flight the tape freezes on the submitted request: the
  // last preview computed for it (App stops auto-previewing during a run).
  const report = historical
    ? viewTurn.context || null
    : stagedPreview && !stagedPreviewError
      ? stagedPreview
      : preview

  const mergedEntries = useMemo(() => {
    if (historical) return viewTurn.cache?.entries || []
    const rows = state?.cache?.entries || []
    return (cacheTokens || []).map((entry) => ({ ...rows.find((row) => row.id === entry.id), ...entry }))
  }, [historical, viewTurn, cacheTokens, state?.cache])

  const prevEntries = prevTurn?.cache?.entries || null
  const generated = running ? tokens : []
  const lookup = historical
    ? viewTurn.cache?.last_lookup || null
    : running
      ? runProgress?.lookup || null
      : null
  const prefill = historical
    ? viewTurn.prefill || null
    : running
      ? null
      : generation?.stats?.prefill || null
  const progress = running ? runProgress?.prefill || null : null

  const model = useMemo(
    () =>
      buildMapModel({
        report,
        entries: mergedEntries,
        generated,
        lookup,
        prefill,
        progress,
        running: running && staged == null,
        reuseEnabled,
        historical,
        turnNumber: viewTurn?.n ?? null,
        prevEntries,
        prevIds: prevTurn?.context?.token_ids || null,
        comparePrev: comparePrev && !historical,
        historyCounts: historical ? { thinking: viewTurn?.thinking || 0, answer: (viewTurn?.generated || 0) - (viewTurn?.thinking || 0) } : null,
      }),
    [report, mergedEntries, generated, lookup, prefill, progress, running, staged, reuseEnabled, historical, viewTurn, prevEntries, prevTurn, comparePrev],
  )

  // The submitted request just finished and the preview hasn't refreshed yet:
  // show the actual numbers with a confirmation mark (the switch to the
  // next-request preview follows within about a second).
  const justFinished =
    !historical && !running && generation?.finish && generation.prompt &&
    preview?.prompt_hash === generation.prompt?.prompt_hash && staged == null
  const captionExtra = justFinished
    ? [
        { kind: 'ok', text: ` ✓ the real prefill restored exactly ${generation.stats?.prefill?.reused_tokens ?? 0} — the prediction held` },
      ]
    : []

  // Staged edit: what-if model replaces the map and the caption gains the
  // "reuse X → Y" line.
  const whatIf = staged != null && stagedPreview && !stagedPreviewError
    ? buildMapModel({
        report: stagedPreview,
        entries: mergedEntries,
        generated: [],
        lookup: null,
        prefill: null,
        progress: null,
        running: false,
        reuseEnabled,
        historical: false,
        prevEntries,
        prevIds: prevTurn?.context?.token_ids || null,
        comparePrev: false,
      })
    : null
  const whatIfModel = whatIf || model
  const beforeReuse = (beforePreview?.cache_prediction?.reused_tokens ?? preview?.cache_prediction?.reused_tokens) || 0
  const afterReuse = whatIf?.reused ?? model.reused

  const savedSummary = running
    ? runProgress?.saved || null
    : !historical && prevEntries
      ? { added: model.rows.filter((row) => row.isNew).length, evicted: model.ghosts.length }
      : historical
        ? null
        : null

  const lensModel = lensOverride != null ? { ...model, lensCenter: lensOverride } : model
  const selectedEntryRow = model.rows.find((row) => row.id === selectedEntryId) || null

  // ---- selection handlers --------------------------------------------------
  function selectEntry(id) {
    setSelectedEntryId(id)
    setSelectedMessageIndex(null)
    setSelectedToken(null)
    setLensOverride(null)
  }
  function selectMessage(index) {
    setSelectedMessageIndex(index)
    setSelectedEntryId(null)
    setSelectedToken(null)
  }
  function selectToken(position) {
    setSelectedToken(position)
    setSelectedEntryId(null)
    setSelectedMessageIndex(null)
  }
  function clearSelection() {
    setSelectedEntryId(null)
    setSelectedMessageIndex(null)
    setSelectedToken(null)
  }

  // ---- staged message edit -------------------------------------------------
  const stagedSession = useMemo(
    () => (staged ? { mode: 'messages', messages: draft.trim() ? [...staged, { role: 'user', content: draft.trim(), enabled: true, reasoning_content: null }] : staged } : null),
    [staged, draft],
  )

  useEffect(() => {
    if (!stagedSession || !modelLoaded) {
      setStagedPreview(null)
      setStagedPreviewError(null)
      setBeforePreview(null)
      return undefined
    }
    setStagedPreview(null)
    setStagedPreviewError(null)
    let cancelled = false
    const timer = setTimeout(async () => {
      try {
        const currentSession = {
          ...session,
          messages: draft.trim()
            ? [...session.messages, { role: 'user', content: draft.trim(), enabled: true, reasoning_content: null }]
            : session.messages,
        }
        const [after, before] = await Promise.all([
          previewPrompt(stagedSession),
          previewPrompt(currentSession),
        ])
        if (!cancelled) {
          setBeforePreview(before)
          setStagedPreview(after)
          setStagedPreviewError(after.ok === false ? after.error || 'Invalid staged context' : null)
        }
      } catch (error) {
        if (!cancelled) setStagedPreviewError(formatError(error.detail || error))
      }
    }, 300)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [stagedSession, modelLoaded, session, draft, state?.cache, state?.config])

  function beginEdit(index) {
    if (historical || running) return
    onViewTurnChange?.(null)
    setStaged(session.messages.map((message) => ({ ...message })))
    setSelectedMessageIndex(index)
  }

  function updateStaged(index, patch) {
    if (staged == null) beginEdit(index)
    setStaged((current) => (current ?? session.messages).map((message, position) =>
      position === index ? { ...message, ...patch } : message,
    ))
  }

  function moveStaged(index, direction) {
    const target = index + direction
    if (staged == null || target < 0 || target >= staged.length) return
    const next = [...staged]
    ;[next[index], next[target]] = [next[target], next[index]]
    setStaged(next)
  }

  function deleteStaged(index) {
    if (staged == null) return
    const next = staged.filter((_, position) => position !== index)
    setStaged(next)
    if (next.length === 0) setSelectedMessageIndex(null)
  }

  function insertAfter(index) {
    if (staged == null) beginEdit(index)
    setStaged((current) => {
      const base = current ?? session.messages
      return [
        ...base.slice(0, index + 1),
        { role: 'user', content: '', reasoning_content: null, enabled: true },
        ...base.slice(index + 1),
      ]
    })
  }

  // The checkpoint boundary number: submitted messages (enabled only), with
  // the configured system prompt as message 1 when set.
  function submittedIndexOf(index) {
    const before = (staged ?? session.messages).slice(0, index).filter((message) => message.enabled !== false).length
    return before + 1 + (systemPrompt ? 1 : 0)
  }

  function checkpointAt(index) {
    setCheckpointBusy(true)
    onCacheAction(
      () => saveCheckpoint(stagedSession, submittedIndexOf(index)),
      (result) =>
        result.used_existing
          ? `saved a checkpoint after message ${submittedIndexOf(index)}: entry #${result.entry_id} already held it (marked used)`
          : `saved a checkpoint after message ${submittedIndexOf(index)}: entry #${result.entry_id}, ${result.tokens} tokens`,
    )
      .catch(() => {})
      .finally(() => setCheckpointBusy(false))
  }

  function applyStaged() {
    if (staged == null) return
    const lines = describeChanges(session.messages, staged)
    onApplyStaged({ mode: 'messages', messages: staged }, lines)
    setStaged(null)
    setSelectedMessageIndex(null)
    setStagedPreview(null)
    setBeforePreview(null)
  }

  function discardStaged() {
    setStaged(null)
    setSelectedMessageIndex(null)
    setStagedPreview(null)
    setBeforePreview(null)
  }

  // ---- entry actions -------------------------------------------------------
  function handleDeleteEntry() {
    if (selectedEntryRow == null) return
    onCacheAction(
      () => deleteCacheEntry(selectedEntryRow.id),
      `deleted cache entry #${selectedEntryRow.id} (${selectedEntryRow.kind}, ${selectedEntryRow.len} tokens)`,
    ).catch(() => {})
    clearSelection()
  }

  function openRelabel(position) {
    const token = entryDetail?.tokens?.[position]
    setRelabelPosition(position)
    setRelabelError(null)
    setRelabelText(token?.text ?? token?.piece ?? '')
  }

  function submitRelabel() {
    const position = relabelPosition
    if (position == null || selectedEntryId == null) return
    const token = entryDetail?.tokens?.[position]
    const before = (token?.text ?? token?.piece ?? '').trim()
    const after = relabelText.trim()
    setRelabelBusy(true)
    setRelabelError(null)
    onCacheAction(
      () => relabelEntry(selectedEntryId, position, relabelText),
      `relabelled token ${position} of entry #${selectedEntryId}: '${before}' → '${after}'`,
      { silent: true },
    )
      .then(() => {
        setRelabelPosition(null)
        setRelabelText('')
        setDetailVersion((previous) => previous + 1)
      })
      .catch((error) => setRelabelError(error.detail || String(error)))
      .finally(() => setRelabelBusy(false))
  }

  function handleRestore() {
    if (selectedEntryId == null || !entryDetail) return
    const count = (entryDetail.relabels || []).length
    setRestoreBusy(true)
    onCacheAction(
      () => restoreEntry(selectedEntryId),
      `restored entry #${selectedEntryId}'s original tokens (${count} relabel${count === 1 ? '' : 's'} undone)`,
    )
      .then(() => {
        setRelabelPosition(null)
        setDetailVersion((previous) => previous + 1)
      })
      .catch(() => {})
      .finally(() => setRestoreBusy(false))
  }

  // ---- the message selected for the inspector ------------------------------
  const inspectedMessage = selectedMessageIndex != null
    ? (staged?.[selectedMessageIndex] ?? session.messages[selectedMessageIndex])
    : null
  const changedMessage =
    staged != null && selectedMessageIndex != null &&
    (() => {
      const prev = session.messages[selectedMessageIndex]
      const next = staged[selectedMessageIndex]
      if (!prev || !next) return true
      return (
        prev.content !== next.content ||
        prev.role !== next.role ||
        (prev.enabled !== false) !== (next.enabled !== false) ||
        (prev.reasoning_content || '') !== (next.reasoning_content || '')
      )
    })()

  const stops = whatIf && beforePreview
    ? stopsMatching(cacheTokens || [], beforePreview.token_ids || [], whatIf.contextIds)
    : []

  const tokenPaneToken = selectedToken != null ? model.pieces[selectedToken] ?? null : null
  const tokenSpanRole = selectedToken != null
    ? model.tape.find((seg) => selectedToken >= seg.start && selectedToken < seg.end)?.role ?? null
    : null
  const tokenMessageIndex = selectedToken != null
    ? model.tape.find((seg) => selectedToken >= seg.start && selectedToken < seg.end)?.messageIndex ?? null
    : null

  const pane = selectedEntryId != null
    ? 'entry'
    : selectedMessageIndex != null
      ? 'message'
      : selectedToken != null
        ? 'token'
        : null

  const relabelPositions = (entryDetail?.relabels || []).map((item) => item.position)
  const entryPieces = entryDetail?.tokens || (preview?.cache_entries?.find((e) => e.id === selectedEntryId)?.token_window || [])

  // The cache header gauge.
  const utilization = Math.min(100, Math.round((state?.cache?.utilization || 0) * 100))

  const selectedEntryIsRow = selectedEntryRow != null
  const previewErrorNote = !historical && previewError && !running ? previewError : null

  return (
    <div className="stage">
      {historical && (
        <div className="row-between">
          <span className="historical-label">Historical snapshot · read-only</span>
          <button type="button" className="mini" onClick={() => onViewTurnChange(null)}>back to current</button>
        </div>
      )}
      {previewErrorNote && <p className="note note-error">{previewErrorNote}</p>}

      <Pipeline
        model={whatIfModel}
        lookupKnown={Boolean(runProgress) || !running}
        running={running && staged == null}
        savedSummary={savedSummary}
      />

      {whatIf ? (
        <p className="caption" aria-live="polite">
          <span className="g">reuse {beforeReuse} → {afterReuse}</span>
          <span> · </span>
          {whatIf.caption.map((part, index) => {
            const cls = { ok: 'g', info: 'b', bad: 'r', warn: 'w' }[part.kind]
            return part.kind === 'code'
              ? <code key={index}>{part.text}</code>
              : <span key={index} className={cls}>{part.text}</span>
          })}
        </p>
      ) : (
        <Caption model={model} extra={captionExtra} />
      )}

      <div className="cache-head row-between">
        <span className="muted small">
          <strong>CACHE</strong> · {model.rows.filter((row) => !row.ghost).length} saved ·
          {' '}{formatBytes(state?.cache?.total_bytes ?? 0)} of {formatBytes(state?.cache?.limit_bytes ?? 0)}
        </span>
        <span className="button-row wrap">
          <span className="muted small" title="cache memory used">
            <span className="budget-bar" style={{ display: 'inline-block', width: 90, height: 8, verticalAlign: 'middle' }}>
              <span className="seg cached" style={{ width: `${utilization}%`, display: 'block', height: '100%' }} />
            </span>
            {' '}{utilization}%
          </span>
          <label className="check" title="Whether requests look up and store prompt caches">
            <input
              type="checkbox"
              checked={reuseEnabled}
              disabled={busy || historical}
              onChange={(event) => onConfigChange({ cache_reuse: event.target.checked })}
            />
            reuse
          </label>
          <select
            aria-label="checkpoint mode"
            value={config.cache_checkpoints || 'last_message'}
            disabled={busy || historical}
            onChange={(event) => onConfigChange({ cache_checkpoints: event.target.value })}
            style={{ width: 'auto' }}
          >
            <option value="off">checkpoints off</option>
            <option value="last_message">last message</option>
            <option value="every_message">every message</option>
          </select>
          <button
            type="button"
            className={comparePrev ? 'active' : ''}
            disabled={historical || !prevTurn}
            title={prevTurn ? 'mark tokens that were not in the previous request with +' : 'send at least two turns to compare'}
            onClick={() => setComparePrev(!comparePrev)}
          >
            compare with last turn
          </button>
          <button
            type="button"
            className="mini"
            disabled={busy || historical || model.rows.length === 0}
            onClick={onClearCache}
          >
            Clear
          </button>
        </span>
      </div>

      <TapeMap
        model={whatIfModel}
        selectedEntryId={selectedEntryId}
        selectedToken={selectedToken}
        highlightMessage={hoveredMessage}
        pulseEntryId={pulseEntryId}
        onSelectEntry={selectEntry}
        onSelectToken={selectToken}
        onSelectMessage={(index) => (index == null ? clearSelection() : selectMessage(index))}
        onDragLens={setLensOverride}
      />

      <div className="stage-legend" aria-hidden="true">
        <span><span className="sw" style={{ background: 'color-mix(in srgb, var(--accent) 35%, transparent)' }} />your message</span>
        <span><span className="sw" style={{ background: 'color-mix(in srgb, var(--answer) 35%, transparent)' }} />model reply</span>
        <span><span className="sw tpl" />template tokens</span>
        <span><span className="sw think" />thinking (never sent again)</span>
        <span><span className="sw ok" />restored from cache</span>
        <span><span className="sw info" />computed now</span>
        <span><span className="sw bad" />entry continues past a difference</span>
      </div>

      {relabelPosition != null && (
        <div className="relabel-form">
          <span className="muted small">relabel token {relabelPosition} of entry #{selectedEntryId}:</span>
          <input
            value={relabelText}
            onChange={(event) => setRelabelText(event.target.value)}
            placeholder="exactly one token, e.g. “ Bob”"
            autoFocus
            onKeyDown={(event) => {
              if (event.key === 'Enter') submitRelabel()
              if (event.key === 'Escape') { setRelabelPosition(null); setRelabelError(null) }
            }}
          />
          <button type="button" className="mini primary" disabled={relabelBusy} onClick={submitRelabel}>relabel</button>
          <button type="button" className="mini" disabled={relabelBusy} onClick={() => { setRelabelPosition(null); setRelabelError(null) }}>cancel</button>
          {relabelError && <span className="note note-error">{relabelError}</span>}
          <span className="muted small">
            This changes token labels only. The computed model state stays as it was.
            One token only; the count never changes.
          </span>
        </div>
      )}

      <Lens
        model={lensModel}
        selectedEntryId={selectedEntryId}
        entryPieces={entryPieces}
        relabelPositions={relabelPositions}
        onContextTokenClick={historical ? undefined : selectToken}
        onEntryTokenClick={historical || running ? undefined : openRelabel}
      />

      <Inspector
        pane={pane}
        onClose={clearSelection}
        entryRow={selectedEntryRow}
        detail={entryDetail}
        detailError={entryDetailError}
        contextLen={model.total}
        historical={historical}
        running={running}
        busy={busy}
        onDeleteEntry={handleDeleteEntry}
        messageIndex={selectedMessageIndex}
        message={inspectedMessage}
        staged={staged}
        systemPrompt={systemPrompt}
        checkpointBusy={checkpointBusy}
        onBeginEdit={beginEdit}
        onUpdate={updateStaged}
        onMove={moveStaged}
        onDeleteMessage={deleteStaged}
        onInsertAfter={insertAfter}
        onCheckpoint={checkpointAt}
        onApply={applyStaged}
        onDiscard={discardStaged}
        stagedReady={Boolean(whatIf)}
        before={beforePreview}
        after={stagedPreview}
        stops={stops}
        changed={changedMessage}
        position={selectedToken}
        token={tokenPaneToken}
        spanRole={tokenSpanRole}
        reused={model.reused}
        containingMessageIndex={tokenMessageIndex}
        onSelectMessage={selectMessage}
        restoreBusy={restoreBusy}
        onRestore={handleRestore}
      />

      {selectedEntryRow && !historical && !running && relabelCountShown(entryDetail, selectedEntryRow) > 0 && (
        <div className="row-between">
          <span className="muted small">
            {relabelCountShown(entryDetail, selectedEntryRow)} token(s) relabelled on entry #{selectedEntryRow.id}.
          </span>
          <button type="button" className="mini" disabled={busy || restoreBusy} onClick={handleRestore}>
            Restore original labels
          </button>
        </div>
      )}

      {!historical && pendingEdits?.length > 0 && (
        <p className="muted small">
          Since the last turn you: {pendingEdits.join(' · ')}
        </p>
      )}

      <Timeline
        turns={turns}
        viewTurnId={viewTurnId}
        onSelect={(id) => { clearSelection(); setStaged(null); onViewTurnChange(id) }}
        disabled={editing || running}
      />

      {!modelLoaded && (
        <div className="note note-warn row-between">
          <span>The pipeline is waiting for the model.</span>
          <button type="button" className="mini" disabled={busy} onClick={onLoadModel}>Load model</button>
        </div>
      )}
    </div>
  )
}

function relabelCountShown(detail, row) {
  return detail?.relabels?.length ?? row?.relabelCount ?? 0
}
