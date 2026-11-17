// Thin fetch wrappers plus the one SSE subscription.
//
// Every call is a plain same-origin /api request (the Vite dev server proxies
// it, and FastAPI serves the built UI in production).  Errors carry the
// server's `detail` so the UI can show validation messages verbatim.

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    let payload = null
    try {
      payload = await response.json()
    } catch {
      payload = await response.text()
    }
    const detail = payload && payload.detail !== undefined ? payload.detail : payload
    const error = new Error(formatError(detail) || `HTTP ${response.status}`)
    error.status = response.status
    error.detail = detail
    throw error
  }
  return response.json()
}

export function formatError(error) {
  if (error == null) return ''
  if (typeof error === 'string') return error
  if (Array.isArray(error)) {
    return error
      .map((item) => {
        const path = Array.isArray(item.loc) ? item.loc.slice(1).join('.') : ''
        return `${path ? `${path}: ` : ''}${item.msg || JSON.stringify(item)}`
      })
      .join('; ')
  }
  if (error.message) return error.message
  return JSON.stringify(error)
}

export const getState = () => request('/api/state')
export const putConfig = (patch) =>
  request('/api/config', { method: 'PUT', body: JSON.stringify(patch) })
export const resetConfig = () => request('/api/config/reset', { method: 'POST' })
export const modelAction = (action) =>
  request(`/api/model/${action}`, { method: 'POST' })
export const previewPrompt = (session) =>
  request('/api/prompt/preview', {
    method: 'POST',
    body: JSON.stringify({ session, include_pieces: true }),
  })
export const startGeneration = (clientRequestId, session) =>
  request('/api/generate', {
    method: 'POST',
    body: JSON.stringify({ client_request_id: clientRequestId, session }),
  })
export const stopGeneration = () =>
  request('/api/generate/stop', { method: 'POST' })
export const getGeneration = (id) => request(`/api/generations/${id}`)
export const clearCache = () => request('/api/cache/clear', { method: 'POST' })
export const getCacheTokens = () => request('/api/cache/tokens')
export const getCacheEntry = (id) => request(`/api/cache/entries/${id}`)
export const diffCacheEntry = (entryId, tokenIds) =>
  request('/api/cache/diff', {
    method: 'POST',
    body: JSON.stringify({ entry_id: entryId, token_ids: tokenIds }),
  })

// Every named event the backend emits.  EventSource only fires onmessage for
// unnamed events, so each type is registered explicitly.
export const EVENT_TYPES = [
  'server.started',
  'server.stopped',
  'config.changed',
  'model.loading',
  'model.loaded',
  'model.unloading',
  'model.unloaded',
  'model.error',
  'generation.started',
  'generation.finished',
  'generation.stop_requested',
  'prefill.chunk',
  'token',
  'cache.lookup',
  'cache.checkpoint',
  'cache.insert',
  'cache.evict',
  'cache.clear',
  'cache.verify',
  'runtime.wired_limit',
  'runtime.mlx_cache_limit',
  'resync',
]

export function subscribeEvents({ onEvent, onOpen, onError }) {
  const source = new EventSource('/api/events')
  source.onopen = () => onOpen && onOpen()
  source.onerror = () => onError && onError()
  EVENT_TYPES.forEach((type) => {
    source.addEventListener(type, (message) => {
      let data = {}
      try {
        data = JSON.parse(message.data)
      } catch {
        data = { raw: message.data }
      }
      onEvent(type, data, Number(message.lastEventId))
    })
  })
  return () => source.close()
}
