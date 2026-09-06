/**
 * AD-BoN Gateway — Frontend API Client
 *
 * Supports:
 * - Environment-configured backend URL (VITE_API_BASE_URL)
 * - Safe response handling (graceful handling of HTML 404/500 pages without JSON crash)
 * - Automatic Bearer authentication
 */

// 1. Resolve base URL: VITE_API_BASE_URL (Render/Railway/ngrok) or default '/api'
const rawBase = (import.meta.env.VITE_API_BASE_URL || '').trim()
export const API_BASE_URL = rawBase ? rawBase.replace(/\/+$/, '') : '/api'

export const API_KEY = import.meta.env.VITE_API_KEY || 'adbon-sec-key-2026-demo'

/**
 * Build the full request URL for a given path.
 * If API_BASE_URL is '/api', '/health' -> '/api/health'
 * If API_BASE_URL is 'https://backend.onrender.com', '/health' -> 'https://backend.onrender.com/health'
 */
export function getApiUrl(endpoint) {
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`
  return `${API_BASE_URL}${cleanEndpoint}`
}

/**
 * Resilient fetch wrapper that prevents JSON parse crashes on HTML error pages
 */
export async function apiFetch(endpoint, options = {}) {
  const url = getApiUrl(endpoint)
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_KEY}`,
    ...(options.headers || {}),
  }

  let res
  try {
    res = await fetch(url, { ...options, headers })
  } catch (err) {
    throw new Error(
      `Cannot connect to backend at ${url}. Ensure the Python backend is running and CORS is enabled.`
    )
  }

  const contentType = res.headers.get('content-type') || ''
  const isJson = contentType.includes('application/json')

  if (!res.ok) {
    if (isJson) {
      const errData = await res.json().catch(() => ({}))
      const msg = errData.detail || errData.message || errData.error || `HTTP ${res.status}: ${res.statusText}`
      throw new Error(msg)
    }

    // HTML / non-JSON response from hosting provider (e.g. Vercel 404/502)
    const text = await res.text().catch(() => '')
    if (res.status === 404) {
      throw new Error(
        `Backend service returned 404 at ${url}. If using Vercel, set VITE_API_BASE_URL in Vercel project settings to your deployed Python backend URL.`
      )
    }
    throw new Error(`Server returned HTTP ${res.status}: ${text.slice(0, 100) || 'Unrecognized response'}`)
  }

  if (isJson) {
    return await res.json()
  }

  const rawText = await res.text()
  try {
    return JSON.parse(rawText)
  } catch {
    return { text: rawText }
  }
}

// ── Service Endpoints ───────────────────────────────────────────

export async function checkHealth() {
  return await apiFetch('/health')
}

export async function routeQuery(payload) {
  return await apiFetch('/route', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function getCacheStats() {
  return await apiFetch('/cache/stats')
}

export async function runBenchmark() {
  return await apiFetch('/benchmark')
}

export async function sendAssistantChat(payload) {
  return await apiFetch('/assistant/chat', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}
