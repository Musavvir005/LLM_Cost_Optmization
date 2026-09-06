/**
 * AD-BoN Gateway — Frontend API Client
 *
 * Supports:
 * - Environment-configured backend URL (VITE_API_BASE_URL)
 * - Safe response handling (graceful handling of HTML 404/500 pages without JSON crash)
 * - Automatic Bearer authentication
 */

function getDefaultApiBase() {
  const envUrl = (import.meta.env.VITE_API_BASE_URL || '').trim()
  if (envUrl) {
    return envUrl.replace(/\/+$/, '')
  }
  // In production (Vercel or any non-localhost host), automatically point to live Render backend
  if (import.meta.env.PROD || (typeof window !== 'undefined' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1')) {
    return 'https://llm-cost-optmization.onrender.com'
  }
  return '/api'
}

export const API_BASE_URL = getDefaultApiBase()

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

    // HTML / non-JSON response from hosting provider (e.g. Vercel 404/405/502)
    const text = await res.text().catch(() => '')
    if (res.status === 404 || res.status === 405) {
      throw new Error(
        `Backend not connected (HTTP ${res.status} at ${url}). Vercel only hosts the frontend. Please deploy the Python backend (e.g. on Render) and set VITE_API_BASE_URL in Vercel.`
      )
    }
    throw new Error(`Server returned HTTP ${res.status}: ${text.slice(0, 100) || 'Unrecognized response'}`)
  }

  if (isJson) {
    return await res.json()
  }

  // If the server returned 200 OK with HTML (e.g. Vercel rewriting /api/* to index.html), the backend is NOT connected
  throw new Error(
    `Received HTML instead of API response from ${url}. Please deploy your Python backend on Render and configure VITE_API_BASE_URL in your Vercel project.`
  )
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
