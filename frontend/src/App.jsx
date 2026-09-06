import React, { useState, useCallback, useEffect } from 'react'
import FloatingAssistant from './components/FloatingAssistant'
import QueryPanel from './components/QueryPanel'
import ResultPanel from './components/ResultPanel'
import BenchmarkPanel from './components/BenchmarkPanel'
import CachePanel from './components/CachePanel'
import HealthPanel from './components/HealthPanel'
import { checkHealth, routeQuery } from './api'
import './index.css'

const NAV = [
  { id: 'route', icon: '⚡', label: 'Route a Query' },
  { id: 'benchmark', icon: '📊', label: 'Benchmark Dashboard' },
  { id: 'cache', icon: '🗄️', label: 'Cache Stats' },
  { id: 'health', icon: '💚', label: 'System Health' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('route')
  const [result, setResult] = useState(() => {
    try {
      const saved = localStorage.getItem('adbon_last_route_result')
      return saved ? JSON.parse(saved) : null
    } catch { return null }
  })
  const [loading, setLoading] = useState(false)
  const [online, setOnline] = useState(null)
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('adbon-theme') || 'dark'
  })

  // Synchronize data-theme attribute on document root
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('adbon-theme', theme)
  }, [theme])

  // Persist last route result to localStorage
  useEffect(() => {
    if (result) {
      try {
        localStorage.setItem('adbon_last_route_result', JSON.stringify(result))
      } catch {}
    }
  }, [result])

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark')
  }

  // Poll API health every 5s
  useEffect(() => {
    const check = async () => {
      try {
        const data = await checkHealth()
        setOnline(data?.status === 'ok')
      } catch { setOnline(false) }
    }
    check()
    const id = setInterval(check, 5000)
    return () => clearInterval(id)
  }, [])

  const handleRoute = useCallback(async (payload) => {
    setLoading(true)
    setResult(null)
    try {
      const data = await routeQuery(payload)
      setResult(data)
      setActiveTab('route')
    } catch (e) {
      setResult({ error: e.message })
    } finally {
      setLoading(false)
    }
  }, [])

  return (
    <div className={`app-shell ${theme}-theme`}>
      {/* Top bar */}
      <header className="topbar">
        <div className="topbar-logo">
          <div className="logo-icon">⚡</div>
          <span>AD-BoN Optimizer</span>
          <span className="topbar-badge">A.m.2</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
          {/* Light / Dark Mode Toggle Button */}
          <button
            id="theme-toggle-btn"
            onClick={toggleTheme}
            className={`theme-toggle-btn ${theme === 'dark' ? 'btn-light-mode' : 'btn-dark-mode'}`}
            title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} mode`}
            aria-label="Toggle theme"
          >
            <span className="theme-toggle-icon">{theme === 'dark' ? '☀️' : '🌙'}</span>
            <span className="theme-toggle-text">{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
          </button>

          <div className="topbar-status">
            <div className={`status-dot ${online === false ? 'offline' : ''}`} />
            {online === null ? 'Connecting...' : online ? ' Authenticated' : 'Offline'}
          </div>
        </div>
      </header>

      <div className="main-layout">
        {/* Sidebar */}
        <nav className="sidebar">
          <div className="sidebar-section-label">Navigation</div>
          {NAV.map(n => (
            <button
              key={n.id}
              className={`nav-item ${activeTab === n.id ? 'active' : ''}`}
              onClick={() => setActiveTab(n.id)}
            >
              <span className="nav-icon">{n.icon}</span>
              {n.label}
            </button>
          ))}

          {/* Security Status Section */}
          <div className="sidebar-section-label" style={{ marginTop: '1rem' }}>Security Status</div>
          <div className="sidebar-security-badge">
            <div className="sidebar-security-title">
              🛡️ Security Layer
            </div>
            <div className="sidebar-security-item">&bull; Authentication: ENABLED</div>
            <div className="sidebar-security-item">&bull; Signal Integrity: ENABLED</div>
            <div className="sidebar-security-item">&bull; Replay Protection: ENABLED</div>
          </div>

          <div className="sidebar-section-label" style={{ marginTop: '1rem' }}>Routing Tiers</div>
          {[
            { label: 'Fast Tier', model: 'Ollama (phi3 / gemma3)', color: 'var(--tier-junior)', desc: '$0.00 / 1M' },
            { label: 'Balanced Tier', model: 'Groq (qwen-27b / compound)', color: 'var(--tier-medium)', desc: '$0.59 / 1M' },
            { label: 'Frontier Tier', model: 'Google Gemini Pro', color: 'var(--tier-senior)', desc: '$1.25 / 1M' },
            { label: 'Instant Cache', model: 'Semantic Vector Memory', color: 'var(--tier-cache)', desc: '$0.00 / Instant' },
          ].map(t => (
            <div key={t.label} className="sidebar-tier-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: t.color, flexShrink: 0 }} />
                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>{t.label}</span>
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', paddingLeft: 16, marginTop: 2 }}>{t.model}</div>
              <div style={{ fontSize: '0.68rem', color: t.color, paddingLeft: 16, fontWeight: 600 }}>{t.desc}</div>
            </div>
          ))}
        </nav>

        {/* Main content */}
        <main className="content-area">
          {activeTab === 'route' && (
            <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <QueryPanel onSubmit={handleRoute} loading={loading} />
              <ResultPanel result={result} loading={loading} />
            </div>
          )}
          {activeTab === 'benchmark' && <BenchmarkPanel />}
          {activeTab === 'cache' && <CachePanel />}
          {activeTab === 'health' && <HealthPanel />}
        </main>
      </div>

      {/* Floating AI Assistant docked at bottom right */}
      <FloatingAssistant />
    </div>
  )
}
