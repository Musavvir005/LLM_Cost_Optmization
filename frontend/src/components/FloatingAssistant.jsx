import React, { useState, useEffect, useRef, useCallback } from 'react'

const BTN_SIZE = 52          // circle diameter in px
const CHAT_W  = 440
const CHAT_H  = 620
const MARGIN  = 16           // min distance from screen edge

export default function FloatingAssistant() {
  const [isOpen, setIsOpen] = useState(false)
  const [message, setMessage] = useState('')
  const [conversation, setConversation] = useState(() => {
    try {
      const saved = localStorage.getItem('adbon_assistant_history')
      return saved ? JSON.parse(saved) : []
    } catch { return [] }
  })
  const [currentResult, setCurrentResult] = useState(() => {
    try {
      const saved = localStorage.getItem('adbon_assistant_result')
      return saved ? JSON.parse(saved) : null
    } catch { return null }
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [copiedIndex, setCopiedIndex] = useState(null)
  const [showTelemetry, setShowTelemetry] = useState(false)

  // ── Drag state ──────────────────────────────────────────────────────────
  const [pos, setPos] = useState({ x: window.innerWidth - BTN_SIZE - 24, y: window.innerHeight - BTN_SIZE - 24 })
  const dragging = useRef(false)
  const dragOffset = useRef({ x: 0, y: 0 })

  const clamp = useCallback((x, y) => {
    const maxX = window.innerWidth  - BTN_SIZE - MARGIN
    const maxY = window.innerHeight - BTN_SIZE - MARGIN
    return {
      x: Math.max(MARGIN, Math.min(x, maxX)),
      y: Math.max(MARGIN, Math.min(y, maxY)),
    }
  }, [])

  const onMouseDown = useCallback((e) => {
    if (isOpen) return          // don't drag when chat window is open
    dragging.current = true
    dragOffset.current = { x: e.clientX - pos.x, y: e.clientY - pos.y }
    e.preventDefault()
  }, [isOpen, pos])

  useEffect(() => {
    const onMouseMove = (e) => {
      if (!dragging.current) return
      setPos(clamp(e.clientX - dragOffset.current.x, e.clientY - dragOffset.current.y))
    }
    const onMouseUp = () => { dragging.current = false }
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup',   onMouseUp)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup',   onMouseUp)
    }
  }, [clamp])

  // Keep inside screen when window resizes
  useEffect(() => {
    const onResize = () => setPos(prev => clamp(prev.x, prev.y))
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [clamp])

  // ── Persists ─────────────────────────────────────────────────────────────
  const messagesEndRef = useRef(null)

  useEffect(() => {
    try { localStorage.setItem('adbon_assistant_history', JSON.stringify(conversation)) } catch {}
  }, [conversation])

  useEffect(() => {
    if (currentResult) {
      try { localStorage.setItem('adbon_assistant_result', JSON.stringify(currentResult)) } catch {}
    }
  }, [currentResult])

  useEffect(() => {
    if (isOpen) messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversation, loading, isOpen])

  // ── Helpers ───────────────────────────────────────────────────────────────
  const quickPrompts = [
    'Write a Python function to check whether a number is prime.',
    'Explain recursion with a short example.',
    'What are the primary architectural advantages of AD-BoN?',
  ]

  const handleSend = async (customPrompt) => {
    const queryText = (typeof customPrompt === 'string' ? customPrompt : message).trim()
    if (!queryText || loading) return
    setLoading(true); setError(null)
    const updatedHistory = [...conversation, { role: 'user', content: queryText }]
    setConversation(updatedHistory); setMessage('')
    try {
      const headers = { 'Content-Type': 'application/json', 'Authorization': 'Bearer adbon-sec-key-2026-demo' }
      let res = await fetch('/api/assistant/chat', {
        method: 'POST', headers,
        body: JSON.stringify({ message: queryText, conversation: conversation.slice(-10), target_quality: 0.76 }),
      })
      if (!res.ok && res.status === 404) {
        res = await fetch('http://127.0.0.1:5000/api/assistant/chat', {
          method: 'POST', headers,
          body: JSON.stringify({ message: queryText, conversation: conversation.slice(-10), target_quality: 0.76 }),
        })
      }
      const data = await res.json()
      if (!res.ok) throw new Error(data.message || data.error || `HTTP error ${res.status}`)
      setCurrentResult(data)
      setConversation([...updatedHistory, { role: 'assistant', content: data.answer, routing: data.routing, cost: data.cost, performance: data.performance, cache: data.cache }])
    } catch (err) {
      setError(err.message || 'Failed to generate response.')
    } finally { setLoading(false) }
  }

  const handleCopy = (text, index) => {
    navigator.clipboard.writeText(text); setCopiedIndex(index)
    setTimeout(() => setCopiedIndex(null), 2000)
  }
  const clearChat = () => { setConversation([]); setCurrentResult(null); setError(null); setMessage('') }

  // ── Chat window position — anchored near the button, clamped to screen ───
  const chatX = Math.min(pos.x, window.innerWidth  - CHAT_W  - MARGIN)
  const chatY = Math.max(MARGIN, Math.min(pos.y - CHAT_H - 10, window.innerHeight - CHAT_H - MARGIN))

  return (
    <>
      {/* ── Floating Circle Button ── */}
      {!isOpen && (
        <button
          id="floating-assistant-btn"
          className="floating-circle-btn"
          style={{ left: pos.x, top: pos.y }}
          onMouseDown={onMouseDown}
          onClick={() => setIsOpen(true)}
          title="Open AI Assistant (drag to move)"
          aria-label="Open AI Assistant"
        >
          <span className="floating-circle-icon">🤖</span>
          <span className="floating-online-dot" />
        </button>
      )}

      {/* ── Floating Chat Window ── */}
      {isOpen && (
        <div
          className="floating-chat-window fade-in"
          id="floating-chat-window"
          style={{ left: chatX, top: chatY, width: CHAT_W }}
        >
          {/* Header */}
          <div className="floating-chat-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div className="floating-avatar">🤖</div>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <strong style={{ fontSize: '0.9rem', color: 'var(--text-primary)', fontWeight: 700 }}>AI Assistant</strong>
                  <span className="pill pill-hit" style={{ fontSize: '0.65rem', padding: '1px 6px' }}>AD-BoN</span>
                </div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Intelligent Router • Groq LLMs</div>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              {conversation.length > 0 && (
                <button onClick={clearChat} className="floating-header-action-btn" title="Clear history">🗑️</button>
              )}
              <button onClick={() => setIsOpen(false)} className="floating-header-close-btn" title="Minimize" aria-label="Close">✕</button>
            </div>
          </div>

          {/* Telemetry Bar */}
          {currentResult && (
            <div className="floating-telemetry-bar">
              <div onClick={() => setShowTelemetry(v => !v)} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', fontSize: '0.72rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span>⚡</span>
                  <span style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>{currentResult.routing?.groq_model || 'Groq'}</span>
                  <span style={{ color: 'var(--text-muted)' }}>• {currentResult.routing?.selected_tier?.toUpperCase()} TIER</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>{currentResult.cost?.savings_percent}% saved</span>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.68rem' }}>{showTelemetry ? '▲' : '▼'}</span>
                </div>
              </div>
              {showTelemetry && (
                <div className="floating-telemetry-details fade-in">
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Target Quality:</span><strong>{(currentResult.routing?.target_quality * 100).toFixed(0)}%</strong></div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Estimated Cost:</span><strong style={{ color: 'var(--accent-green)' }}>${currentResult.cost?.estimated_cost?.toFixed(6)}</strong></div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Latency:</span><strong>{currentResult.performance?.latency_ms} ms</strong></div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Prefix Cache:</span><strong style={{ color: currentResult.cache?.hit ? 'var(--accent-cyan)' : 'var(--text-muted)' }}>{currentResult.cache?.hit ? 'HIT (50% discount)' : 'MISS'}</strong></div>
                </div>
              )}
            </div>
          )}

          {/* Messages */}
          <div className="floating-messages-area">
            {conversation.length === 0 && (
              <div className="floating-welcome-state">
                <div style={{ fontSize: '2rem', marginBottom: 6 }}>🤖</div>
                <strong style={{ fontSize: '0.88rem', color: 'var(--text-primary)' }}>How can I help you today?</strong>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.5 }}>
                  Ask any question. AD-BoN routes it to the optimal Groq model tier automatically.
                </p>
                <div className="floating-quick-prompts">
                  <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>Suggestions:</span>
                  {quickPrompts.map((p, i) => (
                    <button key={i} className="floating-chip" onClick={() => handleSend(p)} disabled={loading}>{p}</button>
                  ))}
                </div>
              </div>
            )}

            {conversation.map((msg, idx) => (
              <div key={idx} className={`floating-message-row ${msg.role === 'user' ? 'user-row' : 'assistant-row'}`}>
                {msg.role === 'assistant' && <div className="msg-avatar">🤖</div>}
                <div className={`floating-bubble ${msg.role === 'user' ? 'user-bubble' : 'assistant-bubble'}`}>
                  {msg.role === 'assistant' && msg.routing && (
                    <div className="msg-model-tag">
                      <span>⚡ {msg.routing.groq_model}</span>
                      <span className="msg-tier-pill">{msg.routing.selected_tier}</span>
                    </div>
                  )}
                  <div className="bubble-content">{msg.content}</div>
                  {msg.role === 'assistant' && (
                    <div className="bubble-actions">
                      <button onClick={() => handleCopy(msg.content, idx)} className="bubble-copy-btn">
                        {copiedIndex === idx ? '✓ Copied' : '📋 Copy'}
                      </button>
                      {msg.cost?.savings_percent && <span className="bubble-savings-tag">{msg.cost.savings_percent}% cheaper</span>}
                    </div>
                  )}
                </div>
                {msg.role === 'user' && <div className="msg-avatar user-avatar-icon">👤</div>}
              </div>
            ))}

            {loading && (
              <div className="floating-message-row assistant-row">
                <div className="msg-avatar">🤖</div>
                <div className="floating-bubble assistant-bubble loading-bubble">
                  <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
                  <span>Evaluating via AD-BoN &amp; generating on Groq…</span>
                </div>
              </div>
            )}

            {error && <div className="floating-error-banner">⚠️ {error}</div>}
            <div ref={messagesEndRef} />
          </div>

          {/* Footer */}
          <div className="floating-chat-footer">
            <div className="floating-input-wrapper">
              <textarea
                id="floating-assistant-input"
                className="floating-input-textarea"
                rows={2}
                placeholder="Ask AI Assistant anything..."
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() } }}
                disabled={loading}
              />
              <button id="floating-send-btn" className="floating-send-btn" onClick={() => handleSend()} disabled={loading || !message.trim()}>
                {loading ? <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> : '🚀'}
              </button>
            </div>
            <div className="floating-input-tip">Press <strong>Enter</strong> to send • <strong>Shift+Enter</strong> for newline</div>
          </div>
        </div>
      )}
    </>
  )
}
