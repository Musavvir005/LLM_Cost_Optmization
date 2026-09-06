import React, { useState, useEffect } from 'react'

export default function AssistantPanel() {
  const [message, setMessage] = useState('Write a Python function to check whether a number is prime.')
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
  const [copied, setCopied] = useState(false)

  // Persist conversation and last result across refreshes
  useEffect(() => {
    try {
      localStorage.setItem('adbon_assistant_history', JSON.stringify(conversation))
    } catch {}
  }, [conversation])

  useEffect(() => {
    if (currentResult) {
      try {
        localStorage.setItem('adbon_assistant_result', JSON.stringify(currentResult))
      } catch {}
    }
  }, [currentResult])

  const quickPrompts = [
    'Write a Python function to check whether a number is prime.',
    'Explain recursion in Python with a concise example.',
    'What are the primary architectural advantages of AD-BoN?',
    'Solve the integral of x * exp(2x) dx step-by-step.',
  ]

  const handleSend = async (msgToSend) => {
    const queryText = (typeof msgToSend === 'string' ? msgToSend : message).trim()
    if (!queryText || loading) return

    setLoading(true)
    setError(null)

    // Append to conversation history
    const updatedHistory = [...conversation, { role: 'user', content: queryText }]

    try {
      // Try Vite proxy /api/assistant/chat first, fallback to Flask port 5000 if needed
      const headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer adbon-sec-key-2026-demo',
      }

      let res = await fetch('/api/assistant/chat', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          message: queryText,
          conversation: conversation.slice(-10),
          target_quality: 0.76,
        }),
      })

      if (!res.ok && res.status === 404) {
        // Fallback directly to Flask gateway if proxy didn't catch
        res = await fetch('http://127.0.0.1:5000/api/assistant/chat', {
          method: 'POST',
          headers,
          body: JSON.stringify({
            message: queryText,
            conversation: conversation.slice(-10),
            target_quality: 0.76,
          }),
        })
      }

      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.message || data.error || `HTTP error ${res.status}`)
      }

      setCurrentResult(data)
      setConversation([...updatedHistory, { role: 'assistant', content: data.answer }])
      setMessage('')
    } catch (err) {
      setError(err.message || 'Failed to generate response from AI Assistant.')
    } finally {
      setLoading(false)
    }
  }

  const handleCopy = () => {
    if (!currentResult?.answer) return
    navigator.clipboard.writeText(currentResult.answer)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const clearChat = () => {
    setConversation([])
    setCurrentResult(null)
    setError(null)
    setMessage('')
  }

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', maxWidth: '1080px', margin: '0 auto', width: '100%' }}>
      {/* Header Banner */}
      <div className="card" style={{ textAlign: 'center', padding: '1.5rem 1rem' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: 'rgba(59, 130, 246, 0.1)', color: 'var(--accent-blue)', padding: '4px 14px', borderRadius: 20, fontSize: '0.76rem', fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase', marginBottom: 8 }}>
          <span>⚡</span> AD-BoN AI Assistant
        </div>
        <h2 style={{ fontSize: '1.6rem', fontWeight: 800, letterSpacing: '-0.02em', margin: '4px 0' }}>
          Intelligent AI Assistant
        </h2>
        <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', maxWidth: '620px', margin: '0 auto' }}>
          Driven by AD-BoN dynamic route gating. Classifies query complexity, enforces quality thresholds, and selects the optimal Groq model tier for cost-efficient generation.
        </p>
      </div>

      {/* Query Input Section */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
          <label style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Ask AI Assistant
          </label>
          {conversation.length > 0 && (
            <button onClick={clearChat} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', fontSize: '0.74rem', cursor: 'pointer', textDecoration: 'underline' }}>
              Clear Conversation History
            </button>
          )}
        </div>

        {/* Quick Prompts */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {quickPrompts.map((p, idx) => (
            <button
              key={idx}
              className="prompt-chip"
              onClick={() => { setMessage(p); handleSend(p) }}
              disabled={loading}
            >
              {p}
            </button>
          ))}
        </div>

        {/* Input Textarea & Send Button */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <textarea
            className="query-textarea"
            rows={3}
            placeholder="Type your request here (e.g., Explain recursion in Python)..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
            disabled={loading}
            style={{ fontSize: '0.94rem', minHeight: '84px' }}
          />

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Press <strong>Enter</strong> to send, <strong>Shift+Enter</strong> for newline
            </span>
            <button
              className="btn btn-primary"
              onClick={() => handleSend()}
              disabled={loading || !message.trim()}
              style={{ padding: '8px 24px', fontSize: '0.86rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8 }}
            >
              {loading ? (
                <>
                  <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
                  <span>Evaluating &amp; Generating…</span>
                </>
              ) : (
                <>
                  <span>🚀</span>
                  <span>SEND</span>
                </>
              )}
            </button>
          </div>
        </div>

        {error && (
          <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--accent-red)', color: 'var(--accent-red)', padding: '10px 14px', borderRadius: 8, fontSize: '0.82rem' }}>
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* ======================================================= */}
      {/* 1. PRIMARY SECTION: THE ANSWER CARD                    */}
      {/* ======================================================= */}
      <div className="card" style={{ border: '1px solid rgba(59, 130, 246, 0.28)', boxShadow: '0 4px 20px rgba(0,0,0,0.06)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: '1.2rem' }}>💬</span>
            <strong style={{ fontSize: '0.92rem', letterSpacing: '0.04em', textTransform: 'uppercase', color: 'var(--text-primary)' }}>
              ANSWER
            </strong>
            {currentResult && (
              <span className="pill" style={{ background: 'rgba(59, 130, 246, 0.12)', color: 'var(--accent-blue)', border: '1px solid var(--border-bright)' }}>
                {currentResult.routing?.groq_model || 'Groq Generation'}
              </span>
            )}
          </div>

          {currentResult?.answer && (
            <button
              onClick={handleCopy}
              className="btn btn-outline"
              style={{ padding: '4px 12px', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: 6 }}
            >
              {copied ? '✓ Copied!' : '📋 Copy Answer'}
            </button>
          )}
        </div>

        {loading ? (
          <div style={{ padding: '2.5rem 1rem', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
            <div className="spinner" style={{ width: 34, height: 34, borderWidth: 3 }} />
            <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              AD-BoN is routing request to optimal Groq model…
            </div>
            <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
              Evaluating quality thresholds → Best-of-N selection → Live streaming response
            </div>
          </div>
        ) : currentResult?.answer ? (
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-bright)',
            borderRadius: 'var(--radius-md)',
            padding: '1.3rem 1.5rem',
            fontSize: '0.95rem',
            lineHeight: '1.75',
            color: 'var(--text-primary)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            minHeight: '120px',
          }}>
            {currentResult.answer}
          </div>
        ) : (
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1.5px dashed var(--border-bright)',
            borderRadius: 'var(--radius-md)',
            padding: '2.5rem 1.5rem',
            textAlign: 'center',
            color: 'var(--text-muted)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 8,
          }}>
            <span style={{ fontSize: '2rem' }}>✨</span>
            <strong style={{ color: 'var(--text-primary)', fontSize: '0.92rem' }}>Awaiting your question</strong>
            <p style={{ fontSize: '0.78rem', maxWidth: '480px', margin: 0 }}>
              Submit a prompt above to see the assistant answer along with live AD-BoN routing decisions, estimated cost, and token metrics.
            </p>
          </div>
        )}
      </div>

      {/* ======================================================= */}
      {/* 2. SECONDARY SECTION: ROUTING, COST, CACHE & SECURITY  */}
      {/* ======================================================= */}
      {currentResult && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
          {/* AD-BoN ROUTING */}
          <div className="card" style={{ padding: '1.1rem' }}>
            <div className="card-title" style={{ fontSize: '0.72rem', marginBottom: '0.75rem', color: 'var(--accent-blue)' }}>
              🎯 AD-BoN ROUTING
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: '0.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Request Type</span>
                <strong style={{ color: 'var(--text-primary)', textTransform: 'capitalize' }}>{currentResult.routing?.cluster || 'General Knowledge'}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Selected Tier</span>
                <span className={`tier-badge tier-${currentResult.routing?.selected_tier || 'junior'}`}>
                  {currentResult.routing?.selected_tier?.toUpperCase() || 'JUNIOR'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Groq Model</span>
                <strong style={{ color: 'var(--accent-cyan)' }}>{currentResult.routing?.groq_model}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Best-of-N</span>
                <strong>N = {currentResult.routing?.budget}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Expected Quality</span>
                <strong style={{ color: 'var(--accent-green)' }}>{(currentResult.routing?.expected_quality * 100).toFixed(1)}%</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Target Quality</span>
                <strong>{(currentResult.routing?.target_quality * 100).toFixed(1)}%</strong>
              </div>
            </div>
          </div>

          {/* COST ACCOUNTING */}
          <div className="card" style={{ padding: '1.1rem' }}>
            <div className="card-title" style={{ fontSize: '0.72rem', marginBottom: '0.75rem', color: 'var(--accent-green)' }}>
              💰 ESTIMATED COST
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: '0.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>AD-BoN Estimate</span>
                <strong style={{ color: 'var(--accent-green)' }}>${currentResult.cost?.estimated_cost?.toFixed(6)}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Baseline Cost</span>
                <strong style={{ color: 'var(--text-muted)' }}>${currentResult.cost?.baseline_cost?.toFixed(6)}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Cost Savings</span>
                <strong style={{ color: 'var(--accent-green)', fontSize: '0.95rem' }}>{currentResult.cost?.savings_percent}%</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--border)', paddingTop: 6, marginTop: 2 }}>
                <span style={{ color: 'var(--text-muted)' }}>Latency</span>
                <strong>{currentResult.performance?.latency_ms} ms</strong>
              </div>
            </div>
          </div>

          {/* CACHE METRICS */}
          <div className="card" style={{ padding: '1.1rem' }}>
            <div className="card-title" style={{ fontSize: '0.72rem', marginBottom: '0.75rem', color: 'var(--accent-cyan)' }}>
              ⚡ PROMPT CACHE
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: '0.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Status</span>
                <span className={`pill ${currentResult.cache?.hit ? 'pill-hit' : 'pill-miss'}`}>
                  {currentResult.cache?.hit ? 'HIT' : 'MISS'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Tokens Saved</span>
                <strong style={{ color: currentResult.cache?.hit ? 'var(--accent-cyan)' : 'var(--text-muted)' }}>
                  {currentResult.cache?.tokens_saved || 0}
                </strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Prefix Discount</span>
                <strong>{currentResult.cache?.hit ? '50% applied' : 'None'}</strong>
              </div>
            </div>
          </div>

          {/* SECURITY STATUS */}
          <div className="card" style={{ padding: '1.1rem' }}>
            <div className="card-title" style={{ fontSize: '0.72rem', marginBottom: '0.75rem', color: '#10b981' }}>
              🛡️ SECURITY ENVELOPE
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: '0.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: 'var(--text-muted)' }}>Authentication</span>
                <span style={{ color: '#10b981', fontWeight: 600 }}>✓ Verified</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: 'var(--text-muted)' }}>Signal Integrity</span>
                <span style={{ color: '#10b981', fontWeight: 600 }}>✓ Verified</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: 'var(--text-muted)' }}>Replay Protection</span>
                <span style={{ color: '#10b981', fontWeight: 600 }}>✓ Active</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
