import React, { useState, useEffect } from 'react'

export default function HealthPanel() {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)

  const load = async () => {
    setLoading(true)
    try {
      const r = await fetch('/api/health')
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      setData(await r.json())
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  const tiers = [
    { key: 'junior', label: 'Fast Tier',     model: 'Ollama (Local phi3 / gemma3)',       color: 'var(--tier-junior)',  price: '$0.00/1M', desc: 'Free local compute, zero network API cost' },
    { key: 'medium', label: 'Balanced Tier', model: 'Groq (qwen-27b / compound)',         color: 'var(--tier-medium)',  price: '$0.59/1M', desc: 'Fast cloud LPU engine, high throughput reasoning' },
    { key: 'senior', label: 'Frontier Tier', model: 'Google Gemini Pro (gemini-2.5-pro)', color: 'var(--tier-senior)',  price: '$1.25/1M', desc: 'Flagship reasoning for complex coding, math, and compliance' },
  ]

  const features = [
    { icon: '🎯', title: 'Request Classifier',     desc: 'Semantic keyword router maps queries to clusters (coding, math, compliance…). Selects cheapest tier that can meet quality threshold.' },
    { icon: '🔀', title: 'Multi-Tier Routing',      desc: 'Junior → Medium → Senior escalation. Confidence Guard (τ=0.75) triggers Self-Fix loop before expensive Senior fallback.' },
    { icon: '📋', title: 'Prompt Cache Layer',      desc: 'SHA-256 hashes repeated system_prompt + context prefixes. Applies 50% input token discount on cache hits.' },
    { icon: '⚡', title: 'Semantic Memory Cache',   desc: 'Vector cosine scan across all past queries. ≥92% match → instant $0.00 answer, zero compute.' },
    { icon: '🔧', title: 'Self-Fix Critic Loop',    desc: 'Low-confidence drafts are critiqued and refined in-tier before escalating. Avoids unnecessary Senior calls.' },
    { icon: '📊', title: 'Quality Measurement',     desc: 'Every response scored. quality_delta and quality_retained_pct prove cost savings ≠ quality degradation.' },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }} className="fade-in">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h2 style={{ fontSize: '1.3rem', fontWeight: 800, letterSpacing: '-0.02em' }}>💚 System Health</h2>
        <button className="btn-route" onClick={load} style={{ padding: '0.5rem 1rem', fontSize: '0.8rem' }}>
          ↻ Refresh
        </button>
      </div>

      {/* API Status */}
      <div className="card">
        <div className="card-title">API Status</div>
        {error ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 12, height: 12, borderRadius: '50%', background: 'var(--accent-red)', boxShadow: '0 0 10px var(--accent-red)' }} />
            <span style={{ color: 'var(--accent-red)', fontWeight: 600 }}>Offline — {error}</span>
          </div>
        ) : loading ? (
          <div className="spinner" />
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{ width: 12, height: 12, borderRadius: '50%', background: 'var(--accent-green)', boxShadow: '0 0 10px var(--accent-green)', animation: 'pulse 2s infinite' }} />
              <div>
                <div style={{ fontWeight: 700, fontSize: '0.9rem' }}>API Online</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>http://127.0.0.1:8000</div>
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Version</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--accent-cyan)' }}>{data?.version ?? '—'}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Semantic Cache</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--accent-yellow)' }}>{data?.semantic_cache_entries ?? '—'} entries</div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Confidence Threshold</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--accent-blue)' }}>{data?.confidence_threshold ?? '—'}</div>
            </div>
          </div>
        )}
      </div>

      {/* Model Tiers */}
      <div className="card">
        <div className="card-title">Model Tier Architecture</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {tiers.map((t, i) => (
            <div key={t.key} style={{
              background: 'var(--bg-secondary)', border: `1px solid ${t.color}22`,
              borderLeft: `3px solid ${t.color}`, borderRadius: 10, padding: '0.875rem 1rem',
              display: 'grid', gridTemplateColumns: '130px 1fr auto', gap: '1rem', alignItems: 'center'
            }}>
              <div>
                <span className={`tier-badge tier-${t.key}`}>{t.label}</span>
              </div>
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--text-primary)' }}>{t.model}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>{t.desc}</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '0.85rem', color: t.color }}>{t.price}</div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 2 }}>input tokens</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Feature grid */}
      <div className="card">
        <div className="card-title">System Capabilities</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '0.875rem' }}>
          {features.map(f => (
            <div key={f.title} style={{
              background: 'var(--bg-secondary)', border: '1px solid var(--border)',
              borderRadius: 10, padding: '0.875rem', display: 'flex', flexDirection: 'column', gap: 8
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: '1.1rem' }}>{f.icon}</span>
                <span style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--text-primary)' }}>{f.title}</span>
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
