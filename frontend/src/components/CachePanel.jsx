import React, { useState, useEffect } from 'react'

function StatRow({ label, value, color }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '0.6rem 0', borderBottom: '1px solid var(--border)',
    }}>
      <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', fontWeight: 600, color: color || 'var(--text-primary)' }}>
        {value}
      </span>
    </div>
  )
}

export default function CachePanel() {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)

  const load = async () => {
    setLoading(true)
    try {
      const r = await fetch('/api/cache/stats')
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      setData(await r.json())
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  if (loading) return (
    <div className="card fade-in">
      <div className="empty-state"><div className="spinner" style={{ width: 32, height: 32, borderWidth: 3 }} /></div>
    </div>
  )

  if (error) return (
    <div className="card fade-in">
      <p style={{ color: 'var(--accent-red)', fontSize: '0.875rem' }}>⚠️ {error}</p>
    </div>
  )

  const sem = data?.semantic_cache || {}
  const pc  = data?.prompt_cache   || {}

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }} className="fade-in">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h2 style={{ fontSize: '1.3rem', fontWeight: 800, letterSpacing: '-0.02em' }}>🗄️ Cache Statistics</h2>
        <button className="btn-route" onClick={load} style={{ padding: '0.5rem 1rem', fontSize: '0.8rem' }}>
          ↻ Refresh
        </button>
      </div>

      <div className="cache-grid">
        {/* Semantic Cache */}
        <div className="card">
          <div className="card-title">⚡ Semantic Memory Cache</div>
          <StatRow label="Total Entries" value={sem.total_entries ?? '—'} color="var(--accent-cyan)" />
          <StatRow label="Similarity Threshold" value={sem.similarity_threshold ? `${(sem.similarity_threshold * 100).toFixed(0)}%` : '—'} color="var(--accent-blue)" />
          <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'var(--bg-secondary)', borderRadius: 8 }}>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Stores vector embeddings of past queries. Any new query with cosine similarity ≥ {sem.similarity_threshold ?? 0.92} is returned <strong style={{ color: 'var(--accent-green)' }}>instantly at $0.00</strong>.
            </p>
          </div>
        </div>

        {/* Prompt Cache */}
        <div className="card">
          <div className="card-title">📋 Prompt Prefix Cache</div>
          <StatRow label="Total Hashes Stored" value={pc.total_hashes ?? '—'} color="var(--accent-purple)" />
          <StatRow label="Total Hits" value={pc.total_hits ?? '—'} color="var(--accent-green)" />
          <StatRow label="Tokens Cached" value={pc.total_tokens_cached ?? '—'} color="var(--accent-cyan)" />
          <StatRow label="Tokens Discounted" value={pc.total_tokens_discounted ?? '—'} color="var(--accent-yellow)" />
          <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'var(--bg-secondary)', borderRadius: 8 }}>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Hashes repeated <code style={{ color: 'var(--accent-blue)' }}>system_prompt</code> + <code style={{ color: 'var(--accent-blue)' }}>context_block</code> prefixes and applies a <strong style={{ color: 'var(--accent-green)' }}>50% input token discount</strong> on cache hits.
            </p>
          </div>
        </div>
      </div>

      {/* How it works */}
      <div className="card">
        <div className="card-title">How the Cache Pipeline Works</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem' }}>
          {[
            { step: '1', title: 'Incoming Query', desc: 'Query hits the gateway. Embedding is computed.', color: 'var(--accent-blue)' },
            { step: '2', title: 'Semantic Scan', desc: 'Cosine similarity checked vs. all cached embeddings.', color: 'var(--accent-cyan)' },
            { step: '3', title: 'Cache Hit?', desc: 'If ≥92% match → return cached answer instantly. Cost = $0.', color: 'var(--tier-cache)' },
            { step: '4', title: 'Prompt Cache', desc: 'System prompt prefix hash checked. 50% discount applied if hit.', color: 'var(--accent-purple)' },
            { step: '5', title: 'Route & Execute', desc: 'Classifier picks tier, optimizer sets budget N, model executes.', color: 'var(--tier-junior)' },
            { step: '6', title: 'Store Result', desc: 'Response + embedding committed to semantic cache for future hits.', color: 'var(--accent-green)' },
          ].map(s => (
            <div key={s.step} style={{
              background: 'var(--bg-secondary)', border: '1px solid var(--border)',
              borderRadius: 10, padding: '0.875rem', display: 'flex', flexDirection: 'column', gap: 6
            }}>
              <div style={{ width: 24, height: 24, borderRadius: 6, background: s.color + '22', border: `1px solid ${s.color}44`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ fontSize: '0.72rem', fontWeight: 700, color: s.color }}>{s.step}</span>
              </div>
              <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>{s.title}</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{s.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
