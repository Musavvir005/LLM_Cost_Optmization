import React, { useState, useCallback } from 'react'
import { runBenchmark } from '../api'

const TIER_COLOR = {
  junior: 'var(--tier-junior)',
  medium: 'var(--tier-medium)',
  senior: 'var(--tier-senior)',
  cache:  'var(--tier-cache)',
}

const TIER_LABEL = {
  junior: 'Fast Tier',
  medium: 'Balanced Tier',
  senior: 'Frontier Tier',
  cache:  'Instant Cache',
}

function KpiCard({ label, value, sub, color }) {
  return (
    <div className="kpi-card">
      <span className="kpi-label">{label}</span>
      <span className="kpi-value" style={{ color }}>{value}</span>
      {sub && <span className="kpi-sub">{sub}</span>}
    </div>
  )
}

function DeltaCell({ delta }) {
  if (delta === undefined || delta === null) return <span className="delta-neu">—</span>
  const cls = delta > 0 ? 'delta-pos' : delta < -0.05 ? 'delta-neg' : 'delta-neu'
  return <span className={cls}>{delta > 0 ? '+' : ''}{(delta * 100).toFixed(1)}pp</span>
}

export default function BenchmarkPanel() {
  const [data,    setData]    = useState(() => {
    try {
      const saved = localStorage.getItem('adbon_benchmark_data')
      return saved ? JSON.parse(saved) : null
    } catch {
      return null
    }
  })
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  React.useEffect(() => {
    try {
      if (data) {
        localStorage.setItem('adbon_benchmark_data', JSON.stringify(data))
      }
    } catch {
      /* ignore storage quota */
    }
  }, [data])

  const run = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const benchmarkData = await runBenchmark()
      setData(benchmarkData)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  const kpis = data?.kpis

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }} className="fade-in">
      {/* Header */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <h2 style={{ fontSize: '1.3rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
              📊 Before vs. After Benchmark
            </h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: 4 }}>
              12-query fixed suite · Coding · Math · QA · Finance · Compliance
            </p>
          </div>
          <button
            id="run-benchmark-btn"
            className="btn-route"
            onClick={run}
            disabled={loading}
          >
            {loading ? <><div className="spinner" /> Running…</> : '▶ Run Benchmark'}
          </button>
        </div>
      </div>

      {error && (
        <div className="card" style={{ borderColor: 'rgba(248,113,113,0.3)' }}>
          <p style={{ color: 'var(--accent-red)', fontSize: '0.875rem' }}>⚠️ {error}</p>
        </div>
      )}

      {loading && (
        <div className="card">
          <div className="empty-state">
            <div className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
            <p className="empty-text">Running 12-query benchmark suite…<br/>This evaluates all routing tiers and caches.</p>
          </div>
        </div>
      )}

      {data && !loading && (
        <>
          {/* KPI Cards */}
          <div className="kpi-grid">
            <KpiCard
              label="Cost Savings"
              value={`${kpis.overall_savings_pct.toFixed(1)}%`}
              sub={`Saved $${kpis.total_savings_usd.toFixed(6)}`}
              color="var(--accent-green)"
            />
            <KpiCard
              label="Baseline Cost"
              value={`$${kpis.total_baseline_cost.toFixed(6)}`}
              sub="Always-Senior, no cache"
              color="var(--accent-red)"
            />
            <KpiCard
              label="Gateway Cost"
              value={`$${kpis.total_gateway_cost.toFixed(6)}`}
              sub="AD-BoN optimised"
              color="var(--accent-cyan)"
            />
            <KpiCard
              label="Quality Retained"
              value={`${kpis.avg_quality_retained_pct?.toFixed(1) ?? '—'}%`}
              sub="vs. Senior baseline"
              color={kpis.avg_quality_retained_pct > 90 ? 'var(--accent-green)' : 'var(--accent-yellow)'}
            />
            <KpiCard
              label="Sem. Cache Hits"
              value={`${kpis.semantic_hit_rate.toFixed(0)}%`}
              sub={`${kpis.semantic_cache_hits} zero-cost hits`}
              color="var(--tier-cache)"
            />
            <KpiCard
              label="Senior Offloaded"
              value={`${kpis.senior_offload_rate.toFixed(0)}%`}
              sub="Queries routed to cheaper tiers"
              color="var(--tier-junior)"
            />
            <KpiCard
              label="Quality Gate"
              value={kpis.quality_degraded_queries === 0 ? 'PASS ✓' : `${kpis.quality_degraded_queries} flagged`}
              sub="Queries with >5pp quality drop"
              color={kpis.quality_degraded_queries === 0 ? 'var(--accent-green)' : 'var(--accent-orange)'}
            />
            <KpiCard
              label="Prompt Cache Hits"
              value={`${kpis.prompt_hit_rate.toFixed(0)}%`}
              sub={`${kpis.prompt_cache_hits} prefix discounts`}
              color="var(--accent-purple)"
            />
          </div>

          {/* Tier distribution pills */}
          <div className="card">
            <div className="card-title">Tier Distribution</div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              {Object.entries(kpis.tier_counts).map(([tier, count]) => (
                <div key={tier} style={{
                  background: 'var(--bg-secondary)', border: `1px solid ${TIER_COLOR[tier] || 'var(--border)'}22`,
                  borderRadius: 12, padding: '0.75rem 1.25rem', minWidth: 100, textAlign: 'center'
                }}>
                  <div style={{ fontSize: '1.6rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: TIER_COLOR[tier] || 'var(--text-primary)' }}>{count}</div>
                  <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginTop: 2 }}>{TIER_LABEL[tier] || tier}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Per-query table */}
          <div className="card">
            <div className="card-title">Per-Query Results — Cost + Quality Breakdown</div>
            <div className="benchmark-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Category</th>
                    <th>Query</th>
                    <th>Route</th>
                    <th>Cache</th>
                    <th>Base ($)</th>
                    <th>GW ($)</th>
                    <th>Save %</th>
                    <th>Base-Q</th>
                    <th>GW-Q</th>
                    <th>ΔQ</th>
                    <th>Q-Ret%</th>
                  </tr>
                </thead>
                <tbody>
                  {data.results.map(r => (
                    <tr key={r.id}>
                      <td style={{ color: 'var(--text-muted)' }}>{r.id}</td>
                      <td>
                        <span className="pill" style={{
                          background: 'rgba(79,168,255,0.08)',
                          color: 'var(--accent-blue)',
                          border: '1px solid rgba(79,168,255,0.2)',
                        }}>{r.category}</span>
                      </td>
                      <td style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', fontFamily: 'var(--font-sans)', color: 'var(--text-primary)' }}>
                        {r.query.length > 40 ? r.query.slice(0, 40) + '…' : r.query}
                      </td>
                      <td>
                        <span style={{ color: TIER_COLOR[r.model_tier] || 'var(--text-secondary)', fontWeight: 600 }}>
                          {r.route_str}
                        </span>
                      </td>
                      <td>
                        {r.semantic_hit
                          ? <span className="pill pill-hit">⚡ SEM</span>
                          : r.prompt_hit
                          ? <span className="pill pill-hit">📋 PRMT</span>
                          : <span className="pill pill-miss">MISS</span>}
                      </td>
                      <td>${r.baseline_cost.toFixed(6)}</td>
                      <td style={{ color: 'var(--accent-cyan)' }}>${r.gateway_cost.toFixed(6)}</td>
                      <td style={{ color: r.savings_pct > 50 ? 'var(--accent-green)' : 'var(--accent-yellow)', fontWeight: 600 }}>
                        {r.savings_pct.toFixed(1)}%
                      </td>
                      <td style={{ color: 'var(--tier-senior)' }}>{(r.baseline_quality_score * 100).toFixed(0)}%</td>
                      <td style={{ color: r.quality_score > 0.85 ? 'var(--accent-green)' : 'var(--accent-yellow)', fontWeight: 600 }}>
                        {(r.quality_score * 100).toFixed(1)}%
                      </td>
                      <td><DeltaCell delta={r.quality_delta} /></td>
                      <td style={{ color: r.quality_retained_pct > 90 ? 'var(--accent-green)' : 'var(--accent-orange)' }}>
                        {r.quality_retained_pct?.toFixed(1)}%
                        {r.quality_delta < -0.05 && <span style={{ marginLeft: 4 }}>⚠️</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {!data && !loading && !error && (
        <div className="card">
          <div className="empty-state">
            <span className="empty-icon">📊</span>
            <p className="empty-text">Click "Run Benchmark" to execute the full 12-query Before vs. After cost and quality comparison suite.</p>
          </div>
        </div>
      )}
    </div>
  )
}
