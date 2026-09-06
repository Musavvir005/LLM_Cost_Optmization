import React, { useState } from 'react'

// Lightweight markdown-to-HTML renderer (no extra dependencies)
function renderMarkdown(text) {
  if (!text) return ''
  let html = text
    // Escape HTML special chars first
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    // Fenced code blocks ```lang\n...\n```
    .replace(/```([\w]*)\n?([\s\S]*?)```/g, (_, lang, code) =>
      `<pre class="md-pre"><code class="md-code${lang ? ' lang-' + lang : ''}">${code.trim()}</code></pre>`
    )
    // Inline code `...`
    .replace(/`([^`]+)`/g, '<code class="md-inline-code">$1</code>')
    // Headers ###, ##, #
    .replace(/^### (.+)$/gm, '<h3 class="md-h3">$1</h3>')
    .replace(/^## (.+)$/gm, '<h2 class="md-h2">$1</h2>')
    .replace(/^# (.+)$/gm, '<h1 class="md-h1">$1</h1>')
    // Bold **text** or __text__
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_]+)__/g, '<strong>$1</strong>')
    // Italic *text* or _text_
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/_([^_]+)_/g, '<em>$1</em>')
    // Unordered list items - item or * item
    .replace(/^[\*\-] (.+)$/gm, '<li class="md-li">$1</li>')
    // Numbered list items 1. item
    .replace(/^\d+\. (.+)$/gm, '<li class="md-li md-ol-li">$1</li>')
    // Horizontal rule ---
    .replace(/^---+$/gm, '<hr class="md-hr" />')
    // Wrap consecutive <li> in <ul>
    .replace(/(<li[^>]*>.*?<\/li>)(\n<li[^>]*>.*?<\/li>)*/gs, m =>
      m.includes('md-ol-li') ? `<ol class="md-ol">${m}</ol>` : `<ul class="md-ul">${m}</ul>`
    )
    // Paragraphs: double newline → <p>
    .replace(/\n{2,}/g, '</p><p class="md-p">')
    // Single newlines → <br> (inside paragraphs)
    .replace(/\n/g, '<br />')
  return `<p class="md-p">${html}</p>`
}

function TierBadge({ tier, model }) {
  const map = {
    junior: { label: 'Fast Tier (Lightweight)',    cls: 'tier-junior',  icon: '🟢', color: 'var(--tier-junior)' },
    medium: { label: 'Balanced Tier (Standard)',   cls: 'tier-medium',  icon: '🔵', color: 'var(--tier-medium)' },
    senior: { label: 'Frontier Tier (Flagship)',    cls: 'tier-senior',  icon: '🟣', color: 'var(--tier-senior)' },
    cache:  { label: 'Instant Cache Memory',        cls: 'tier-cache',   icon: '⚡', color: 'var(--tier-cache)' },
  }
  const t = map[tier] || map.medium
  return (
    <span className={`tier-badge ${t.cls}`} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <span>{t.icon}</span>
      <span>{t.label}</span>
    </span>
  )
}

function QualityBar({ label, value, max = 1, color }) {
  const pct = Math.min((value / max) * 100, 100)
  return (
    <div className="quality-bar-wrap">
      <div className="quality-row">
        <span className="quality-label">{label}</span>
        <span className="quality-pct" style={{ color }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="quality-bar-track">
        <div className="quality-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  )
}

function MetricCard({ label, value, sub, color }) {
  return (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <span className="metric-value" style={{ color }}>{value}</span>
      {sub && <span className="metric-sub">{sub}</span>}
    </div>
  )
}

export default function ResultPanel({ result, loading }) {
  const [copied, setCopied] = useState(false)
  const [paramsVisible, setParamsVisible] = useState(false)
  const [paramsHovered, setParamsHovered] = useState(false)

  const isParamsActive = paramsVisible || paramsHovered

  const copyToClipboard = () => {
    if (!result?.final_response) return
    navigator.clipboard.writeText(result.final_response)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (loading) {
    return (
      <div className="card fade-in" style={{ border: '1px solid rgba(79, 168, 255, 0.3)' }}>
        <div className="empty-state" style={{ padding: '2rem 1rem' }}>
          <div className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
          <p className="empty-text" style={{ marginTop: 12 }}>
            <strong>Routing query through AD-BoN Gateway…</strong><br/>
            Evaluating cluster embeddings → Cost-quality optimization → Live model execution
          </p>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="card fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', border: '1px solid var(--border)' }}>
        {/* Top Status Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8 }}>
            <span className="pill" style={{ background: 'rgba(59, 130, 246, 0.1)', color: 'var(--accent-blue)', border: '1px solid var(--border-bright)' }}>
              ⚡ AD-BoN Output Window
            </span>
            <span className="pill" style={{ background: 'var(--bg-secondary)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
              Status: Awaiting Query
            </span>
          </div>
          <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
            Output and telemetry update dynamically
          </span>
        </div>

        {/* Security Readiness */}
        <div style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          padding: '8px 14px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 12
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Authentication:</span>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--accent-green)' }}>✓ Ready (Bearer)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Routing Signal:</span>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--accent-green)' }}>✓ Ready (HMAC-SHA256)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Replay Protection:</span>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--accent-green)' }}>✓ Active (Nonce/Timestamp)</span>
          </div>
        </div>

        {/* Output Box */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <div className="card-title" style={{ margin: 0, fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>💬</span> Model Generation / Answer
            </div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Awaiting execution...</span>
          </div>

          <div style={{
            background: 'var(--bg-secondary)',
            border: '1.5px dashed var(--border-bright)',
            borderRadius: 'var(--radius-md)',
            padding: '2.5rem 1.5rem',
            textAlign: 'center',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            color: 'var(--text-muted)',
            minHeight: '130px',
          }}>
            <div style={{ fontSize: '1.8rem', opacity: 0.8 }}>⚡</div>
            <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Output will appear here
            </div>
            <div style={{ fontSize: '0.76rem', maxWidth: '540px', lineHeight: 1.5 }}>
              Enter your prompt above, optionally upload a media file (up to 10 MB), or click one of the quick presets and hit <strong>Route &amp; Execute Query</strong>.
            </div>
          </div>
        </div>

        {/* Metrics Placeholder Grid (Cover / Hover to Reveal) */}
        <div
          className="metrics-reveal-wrapper"
          onMouseEnter={() => setParamsHovered(true)}
          onMouseLeave={() => setParamsHovered(false)}
        >
          <div
            className="metrics-reveal-header"
            onClick={() => setParamsVisible(v => !v)}
            title="Hover over text to reveal parameters, or click to lock open"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              cursor: 'pointer',
              padding: '8px 12px',
              borderRadius: '8px',
              background: isParamsActive ? 'rgba(79, 168, 255, 0.08)' : 'transparent',
              border: `1px solid ${isParamsActive ? 'var(--border-bright)' : 'var(--border)'}`,
              transition: 'all 0.2s ease',
              userSelect: 'none',
              marginBottom: isParamsActive ? '0.75rem' : 0
            }}
          >
            <div className="card-title" style={{ margin: 0, fontSize: '0.78rem', display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>📊</span> Execution Parameters &amp; Performance Metrics
            </div>
            <span className="pill" style={{
              fontSize: '0.7rem',
              background: isParamsActive ? 'rgba(79, 168, 255, 0.15)' : 'var(--bg-secondary)',
              color: isParamsActive ? 'var(--accent-blue)' : 'var(--text-muted)',
              border: '1px solid var(--border)'
            }}>
              {paramsVisible ? '📌 Pinned Open (Click to Hide)' : paramsHovered ? '👁️ Visible (Covering Text)' : 'Hover text to reveal ▾'}
            </span>
          </div>

          {isParamsActive && (
            <div className="result-grid fade-in">
              <MetricCard label="Selected Model" value="—" sub="Auto-determined by AD-BoN" color="var(--text-muted)" />
              <MetricCard label="Cost Savings" value="—%" sub="Awaiting routing" color="var(--text-muted)" />
              <MetricCard label="Gateway Cost" value="—" sub="Optimized across tiers" color="var(--text-muted)" />
              <MetricCard label="Quality Score" value="—" sub="Threshold benchmark" color="var(--text-muted)" />
              <MetricCard label="Quality Delta" value="—" sub="vs Always-Frontier" color="var(--text-muted)" />
              <MetricCard label="Latency" value="—" sub="Fast retrieval" color="var(--text-muted)" />
            </div>
          )}
        </div>
      </div>
    )
  }

  if (result.error) {
    return (
      <div className="card fade-in" style={{ borderColor: 'rgba(248, 113, 113, 0.4)' }}>
        <div className="empty-state">
          <span className="empty-icon" style={{ fontSize: '2rem' }}>⚠️</span>
          <p className="empty-text" style={{ color: 'var(--accent-red)', marginTop: 8 }}>{result.error}</p>
        </div>
      </div>
    )
  }

  const savingsColor = result.cost_savings_pct > 50 ? 'var(--accent-green)'
    : result.cost_savings_pct > 20 ? 'var(--accent-yellow)' : 'var(--accent-red)'

  const deltaColor = result.quality_delta >= 0 ? 'var(--accent-green)'
    : result.quality_delta > -0.05 ? 'var(--accent-yellow)' : 'var(--accent-red)'

  const tierFriendlyName = {
    junior: 'Fast Tier (Lightweight)',
    medium: 'Balanced Tier (Standard)',
    senior: 'Frontier Tier (Flagship)',
    cache:  'Instant Cache Hit',
  }[result.routing_tier] || result.routing_tier

  return (
    <div className="card fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', border: '1px solid rgba(79, 168, 255, 0.25)' }}>
      {/* Top Status & Badges */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8 }}>
          <TierBadge tier={result.routing_tier} model={result.selected_model} />
          <span className="pill" style={{ background: 'rgba(255,255,255,0.06)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}>
            🤖 Model: <strong style={{ color: 'var(--accent-cyan)', marginLeft: 4 }}>{result.selected_model}</strong>
          </span>
          {result.is_semantic_cache_hit && (
            <span className="pill pill-hit">⚡ Semantic Cache Hit ({result.retrieval_latency_ms.toFixed(1)}ms)</span>
          )}
          {result.is_prompt_cache_hit && (
            <span className="pill pill-hit">📋 Prompt Cache Hit (50% Token Discount)</span>
          )}
          {!result.is_semantic_cache_hit && !result.is_prompt_cache_hit && (
            <span className="pill pill-miss">Live Generation</span>
          )}
          <span className={`self-fix-badge ${result.self_fix_triggered ? 'self-fix-triggered' : 'self-fix-none'}`}>
            {result.self_fix_triggered ? `🔧 Self-Fix: ${result.self_fix_status}` : '✓ High Confidence'}
          </span>
        </div>

        <button
          onClick={copyToClipboard}
          style={{
            background: copied ? 'rgba(52, 211, 153, 0.15)' : 'rgba(79, 168, 255, 0.12)',
            border: `1px solid ${copied ? 'var(--accent-green)' : 'rgba(79, 168, 255, 0.3)'}`,
            borderRadius: 8, padding: '5px 12px', fontSize: '0.78rem',
            color: copied ? 'var(--accent-green)' : 'var(--accent-blue)',
            cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6,
            transition: 'all 0.15s ease'
          }}
        >
          {copied ? '✓ Copied Output!' : '📋 Copy Output'}
        </button>
      </div>

      {/* Security Verification Status */}
      <div style={{
        background: 'rgba(16, 185, 129, 0.08)',
        border: '1px solid rgba(52, 211, 153, 0.25)',
        borderRadius: 8,
        padding: '8px 14px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: 12
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Authentication:</span>
          <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#34d399' }}>✓ Request authenticated</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Routing Signal:</span>
          <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#34d399' }}>✓ Integrity verified</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Replay Protection:</span>
          <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#34d399' }}>✓ Valid timestamp/nonce</span>
        </div>
      </div>

      {/* Multimodal Bypass Triggered Banner */}
      {result.multimodal_bypass_triggered && (
        <div style={{
          background: 'linear-gradient(135deg, rgba(167, 139, 250, 0.15) 0%, rgba(56, 189, 248, 0.08) 100%)',
          border: '1px solid rgba(167, 139, 250, 0.4)',
          borderRadius: 8,
          padding: '12px 16px',
          display: 'flex',
          flexDirection: 'column',
          gap: 6
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: '1.2rem' }}>⚡</span>
              <strong style={{ color: '#c4b5fd', fontSize: '0.88rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                [MULTIMODAL BYPASS TRIGGERED]
              </strong>
            </div>
            <span style={{
              background: 'rgba(167, 139, 250, 0.25)', color: '#e9d5ff',
              padding: '3px 9px', borderRadius: 6, fontSize: '0.72rem', fontWeight: 600
            }}>
              Force-Routed to Frontier Tier (Gemini 1.5 Pro)
            </span>
          </div>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            {result.bypass_reason}
          </div>
          {result.file_metadata && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>
              <span>📁 File: <strong style={{ color: 'var(--text-primary)' }}>{result.file_metadata.filename}</strong></span>
              <span>🏷️ Category: <strong style={{ color: 'var(--accent-cyan)' }}>{result.file_metadata.category?.toUpperCase()}</strong></span>
              <span>💾 Size: <strong>{result.file_metadata.size_kb} KB</strong></span>
              {result.multimodal_surcharge > 0 && (
                <span>💰 Fee: <strong style={{ color: 'var(--accent-yellow)' }}>${result.multimodal_surcharge.toFixed(4)}</strong></span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Small Document Programmatic Ingestion Banner */}
      {!result.multimodal_bypass_triggered && result.file_metadata && (
        <div style={{
          background: 'rgba(52, 211, 153, 0.08)',
          border: '1px solid rgba(52, 211, 153, 0.3)',
          borderRadius: 8,
          padding: '10px 14px',
          fontSize: '0.8rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 8
        }}>
          <div>
            📄 <strong style={{ color: '#6ee7b7' }}>Document Ingested:</strong> {result.file_metadata.filename} ({result.file_metadata.size_kb} KB)
            <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>— Programmatically extracted &amp; routed through sequence classifier</span>
          </div>
          <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>Standard Gating Applied</span>
        </div>
      )}

      {/* 1. EXACT MODEL OUTPUT (Prominently displayed right below input section) */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <label className="input-label" style={{ color: 'var(--accent-blue)', fontSize: '0.85rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span>💬</span> Exact Model Output
          </label>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            Delivered via {tierFriendlyName}
          </span>
        </div>

        <div style={{
          background: 'var(--bg-input-box, var(--bg-secondary))',
          border: '1px solid var(--border-bright)',
          borderRadius: 'var(--radius-md)',
          padding: '1.1rem 1.3rem',
          fontSize: '0.92rem',
          lineHeight: '1.75',
          color: 'var(--text-primary)',
          fontFamily: 'var(--font-sans)',
          minHeight: '100px',
          maxHeight: '520px',
          overflowY: 'auto',
          wordBreak: 'break-word',
          boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.04)',
        }}
          className="md-output"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(result.final_response) }}
        />
      </div>

      {/* 2. ROUTING INTELLIGENCE & PARAMETERS GRID (COVER / HOVER TO REVEAL) */}
      <div
        className="metrics-reveal-wrapper"
        onMouseEnter={() => setParamsHovered(true)}
        onMouseLeave={() => setParamsHovered(false)}
      >
        <div
          className="metrics-reveal-header"
          onClick={() => setParamsVisible(v => !v)}
          title="Hover over text to reveal parameters, or click to lock open"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            cursor: 'pointer',
            padding: '10px 14px',
            borderRadius: '8px',
            background: isParamsActive ? 'rgba(79, 168, 255, 0.1)' : 'var(--bg-secondary)',
            border: `1px solid ${isParamsActive ? 'var(--border-bright)' : 'var(--border)'}`,
            transition: 'all 0.2s ease',
            userSelect: 'none',
            marginBottom: isParamsActive ? '0.85rem' : 0
          }}
        >
          <div className="card-title" style={{ margin: 0, fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>📊</span> Execution Parameters &amp; Performance Metrics
          </div>
          <span className="pill" style={{
            fontSize: '0.72rem',
            background: isParamsActive ? 'rgba(79, 168, 255, 0.18)' : 'var(--bg-card)',
            color: isParamsActive ? 'var(--accent-blue)' : 'var(--text-muted)',
            border: '1px solid var(--border)',
            fontWeight: 600
          }}>
            {paramsVisible ? '📌 Pinned Open (Click to Hide)' : paramsHovered ? '👁️ Visible (Covering Text)' : 'Hover text to view ▾'}
          </span>
        </div>

        {isParamsActive && (
          <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div className="result-grid">
              <MetricCard
                label="Selected Model"
                value={result.selected_model || result.routing_tier}
                sub={`Tier: ${tierFriendlyName}`}
                color="var(--accent-cyan)"
              />
              <MetricCard
                label="Cost Savings"
                value={`${result.cost_savings_pct.toFixed(1)}%`}
                sub={`Saved $${(result.senior_baseline_cost_usd - result.gateway_cost_usd).toFixed(6)}`}
                color={savingsColor}
              />
              <MetricCard
                label="Gateway Cost"
                value={`$${result.gateway_cost_usd.toFixed(6)}`}
                sub={`vs $${result.senior_baseline_cost_usd.toFixed(6)} baseline`}
                color="var(--text-primary)"
              />
              <MetricCard
                label="Quality Score"
                value={`${(result.final_quality_score * 100).toFixed(1)}%`}
                sub={`Retained ${result.quality_retained_pct.toFixed(1)}% of Frontier`}
                color={result.final_quality_score > 0.85 ? 'var(--accent-green)' : 'var(--accent-yellow)'}
              />
              <MetricCard
                label="Quality Delta"
                value={`${result.quality_delta >= 0 ? '+' : ''}${(result.quality_delta * 100).toFixed(1)}pp`}
                sub="vs Always-Frontier baseline"
                color={deltaColor}
              />
              <MetricCard
                label="Latency"
                value={`${result.retrieval_latency_ms < 10 ? result.retrieval_latency_ms.toFixed(1) : result.retrieval_latency_ms.toFixed(0)}ms`}
                sub={result.is_semantic_cache_hit ? 'Instant memory cache' : 'End-to-end routing & inference'}
                color="var(--accent-purple)"
              />
            </div>

            {/* Quality Retention Analysis */}
            <div className="card" style={{ background: 'var(--bg-secondary)', padding: '1rem', gap: 10, display: 'flex', flexDirection: 'column', border: '1px solid var(--border)' }}>
              <div className="card-title" style={{ marginBottom: 0, fontSize: '0.75rem' }}>Quality Retention Benchmark Analysis</div>
              <QualityBar
                label={`Frontier Baseline (${(result.baseline_quality_score * 100).toFixed(0)}%)`}
                value={result.baseline_quality_score}
                color="var(--tier-senior)"
              />
              <QualityBar
                label={`AD-BoN Gateway (${(result.final_quality_score * 100).toFixed(1)}%)`}
                value={result.final_quality_score}
                color={result.final_quality_score > 0.85 ? 'var(--accent-green)' : 'var(--accent-yellow)'}
              />
              <QualityBar
                label={`Quality Retained (${result.quality_retained_pct.toFixed(1)}%)`}
                value={result.quality_retained_pct / 100}
                color="var(--accent-cyan)"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
