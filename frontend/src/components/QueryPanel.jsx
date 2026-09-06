import React, { useState } from 'react'

const EXAMPLE_QUERIES = [
  "Implement a balanced red-black tree with insertion and deletion in Python",
  "Explain Newton's second law of motion and give 3 real-world examples",
  "Write a summary of the quarterly financial report showing 23% revenue growth",
  "Review the vendor data processing agreement for GDPR Article 28 compliance",
  "What is the capital of France?",
]

export default function QueryPanel({ onSubmit, loading }) {
  const [query, setQuery]     = useState('')
  const [quality, setQuality] = useState(0.76)
  const [sysPrompt, setSys]   = useState('')
  const [advanced, setAdv]    = useState(false)
  const [attachedFile, setAttachedFile] = useState(null)
  const [fileError, setFileError]       = useState(null)
  const [isDragging, setIsDragging]     = useState(false)

  const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 // 10 MB limit

  const processFile = (f) => {
    if (!f) return
    setFileError(null)

    // Enforce 10 MB file size limit
    if (f.size > MAX_FILE_SIZE_BYTES) {
      const sizeMb = (f.size / (1024 * 1024)).toFixed(1)
      setFileError(`File "${f.name}" (${sizeMb} MB) exceeds the 10 MB size limit. Please upload a file under 10 MB.`)
      setAttachedFile(null)
      return
    }

    const reader = new FileReader()
    reader.onload = () => {
      setAttachedFile({
        name: f.name,
        size: f.size > 1024 * 1024 ? (f.size / (1024 * 1024)).toFixed(1) + ' MB' : (f.size / 1024).toFixed(1) + ' KB',
        type: f.type || f.name.split('.').pop(),
        base64: reader.result,
      })
    }
    reader.readAsDataURL(f)
  }

  const handleFileChange = (e) => {
    processFile(e.target.files?.[0])
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0])
    }
  }

  const selectSampleFile = (type) => {
    setFileError(null)
    if (type === 'image') {
      setQuery("Analyze this market chart and identify key support/resistance levels.")
      setAttachedFile({ name: 'sample_chart.png', path: 'sample_chart.png', size: '1.6 KB', type: 'image/png' })
    } else if (type === 'pdf') {
      setQuery("Perform a financial risk audit on this quarterly filing.")
      setAttachedFile({ name: 'financial_statement.pdf', path: 'financial_statement.pdf', size: '134.8 KB', type: 'application/pdf' })
    } else if (type === 'csv') {
      setQuery("Summarize regional quarterly sales trends from this dataset.")
      setAttachedFile({ name: 'small_data.csv', path: 'small_data.csv', size: '0.2 KB', type: 'text/csv' })
    }
  }

  const handle = () => {
    if (!query.trim() && !attachedFile) return
    const finalQuery = query.trim() || (attachedFile ? `Please analyze and summarize the attached file: ${attachedFile.name}` : '')
    onSubmit({
      query: finalQuery,
      target_quality: quality,
      system_prompt: sysPrompt || undefined,
      file_path: attachedFile?.path,
      file_name: attachedFile?.name,
      file_base64: attachedFile?.base64,
    })
  }

  return (
    <div className="card query-panel">
      <div className="query-header">
        <div>
          <h2>Route a Query</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: 2 }}>
            Adaptive Dynamic Best-of-N Gateway — routes to the most cost-effective tier meeting your quality bar
          </p>
        </div>
      </div>

      {/* Quick text examples */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 6, alignItems: 'center' }}>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Try Prompt:</span>
        {EXAMPLE_QUERIES.slice(0, 3).map((q, i) => (
          <button
            key={i}
            onClick={() => setQuery(q)}
            className="prompt-chip"
          >
            {q.length > 38 ? q.slice(0, 38) + '…' : q}
          </button>
        ))}
      </div>

      {/* ================================================================ */}
      {/* 1. DEDICATED FILE & MEDIA UPLOAD SECTION (BEFORE TEXT INPUT)     */}
      {/* ================================================================ */}
      <div className="upload-card-wrapper">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: '1rem' }}>📁</span>
            <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Attach File / Media
            </span>
            <span style={{
              background: 'rgba(255,255,255,0.08)',
              border: '1px solid var(--border)',
              padding: '1px 7px',
              borderRadius: 10,
              fontSize: '0.68rem',
              color: 'var(--text-muted)',
              fontWeight: 500
            }}>
              Optional
            </span>
            <span style={{
              background: 'rgba(245, 158, 11, 0.12)',
              border: '1px solid rgba(245, 158, 11, 0.35)',
              padding: '1px 7px',
              borderRadius: 10,
              fontSize: '0.68rem',
              color: 'var(--accent-yellow)',
              fontWeight: 600
            }}>
              Max Size: 10 MB
            </span>
          </div>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            Images &bull; PDFs &bull; CSVs &bull; Audio &bull; Video &bull; <strong>Up to 10 MB</strong>
          </span>
        </div>

        {/* Dropzone / Upload Box */}
        {!attachedFile ? (
          <div
            className={`dropzone-box ${isDragging ? 'dragging' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-upload-input').click()}
          >
            <input
              id="file-upload-input"
              type="file"
              onChange={handleFileChange}
              accept="image/*,audio/*,video/*,.pdf,.csv,.txt"
              style={{ display: 'none' }}
            />
            <div className="dropzone-cloud-icon-wrapper">
              <svg className="dropzone-cloud-icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96z"/>
              </svg>
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-primary)', fontWeight: 600 }}>
              Click to browse or drag &amp; drop a file here (up to 10 MB)
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              Supports Images (.png, .jpg), Documents (.pdf, .csv, .txt), Audio (.mp3, .wav), Video (.mp4)
            </div>

            {/* Quick 1-click test file buttons */}
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }} onClick={e => e.stopPropagation()}>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', alignSelf: 'center' }}>Quick Presets:</span>
              <button
                type="button"
                onClick={() => selectSampleFile('image')}
                style={{
                  background: 'var(--preset-image-bg)', border: '1px solid var(--preset-image-border)',
                  borderRadius: 14, padding: '3px 9px', fontSize: '0.7rem', color: 'var(--preset-image-color)', cursor: 'pointer',
                  fontWeight: 500
                }}
              >
                🖼️ Sample Chart (.png)
              </button>
              <button
                type="button"
                onClick={() => selectSampleFile('pdf')}
                style={{
                  background: 'var(--preset-pdf-bg)', border: '1px solid var(--preset-pdf-border)',
                  borderRadius: 14, padding: '3px 9px', fontSize: '0.7rem', color: 'var(--preset-pdf-color)', cursor: 'pointer',
                  fontWeight: 500
                }}
              >
                📑 Financial Statement (.pdf &gt;100KB)
              </button>
              <button
                type="button"
                onClick={() => selectSampleFile('csv')}
                style={{
                  background: 'var(--preset-csv-bg)', border: '1px solid var(--preset-csv-border)',
                  borderRadius: 14, padding: '3px 9px', fontSize: '0.7rem', color: 'var(--preset-csv-color)', cursor: 'pointer',
                  fontWeight: 500
                }}
              >
                📊 Sales Data (.csv &lt;100KB)
              </button>
            </div>
          </div>
        ) : (
          /* Active Attached File Display Card */
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(167, 139, 250, 0.08) 100%)',
            border: '1px solid var(--border-bright)',
            borderRadius: 8,
            padding: '10px 14px',
            flexWrap: 'wrap',
            gap: 10
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: '1.4rem' }}>
                {attachedFile.name.endsWith('.png') || attachedFile.name.endsWith('.jpg') ? '🖼️' :
                 attachedFile.name.endsWith('.pdf') ? '📑' :
                 attachedFile.name.endsWith('.csv') ? '📊' : '📁'}
              </span>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <strong style={{ color: 'var(--accent-cyan)', fontSize: '0.85rem' }}>{attachedFile.name}</strong>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>({attachedFile.size || attachedFile.type})</span>
                </div>
                <div style={{ fontSize: '0.72rem', color: '#6ee7b7', marginTop: 2 }}>
                  {attachedFile.name.endsWith('.png') || attachedFile.name.endsWith('.jpg') ?
                    '⚡ Image detected: Auto-routes directly to Frontier Tier (Gemini 1.5 Pro)' :
                   attachedFile.name.endsWith('.pdf') ?
                    '⚡ PDF detected: Evaluates size (>100KB triggers Frontier Tier long-context)' :
                    '⚡ Small tabular data: Content will be extracted and routed via sequence classifier'}
                </div>
              </div>
            </div>
            <button
              onClick={() => setAttachedFile(null)}
              style={{
                background: 'rgba(248, 113, 113, 0.15)',
                border: '1px solid rgba(248, 113, 113, 0.35)',
                color: '#fca5a5',
                borderRadius: 6,
                padding: '4px 10px',
                fontSize: '0.75rem',
                cursor: 'pointer',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: 4
              }}
            >
              ✕ Remove File
            </button>
          </div>
        )}

        {/* File Size or Format Error Banner */}
        {fileError && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '8px 12px',
            borderRadius: 6,
            background: 'rgba(239, 68, 68, 0.12)',
            border: '1px solid rgba(239, 68, 68, 0.35)',
            color: '#fca5a5',
            fontSize: '0.78rem'
          }}>
            <span>⚠️</span>
            <span style={{ flex: 1, fontWeight: 500 }}>{fileError}</span>
            <button
              type="button"
              onClick={() => setFileError(null)}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#fca5a5',
                cursor: 'pointer',
                fontSize: '0.9rem',
                padding: '0 4px',
                lineHeight: 1
              }}
            >
              ✕
            </button>
          </div>
        )}
      </div>

      {/* ================================================================ */}
      {/* 2. TEXT INPUT SECTION                                            */}
      {/* ================================================================ */}
      <div className="input-group" style={{ marginTop: 4 }}>
        <label className="input-label">Your Query / Prompt Instructions</label>
        <textarea
          id="query-input"
          className="query-textarea"
          placeholder={attachedFile ? `Enter question or instructions for ${attachedFile.name}…` : "Ask anything — coding, math, compliance, finance, general knowledge…"}
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => { if (e.ctrlKey && e.key === 'Enter') handle() }}
        />
        <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Ctrl+Enter to submit</span>
      </div>

      {/* Controls row */}
      <div className="query-controls">
        <div className="quality-slider-wrap">
          <div className="slider-header">
            <label className="input-label">Min Quality Threshold</label>
            <span className="slider-value">{quality.toFixed(2)}</span>
          </div>
          <input
            type="range" min="0.5" max="0.99" step="0.01"
            value={quality} onChange={e => setQuality(parseFloat(e.target.value))}
            id="quality-slider"
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text-muted)' }}>
            <span>0.50 — Fast Tier (Lightweight)</span>
            <span>0.76 — Balanced Tier</span>
            <span>0.99 — Frontier Tier (Flagship)</span>
          </div>
        </div>
        <button
          id="route-btn"
          className="btn-route"
          onClick={handle}
          disabled={loading || !query.trim()}
        >
          {loading ? <><div className="spinner" /> Routing &amp; Executing…</> : <>⚡ Route &amp; Execute Query</>}
        </button>
      </div>

      {/* Advanced: system prompt */}
      <div>
        <button
          onClick={() => setAdv(v => !v)}
          style={{ background: 'none', border: 'none', color: 'var(--text-muted)', fontSize: '0.75rem', cursor: 'pointer', padding: 0 }}
        >
          {advanced ? '▾' : '▸'} Advanced — System Prompt (enables prompt-cache discount)
        </button>
        {advanced && (
          <textarea
            className="query-textarea"
            style={{ marginTop: 8, minHeight: 60, fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}
            placeholder="Optional system instruction (e.g. 'You are a legal compliance assistant…')"
            value={sysPrompt}
            onChange={e => setSys(e.target.value)}
          />
        )}
      </div>
    </div>
  )
}
