import { useLocation, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import {
  ArrowLeft, Shield, AlertTriangle, CheckCircle, XCircle,
  Clock, ChevronDown, ChevronUp, ExternalLink, Info, Award
} from 'lucide-react'
import { getStandard } from '../services/api'

// ── Status badge ─────────────────────────────────────────────
function StatusBadge({ status }) {
  if (status === 'ACTIVE')
    return <span className="badge-active flex items-center gap-1"><CheckCircle className="w-3 h-3" /> ACTIVE</span>
  if (status === 'WITHDRAWN' || status === 'SUPERSEDED')
    return <span className="badge-withdrawn flex items-center gap-1"><XCircle className="w-3 h-3" /> {status}</span>
  return <span className="badge-review flex items-center gap-1"><Clock className="w-3 h-3" /> {status}</span>
}

// ── Verification badge ─────────────────────────────────────────
function VerificationBadge({ level }) {
  const map = {
    verified_multi_source: { cls: 'badge-active', label: '✓ Multi-source verified' },
    needs_review: { cls: 'badge-review', label: '⚠ Needs review' },
    unverified: { cls: 'badge-withdrawn', label: '✗ Unverified' },
    single_source_unconfirmed: { cls: 'badge-single', label: '~ Single source' },
  }
  const { cls, label } = map[level] || { cls: 'badge-single', label: level }
  return <span className={cls}>{label}</span>
}

// ── Confidence bar ─────────────────────────────────────────────
function ConfidenceBar({ value }) {
  const pct = Math.round(value * 100)
  const color = pct >= 75 ? 'bg-success-400' : pct >= 50 ? 'bg-brand-400' : 'bg-warning-400'
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
        <div className={`h-full ${color} transition-all duration-700`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-slate-300 w-8 text-right">{pct}%</span>
    </div>
  )
}

// ── Certification block ────────────────────────────────────────
function CertificationBlock({ cert }) {
  if (!cert) return (
    <div className="text-slate-500 text-xs italic">Certification status: unknown</div>
  )
  return (
    <div className={`rounded-lg px-3 py-2 text-xs ${cert.mandatory ? 'bg-danger-600/10 border border-danger-600/20' : 'bg-white/3 border border-white/10'}`}>
      <div className="flex items-center gap-2 mb-1">
        <Award className={`w-3.5 h-3.5 ${cert.mandatory ? 'text-danger-400' : 'text-slate-400'}`} />
        <span className={`font-semibold ${cert.mandatory ? 'text-danger-400' : 'text-slate-300'}`}>
          {cert.mandatory === true ? 'Mandatory Certification' : cert.mandatory === false ? 'Voluntary' : 'Status Unknown'}
        </span>
        {cert.scheme_name && <span className="text-slate-400 ml-1">— {cert.scheme_name}</span>}
      </div>
      {cert.gazette_reference && <div className="text-slate-400">Gazette: {cert.gazette_reference}</div>}
      {cert.penalty && <div className="text-danger-400/70 mt-1">Penalty: {cert.penalty.substring(0, 120)}{cert.penalty.length > 120 ? '…' : ''}</div>}
      <div className="text-slate-500 mt-1">Source: {cert.evidence_source}</div>
    </div>
  )
}

// ── Standard detail modal ──────────────────────────────────────
function StandardDetailModal({ standardKey, onClose }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useState(() => {
    getStandard(standardKey)
      .then(setData)
      .catch(() => setError('Could not load standard details'))
      .finally(() => setLoading(false))
  }, [standardKey])

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4" onClick={onClose}>
      <div className="glass-card max-w-lg w-full p-6 max-h-[80vh] overflow-y-auto animate-slide-up" onClick={e => e.stopPropagation()}>
        {loading && <div className="text-slate-400 text-center py-8">Loading…</div>}
        {error && <div className="text-danger-400 text-sm">{error}</div>}
        {data && (
          <>
            <div className="flex items-start justify-between mb-4">
              <div>
                <div className="text-brand-400 font-mono font-bold text-lg">{data.display_code}</div>
                <div className="text-white font-semibold text-sm mt-1">{data.title}</div>
              </div>
              <button onClick={onClose} className="text-slate-400 hover:text-white ml-4 flex-shrink-0">✕</button>
            </div>
            <div className="flex flex-wrap gap-2 mb-4">
              <StatusBadge status={data.status} />
              <VerificationBadge level={data.verification_level} />
            </div>
            {data.superseded_by?.length > 0 && (
              <div className="mb-4 px-3 py-2 bg-danger-600/10 border border-danger-600/20 rounded-lg text-xs text-danger-400">
                ⚠ Succeeded by: {data.superseded_by.join(', ')}
              </div>
            )}
            {data.scope && <p className="text-slate-300 text-sm mb-4 leading-relaxed">{data.scope}</p>}
            {data.related_standards?.length > 0 && (
              <div>
                <div className="text-xs text-slate-400 uppercase font-medium tracking-wider mb-2">Related Standards</div>
                <div className="space-y-1">
                  {data.related_standards.map(r => (
                    <div key={r.key} className="text-xs flex items-center gap-2 text-slate-300">
                      <span className="text-brand-400 font-mono">{r.display_code}</span>
                      <span className="text-slate-500">—</span>
                      <span>{r.title.substring(0, 60)}{r.title.length > 60 ? '…' : ''}</span>
                      <span className="badge-single text-[10px]">{r.relationship_type}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

// ── Recommendation card ────────────────────────────────────────
function RecommendationCard({ rec, rank }) {
  const [expanded, setExpanded] = useState(rank === 0)
  const [modalKey, setModalKey] = useState(null)

  const isProblematic = rec.status !== 'ACTIVE' || rec.flags?.some(f =>
    f.includes('needs_review') || f.includes('unverified')
  )

  const cardBorder = rec.status === 'ACTIVE'
    ? 'border-white/10'
    : rec.status === 'WITHDRAWN' || rec.status === 'SUPERSEDED'
    ? 'border-danger-600/30'
    : 'border-warning-600/30'

  return (
    <>
      <div className={`glass-card-hover border ${cardBorder} overflow-hidden animate-slide-up`}>
        {/* Card header */}
        <div className="p-5">
          <div className="flex items-start gap-4">
            {/* Rank */}
            <div className="w-8 h-8 rounded-xl bg-brand-600/30 flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-brand-300 text-sm font-bold">#{rank + 1}</span>
            </div>

            {/* Main info */}
            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap items-start gap-2 mb-1">
                <span className="text-brand-400 font-mono font-bold text-base">{rec.is_code}</span>
                <StatusBadge status={rec.status} />
                <VerificationBadge level={rec.verification_level} />
              </div>
              <h3 className="text-white text-sm font-semibold leading-snug mb-2">
                {rec.title}
              </h3>
              <ConfidenceBar value={rec.confidence} />
            </div>

            <button
              onClick={() => setExpanded(!expanded)}
              className="flex-shrink-0 text-slate-400 hover:text-white transition-colors p-1"
            >
              {expanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </button>
          </div>

          {/* Superseded warning — always visible if applicable */}
          {rec.superseded_by?.length > 0 && (
            <div className="ml-12 mt-3 px-3 py-2 bg-danger-600/10 border border-danger-600/25 rounded-lg flex items-start gap-2">
              <AlertTriangle className="w-3.5 h-3.5 text-danger-400 flex-shrink-0 mt-0.5" />
              <div className="text-xs text-danger-400">
                <span className="font-semibold">This standard has been {rec.status.toLowerCase()}.</span>{' '}
                Successor: {rec.superseded_by.join(', ')}
              </div>
            </div>
          )}

          {/* Quality flags — always visible */}
          {rec.flags?.length > 0 && (
            <div className="ml-12 mt-2 flex flex-wrap gap-1.5">
              {rec.flags.map(f => (
                <span key={f} className="text-[10px] px-2 py-0.5 bg-warning-600/10 border border-warning-600/20 text-warning-400/80 rounded-full">
                  {f.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Expanded content */}
        {expanded && (
          <div className="px-5 pb-5 border-t border-white/5 pt-4 space-y-4 animate-fade-in">
            {/* Reasoning */}
            <div>
              <div className="text-xs text-slate-400 uppercase font-medium tracking-wider mb-2 flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5" /> Why this standard?
              </div>
              <p className="text-slate-300 text-sm leading-relaxed">{rec.reasoning}</p>
            </div>

            {/* Certification */}
            <div>
              <div className="text-xs text-slate-400 uppercase font-medium tracking-wider mb-2 flex items-center gap-1.5">
                <Award className="w-3.5 h-3.5" /> Certification Requirement
              </div>
              <CertificationBlock cert={rec.certification} />
            </div>

            {/* Related standards */}
            {rec.related_standards?.length > 0 && (
              <div>
                <div className="text-xs text-slate-400 uppercase font-medium tracking-wider mb-2">
                  Related Standards (from knowledge graph)
                </div>
                <div className="space-y-1.5">
                  {rec.related_standards.slice(0, 6).map(r => (
                    <button
                      key={r.key}
                      onClick={() => setModalKey(r.key)}
                      className="flex items-center gap-2 text-xs w-full text-left px-3 py-1.5 bg-white/3 hover:bg-white/6 rounded-lg transition-colors group"
                    >
                      <span className="text-brand-400 font-mono">{r.display_code}</span>
                      <span className="text-slate-500">—</span>
                      <span className="text-slate-300 flex-1 truncate">{r.title}</span>
                      <span className="badge-single text-[10px]">{r.relationship_type}</span>
                      <ExternalLink className="w-3 h-3 text-slate-500 group-hover:text-brand-400 flex-shrink-0" />
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Evidence sources */}
            {rec.evidence_sources?.length > 0 && (
              <div className="text-xs text-slate-500">
                Sources: {rec.evidence_sources.join(' · ')}
              </div>
            )}
          </div>
        )}
      </div>

      {modalKey && (
        <StandardDetailModal standardKey={modalKey} onClose={() => setModalKey(null)} />
      )}
    </>
  )
}

// ── Audit banner ──────────────────────────────────────────────
function AuditBanner({ warnings }) {
  const hasIssues = warnings?.length > 0
  return (
    <div className={`flex items-center gap-3 px-4 py-3 rounded-xl text-xs border ${
      hasIssues
        ? 'bg-warning-600/10 border-warning-600/20 text-warning-400'
        : 'bg-success-600/10 border-success-600/20 text-success-400'
    }`}>
      <Shield className="w-4 h-4 flex-shrink-0" />
      <span className="font-semibold">
        Verified Indian Standards + Evidence + Human Review
        {' · '}<span className="opacity-80">Transparent | Auditable | Reliable</span>
      </span>
      {hasIssues && (
        <span className="ml-auto flex-shrink-0 opacity-70">{warnings.length} degraded store warning(s)</span>
      )}
    </div>
  )
}

// ── Results page ──────────────────────────────────────────────
export default function ResultsPage() {
  const { state } = useLocation()
  const navigate = useNavigate()
  const result = state?.recommendResult

  if (!result) {
    navigate('/')
    return null
  }

  const { audit_id, query_summary, recommendations, warnings, pipeline_stages_completed } = result

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="px-6 py-5 flex items-center justify-between border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-brand-600 flex items-center justify-center">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <span className="font-bold text-lg text-white">BIS Sahayak</span>
        </div>
        <button onClick={() => navigate('/')} className="btn-secondary flex items-center gap-2 text-sm py-2">
          <ArrowLeft className="w-4 h-4" /> New Search
        </button>
      </header>

      <main className="flex-1 max-w-3xl mx-auto w-full px-4 py-8">
        {/* Query summary */}
        <div className="mb-6 animate-fade-in">
          <h2 className="text-xl font-bold text-white mb-1">Recommendations</h2>
          <p className="text-slate-400 text-sm">{query_summary}</p>
          <div className="flex flex-wrap gap-2 mt-3">
            <span className="text-xs text-slate-500 font-mono">Audit: {audit_id?.substring(0, 8)}…</span>
            {pipeline_stages_completed?.map(s => (
              <span key={s} className="badge-verified text-[10px]">{s}</span>
            ))}
          </div>
        </div>

        {/* Degraded warnings */}
        {warnings?.length > 0 && (
          <div className="mb-5 glass-card p-4 border-warning-600/20 bg-warning-600/5">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-warning-400 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-warning-400 text-sm font-medium mb-1">System Warnings (degraded mode)</p>
                {warnings.map((w, i) => (
                  <p key={i} className="text-warning-400/70 text-xs">{w}</p>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {recommendations.length === 0 ? (
          <div className="glass-card p-10 text-center">
            <XCircle className="w-12 h-12 text-slate-500 mx-auto mb-3" />
            <p className="text-slate-300 font-medium">No recommendations found</p>
            <p className="text-slate-500 text-sm mt-1">Try rephrasing your requirement or check the API health.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {recommendations.map((rec, i) => (
              <RecommendationCard key={rec.key || i} rec={rec} rank={i} />
            ))}
          </div>
        )}

        {/* Audit banner */}
        <div className="mt-8 animate-fade-in">
          <AuditBanner warnings={warnings} />
        </div>
      </main>
    </div>
  )
}
