import { useLocation, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import { ArrowLeft, ArrowRight, AlertTriangle, CheckCircle, Loader2, Shield } from 'lucide-react'
import { recommend } from '../services/api'

const Field = ({ label, value }) => {
  if (!value) return null
  return (
    <div className="glass-card p-4">
      <div className="text-xs text-slate-400 font-medium uppercase tracking-wider mb-1.5">{label}</div>
      <div className="text-slate-200 text-sm leading-relaxed">{value}</div>
    </div>
  )
}

export default function ReviewPage() {
  const { state } = useLocation()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const ingestResult = state?.ingestResult
  if (!ingestResult) {
    navigate('/')
    return null
  }

  const { audit_id, structured_requirement: req, warnings } = ingestResult

  const handleConfirm = async () => {
    setLoading(true)
    setError('')
    try {
      const result = await recommend({
        structuredRequirement: req,
        auditId: audit_id,
      })
      navigate('/results', { state: { recommendResult: result } })
    } catch (err) {
      setError(err?.response?.data?.detail || 'Recommendation failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const verLevel = req.verification_level

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
          <ArrowLeft className="w-4 h-4" /> Back
        </button>
      </header>

      <main className="flex-1 max-w-3xl mx-auto w-full px-4 py-10 animate-fade-in">
        {/* Title */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <CheckCircle className="w-6 h-6 text-success-400" />
            <h2 className="text-2xl font-bold text-white">Human Review Checkpoint</h2>
          </div>
          <p className="text-slate-400 text-sm ml-9">
            Review what the AI extracted from your input. Confirm to run the recommendation engine.
          </p>
        </div>

        {/* Warnings */}
        {warnings?.length > 0 && (
          <div className="mb-6 glass-card p-4 border-warning-600/30 bg-warning-600/5">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-warning-400 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-warning-400 text-sm font-medium mb-1">Pipeline Warnings</p>
                {warnings.map((w, i) => (
                  <p key={i} className="text-warning-400/80 text-xs">{w}</p>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Language / input type metadata */}
        <div className="flex gap-3 mb-6">
          <span className="badge-verified">Language: {req.language || 'en'}</span>
          <span className="badge-verified">Input: {req.input_type || 'text'}</span>
          {req.thesaurus_expansions?.length > 0 && (
            <span className="badge-verified">{req.thesaurus_expansions.length} thesaurus matches</span>
          )}
        </div>

        {/* Structured requirement fields */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
          <Field label="Product" value={req.product} />
          <Field label="Material / Grade" value={req.material} />
          <Field label="Specifications" value={req.specifications} />
          <Field label="Performance Requirements" value={req.performance_requirements} />
          <Field label="Safety Requirements" value={req.safety_requirements} />
          <Field label="Application" value={req.application} />
          <Field label="Category Hint" value={req.category_hint} />
        </div>

        {/* Normalized text */}
        {req.normalized_text && (
          <div className="glass-card p-4 mb-6">
            <div className="text-xs text-slate-400 font-medium uppercase tracking-wider mb-1.5">
              Normalised Input Text
            </div>
            <div className="text-slate-400 text-xs leading-relaxed font-mono max-h-28 overflow-y-auto">
              {req.normalized_text.substring(0, 600)}{req.normalized_text.length > 600 ? '…' : ''}
            </div>
          </div>
        )}

        {error && (
          <div className="mb-6 px-4 py-3 bg-danger-600/15 border border-danger-600/30 rounded-xl text-danger-400 text-sm">
            {error}
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-3">
          <button onClick={() => navigate('/')} className="btn-secondary flex items-center gap-2">
            <ArrowLeft className="w-4 h-4" /> Start Over
          </button>
          <button
            id="confirm-recommend-btn"
            onClick={handleConfirm}
            disabled={loading}
            className="btn-primary flex-1 flex items-center justify-center gap-2"
          >
            {loading ? (
              <><Loader2 className="w-5 h-5 animate-spin" /> Finding Standards…</>
            ) : (
              <>Confirm & Get Recommendations <ArrowRight className="w-5 h-5" /></>
            )}
          </button>
        </div>

        <p className="text-center text-slate-500 text-xs mt-6">
          Audit ID: <span className="font-mono text-slate-400">{audit_id}</span>
        </p>
      </main>
    </div>
  )
}
