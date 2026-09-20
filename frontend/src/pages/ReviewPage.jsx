import { useLocation, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import { ArrowLeft, ArrowRight, AlertTriangle, CheckCircle, Loader2, Shield, Edit3, X, RotateCcw } from 'lucide-react'
import { recommend } from '../services/api'

// ── Editable field ──────────────────────────────────────────────
function EditableField({ label, fieldKey, value, onChange }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(value || '')

  const save = () => {
    onChange(fieldKey, draft.trim())
    setEditing(false)
  }
  const cancel = () => {
    setDraft(value || '')
    setEditing(false)
  }

  if (!value && !editing) return null

  return (
    <div className="glass-card p-4 group relative">
      <div className="flex items-center justify-between mb-1.5">
        <div className="text-xs text-slate-400 font-medium uppercase tracking-wider">{label}</div>
        {!editing && (
          <button
            onClick={() => setEditing(true)}
            className="opacity-0 group-hover:opacity-100 transition-opacity text-slate-500 hover:text-brand-400 p-0.5 rounded"
            title={`Edit ${label}`}
          >
            <Edit3 className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {editing ? (
        <div className="space-y-2">
          <textarea
            className="w-full bg-white/5 border border-brand-500/40 rounded-lg px-3 py-2 text-slate-200 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-brand-500"
            rows={Math.max(2, Math.ceil((draft.length || 1) / 60))}
            value={draft}
            autoFocus
            onChange={e => setDraft(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); save() } }}
          />
          <div className="flex gap-2">
            <button onClick={save} className="btn-primary py-1 px-3 text-xs flex items-center gap-1">
              <CheckCircle className="w-3 h-3" /> Save
            </button>
            <button onClick={cancel} className="btn-secondary py-1 px-3 text-xs flex items-center gap-1">
              <X className="w-3 h-3" /> Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="text-slate-200 text-sm leading-relaxed">{value}</div>
      )}
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

  const { audit_id, structured_requirement: originalReq, warnings } = ingestResult

  // Local editable copy of the structured requirement
  const [req, setReq] = useState({ ...originalReq })

  const handleFieldChange = (fieldKey, newValue) => {
    setReq(prev => ({ ...prev, [fieldKey]: newValue }))
  }

  const handleReset = () => {
    setReq({ ...originalReq })
  }

  const hasEdits = JSON.stringify(req) !== JSON.stringify(originalReq)

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

  const editableFields = [
    { label: 'Product / Item', key: 'product' },
    { label: 'Material / Grade', key: 'material' },
    { label: 'Specifications', key: 'specifications' },
    { label: 'Performance Requirements', key: 'performance_requirements' },
    { label: 'Safety Requirements', key: 'safety_requirements' },
    { label: 'Application', key: 'application' },
    { label: 'Category Hint', key: 'category_hint' },
  ]

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
            Review what the AI extracted. You can <span className="text-brand-400 font-medium">edit any field</span> by
            hovering and clicking the pencil icon. Confirm to run the recommendation engine.
          </p>
        </div>

        {/* Edit tip */}
        <div className="mb-5 flex items-center gap-2 px-3 py-2 bg-brand-600/10 border border-brand-600/20 rounded-xl text-xs text-brand-300">
          <Edit3 className="w-3.5 h-3.5 flex-shrink-0" />
          <span>Hover over any field below to edit it. Press <kbd className="px-1 py-0.5 bg-white/10 rounded text-white">Enter</kbd> to save.</span>
          {hasEdits && (
            <button onClick={handleReset} className="ml-auto flex items-center gap-1 text-slate-400 hover:text-white transition-colors">
              <RotateCcw className="w-3 h-3" /> Reset
            </button>
          )}
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

        {/* Metadata badges */}
        <div className="flex gap-3 mb-6 flex-wrap">
          <span className="badge-verified">Language: {req.language || 'en'}</span>
          <span className="badge-verified">Input: {req.input_type || 'text'}</span>
          {req.thesaurus_expansions?.length > 0 && (
            <span className="badge-verified">{req.thesaurus_expansions.length} thesaurus matches</span>
          )}
          {hasEdits && (
            <span className="px-2 py-0.5 text-[10px] bg-brand-600/20 border border-brand-600/30 text-brand-300 rounded-full font-medium">
              ✎ Edited
            </span>
          )}
        </div>

        {/* Editable structured requirement fields */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
          {editableFields.map(({ label, key }) => (
            <EditableField
              key={key}
              label={label}
              fieldKey={key}
              value={req[key]}
              onChange={handleFieldChange}
            />
          ))}
        </div>

        {/* Normalized text (read-only) */}
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
              <>Confirm {hasEdits ? '(with edits)' : ''} & Get Recommendations <ArrowRight className="w-5 h-5" /></>
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
