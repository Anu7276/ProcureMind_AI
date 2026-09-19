import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, Image, Loader2, Sparkles, Shield, Zap } from 'lucide-react'
import { ingestFile, ingestText } from '../services/api'

const ACCEPTED_TYPES = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'text/plain': ['.txt'],
}

const features = [
  { icon: Shield, label: 'Verified Standards', desc: '1,380 IS codes from BIS' },
  { icon: Sparkles, label: 'AI Powered', desc: 'LangGraph 5-node pipeline' },
  { icon: Zap, label: 'Human Review', desc: 'Inspect before you commit' },
]

export default function UploadPage() {
  const navigate = useNavigate()
  const [mode, setMode] = useState('file') // 'file' | 'text'
  const [textInput, setTextInput] = useState('')
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const onDrop = useCallback((accepted) => {
    if (accepted.length > 0) {
      setFile(accepted[0])
      setError('')
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxFiles: 1,
    onDropRejected: () => setError('Unsupported file type. Use PDF, DOCX, PNG, JPEG, or TXT.'),
  })

  const handleSubmit = async () => {
    setError('')
    setLoading(true)
    try {
      let result
      if (mode === 'file' && file) {
        result = await ingestFile(file)
      } else if (mode === 'text' && textInput.trim()) {
        result = await ingestText(textInput.trim())
      } else {
        setError(mode === 'file' ? 'Please select a file.' : 'Please enter your requirement.')
        setLoading(false)
        return
      }
      // Pass to review page via state
      navigate('/review', { state: { ingestResult: result } })
    } catch (err) {
      setError(err?.response?.data?.detail || 'Failed to process input. Is the API running?')
    } finally {
      setLoading(false)
    }
  }

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
        <span className="text-xs text-slate-400 font-mono bg-white/5 px-3 py-1 rounded-full border border-white/10">
          v1.0 · Gemini 1.5 Flash
        </span>
      </header>

      {/* Hero */}
      <main className="flex-1 flex flex-col items-center justify-center px-4 py-12 max-w-4xl mx-auto w-full">
        <div className="text-center mb-10 animate-fade-in">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-brand-600/20 border border-brand-500/30 text-brand-300 text-sm font-medium mb-6">
            <Sparkles className="w-4 h-4" />
            AI-Powered Indian Standards Recommendation
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold text-white mb-4 leading-tight">
            Find the Right<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-400 to-indigo-400">
              IS Code, Every Time
            </span>
          </h1>
          <p className="text-slate-400 text-lg max-w-xl mx-auto">
            Submit a tender specification or procurement document. Our AI engine returns
            verified Indian Standards with evidence, compliance status, and plain-language reasoning.
          </p>
        </div>

        {/* Feature pills */}
        <div className="flex flex-wrap gap-3 justify-center mb-10">
          {features.map(({ icon: Icon, label, desc }) => (
            <div key={label} className="glass-card px-4 py-3 flex items-center gap-3 animate-slide-up">
              <div className="w-8 h-8 rounded-lg bg-brand-600/30 flex items-center justify-center">
                <Icon className="w-4 h-4 text-brand-300" />
              </div>
              <div>
                <div className="text-sm font-semibold text-white">{label}</div>
                <div className="text-xs text-slate-400">{desc}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Input mode toggle */}
        <div className="w-full max-w-2xl glass-card p-6 animate-slide-up">
          <div className="flex gap-2 mb-6 p-1 bg-white/5 rounded-xl">
            {[['file', 'Upload Document', Upload], ['text', 'Type Requirement', FileText]].map(([m, label, Icon]) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all duration-200
                  ${mode === m ? 'bg-brand-600 text-white shadow-lg shadow-brand-600/30' : 'text-slate-400 hover:text-slate-200'}`}
              >
                <Icon className="w-4 h-4" />
                {label}
              </button>
            ))}
          </div>

          {mode === 'file' ? (
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all duration-200
                ${isDragActive
                  ? 'border-brand-400 bg-brand-500/10'
                  : 'border-white/20 hover:border-brand-500/50 hover:bg-white/3'
                }
                ${file ? 'border-brand-500/50 bg-brand-500/5' : ''}`}
            >
              <input {...getInputProps()} />
              {file ? (
                <div className="flex flex-col items-center gap-3">
                  <div className="w-14 h-14 rounded-2xl bg-brand-600/20 flex items-center justify-center">
                    <FileText className="w-7 h-7 text-brand-400" />
                  </div>
                  <div>
                    <p className="text-white font-medium">{file.name}</p>
                    <p className="text-slate-400 text-sm">{(file.size / 1024).toFixed(1)} KB · Click to change</p>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3">
                  <div className="w-14 h-14 rounded-2xl bg-white/5 flex items-center justify-center">
                    <Upload className={`w-7 h-7 ${isDragActive ? 'text-brand-400' : 'text-slate-500'}`} />
                  </div>
                  <div>
                    <p className="text-slate-300 font-medium">
                      {isDragActive ? 'Drop it here!' : 'Drop file or click to browse'}
                    </p>
                    <p className="text-slate-500 text-sm mt-1">PDF, DOCX, PNG, JPEG, TXT</p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <textarea
              id="requirement-text"
              className="input-field h-40 text-sm"
              placeholder="Describe your procurement requirement, e.g.:&#10;Supply of TMT reinforcement bars, Fe 500D grade, for RCC column and beam work..."
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
            />
          )}

          {error && (
            <div className="mt-4 px-4 py-3 bg-danger-600/15 border border-danger-600/30 rounded-xl text-danger-400 text-sm">
              {error}
            </div>
          )}

          <button
            id="submit-btn"
            onClick={handleSubmit}
            disabled={loading}
            className="btn-primary w-full mt-5 flex items-center justify-center gap-2 text-base"
          >
            {loading ? (
              <><Loader2 className="w-5 h-5 animate-spin" /> Analysing…</>
            ) : (
              <><Sparkles className="w-5 h-5" /> Analyse & Extract Requirements</>
            )}
          </button>
        </div>
      </main>
    </div>
  )
}
