import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

export const ingestText = async (text) => {
  const formData = new FormData()
  formData.append('text', text)
  const res = await api.post('/ingest', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export const ingestFile = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  const res = await api.post('/ingest', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export const recommend = async ({ rawQuery, structuredRequirement, auditId }) => {
  const payload = {}
  if (rawQuery) payload.raw_query = rawQuery
  if (structuredRequirement) payload.structured_requirement = structuredRequirement
  if (auditId) payload.audit_id = auditId
  const res = await api.post('/recommend', payload)
  return res.data
}

export const getStandard = async (key) => {
  const res = await api.get(`/standard/${encodeURIComponent(key)}`)
  return res.data
}

export const getHealth = async () => {
  const res = await api.get('/health')
  return res.data
}

export default api
