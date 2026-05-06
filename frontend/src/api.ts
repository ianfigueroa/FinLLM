import type {
  ChatResponse,
  DocumentUploadRequest,
  EvalSummary,
  IngestionStatus,
  IngestionResult,
  RaftExperimentResult,
  SecUrlIngestionRequest
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    const detail = payload?.detail ? `: ${payload.detail}` : ''
    throw new Error(`API request failed with ${response.status}${detail}`)
  }
  return (await response.json()).data
}

export function ingestSample(): Promise<IngestionResult> {
  return postJson('/api/v1/ingestions/sample')
}

export function ingestSecUrl(payload: SecUrlIngestionRequest): Promise<IngestionResult> {
  return postJson('/api/v1/ingestions/sec-url', payload)
}

export async function uploadDocument(payload: DocumentUploadRequest): Promise<IngestionResult> {
  const body = new FormData()
  body.set('ticker', payload.ticker)
  body.set('company', payload.company)
  body.set('form_type', payload.form_type)
  body.set('filing_date', payload.filing_date)
  body.set('file', payload.file)

  const response = await fetch(`${API_BASE}/api/v1/documents/upload`, {
    method: 'POST',
    body
  })
  if (!response.ok) {
    const errorPayload = await response.json().catch(() => null)
    const detail = errorPayload?.detail ? `: ${errorPayload.detail}` : ''
    throw new Error(`API request failed with ${response.status}${detail}`)
  }
  return (await response.json()).data
}

export function sendChat(
  question: string,
  mode: string,
  filters?: Record<string, string>
): Promise<ChatResponse> {
  return postJson('/api/v1/chat', { question, mode, filters })
}

export function runEval(): Promise<EvalSummary> {
  return postJson('/api/v1/evals')
}

export function runRaftExperiment(): Promise<RaftExperimentResult> {
  return postJson('/api/v1/finetuning/raft', {
    max_examples: 20,
    distractor_count: 2,
    base_model: 'local-sim'
  })
}

export async function getIngestionStatus(): Promise<IngestionStatus> {
  const response = await fetch(`${API_BASE}/api/v1/ingestions/status`)
  if (!response.ok) {
    throw new Error(`API request failed with ${response.status}`)
  }
  return (await response.json()).data
}
