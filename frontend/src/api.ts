import type { ChatResponse, EvalSummary, IngestionResult, SecUrlIngestionRequest } from './types'

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

export function sendChat(question: string, mode: string): Promise<ChatResponse> {
  return postJson('/api/v1/chat', { question, mode })
}

export function runEval(): Promise<EvalSummary> {
  return postJson('/api/v1/evals')
}

export async function getIngestionStatus(): Promise<{ chunks_indexed: number }> {
  const response = await fetch(`${API_BASE}/api/v1/ingestions/status`)
  if (!response.ok) {
    throw new Error(`API request failed with ${response.status}`)
  }
  return (await response.json()).data
}
