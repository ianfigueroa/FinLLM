export interface Citation {
  marker: string
  chunk_id: string
  source: string
  section: string | null
  ticker: string | null
  start_char: number
  end_char: number
  snippet: string
}

export interface RetrievedChunk {
  chunk_id: string
  text: string
  score: number
  metadata: Record<string, string | null>
}

export interface ChatResponse {
  answer: string
  confidence: number
  limitations: string[]
  disclaimer: string
  mode: string
  citations: Citation[]
  tool_calls: ToolCall[]
  retrieved_chunks: RetrievedChunk[]
  verification: {
    passed: boolean
    used_markers: string[]
    invented_markers: string[]
    missing_citations: boolean
  }
}

export interface ToolCall {
  name: string
  input: Record<string, unknown>
  ok: boolean
  output: unknown
  error: string
}

export interface EvalSummary {
  regression_pass_rate: number
  cases: Array<Record<string, unknown>>
  mode_results: ModeEvalResult[]
}

export interface ModeEvalResult {
  mode: string
  retrieval_precision: number
  context_relevance: number
  citation_correctness: number
  hallucination_rate: number
  answer_relevance: number
  avg_latency_ms: number
  estimated_cost_usd: number
  tool_call_success_rate: number
}

export interface IngestionResult {
  documents_ingested: number
  chunks_indexed: number
  ticker?: string
  source_url?: string
}

export interface SecUrlIngestionRequest {
  url: string
  ticker: string
  company: string
  form_type: string
  filing_date: string
}

export interface RaftExperimentResult {
  raft_examples: number
  lora_records: number
  preview: Array<Record<string, unknown>>
  training_report: Record<string, unknown>
  eval_report: Record<string, unknown>
}

export interface DocumentUploadRequest {
  ticker: string
  company: string
  form_type: string
  filing_date: string
  file: File
}
