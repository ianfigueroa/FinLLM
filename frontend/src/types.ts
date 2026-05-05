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
  retrieved_chunks: RetrievedChunk[]
  verification: {
    passed: boolean
    used_markers: string[]
    invented_markers: string[]
    missing_citations: boolean
  }
}

export interface EvalSummary {
  regression_pass_rate: number
  cases: Array<Record<string, unknown>>
}
