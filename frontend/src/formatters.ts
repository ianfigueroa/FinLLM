import type { EvalSummary } from './types'

const MODE_LABELS: Record<string, string> = {
  basic_rag: 'Basic',
  rag_rerank: 'Rerank',
  self_verify: 'Verify'
}

export function modeLabel(mode: string): string {
  return MODE_LABELS[mode] ?? mode
}

export function retrievalMethod(mode: string): string {
  switch (mode) {
    case 'basic_rag':
      return 'BM25 + dense'
    case 'rag_rerank':
      return 'BM25 + dense - reranked'
    case 'self_verify':
      return 'BM25 + dense - reranked - verified'
    default:
      return mode
  }
}

export function clamp(text: string, max: number): string {
  if (text.length <= max) return text
  return `${text.slice(0, max).trimEnd()}...`
}

export function formatClock(): string {
  const now = new Date()
  const hh = String(now.getHours()).padStart(2, '0')
  const mm = String(now.getMinutes()).padStart(2, '0')
  return `${hh}:${mm}`
}

export function formatCost(summary: EvalSummary | null): string {
  if (!summary || summary.mode_results.length === 0) return 'Not estimated'
  const total = summary.mode_results.reduce((acc, m) => acc + m.estimated_cost_usd, 0)
  const avg = total / summary.mode_results.length
  if (avg <= 0) return 'Local'
  return `est $${avg.toFixed(5)}`
}
