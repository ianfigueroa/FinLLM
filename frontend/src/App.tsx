import {
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Database,
  FileSearch,
  Link2,
  Send,
  ShieldCheck,
  Upload
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import {
  detectSecMetadata,
  getIngestionStatus,
  ingestSample,
  ingestSecUrl,
  runEval,
  runRaftExperiment,
  sendChat,
  uploadDocument
} from './api'
import type { ChatResponse, Citation, EvalSummary, IngestionStatus, RaftExperimentResult } from './types'

const FALLBACK_QUESTION = 'What are the main risk factors in the indexed filing?'

const MODES = [
  { id: 'basic_rag', label: 'Basic' },
  { id: 'rag_rerank', label: 'Rerank' },
  { id: 'self_verify', label: 'Verify' }
]

const SOURCE_TABS = [
  { id: 'sec', label: 'SEC filing' },
  { id: 'upload', label: 'Upload' },
  { id: 'demo', label: 'Demo' }
] as const

const SEARCH_SCOPES = [
  { id: 'active', label: 'Active source' },
  { id: 'all', label: 'All corpus' }
] as const

const LOADING_LABELS: Record<string, string> = {
  ingest: 'Indexing',
  'sec-url': 'Indexing SEC',
  upload: 'Uploading',
  chat: 'Researching',
  eval: 'Evaluating',
  raft: 'Generating'
}

const EVIDENCE_PREVIEW_CHARS = 280
const QUESTION_ROTATION_MS = 6000
const SEC_METADATA_DEBOUNCE_MS = 700

type SourceTab = (typeof SOURCE_TABS)[number]['id']
type SearchScope = (typeof SEARCH_SCOPES)[number]['id']

export function App() {
  const [question, setQuestion] = useState(FALLBACK_QUESTION)
  const [mode, setMode] = useState('rag_rerank')
  const [sourceTab, setSourceTab] = useState<SourceTab>('sec')
  const [searchScope, setSearchScope] = useState<SearchScope>('active')
  const [suggestionIndex, setSuggestionIndex] = useState(0)
  const [status, setStatus] = useState<IngestionStatus>({ chunks_indexed: 0, documents: [] })
  const [indexedAt, setIndexedAt] = useState<string>('')
  const [chat, setChat] = useState<ChatResponse | null>(null)
  const [evalSummary, setEvalSummary] = useState<EvalSummary | null>(null)
  const [raftSummary, setRaftSummary] = useState<RaftExperimentResult | null>(null)
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null)
  const [secForm, setSecForm] = useState({
    url: 'https://www.sec.gov/ix?doc=/Archives/edgar/data/0001045810/000104581026000021/nvda-20260125.htm',
    ticker: 'NVDA',
    company: 'NVIDIA',
    form_type: '10-K',
    filing_date: '2026-01-25'
  })
  const [uploadForm, setUploadForm] = useState({
    ticker: 'DOC',
    company: 'Uploaded Company',
    form_type: '10-K',
    filing_date: '2026-01-01'
  })
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [lastSource, setLastSource] = useState('No source indexed')
  const [loading, setLoading] = useState('')
  const [metadataStatus, setMetadataStatus] = useState('')
  const [error, setError] = useState('')
  const [latencyMs, setLatencyMs] = useState(0)

  useEffect(() => {
    getIngestionStatus().then(setStatus).catch(() => setStatus({ chunks_indexed: 0, documents: [] }))
  }, [])

  useEffect(() => {
    const url = secForm.url.trim()
    if (!isSecFilingCandidate(url)) {
      setMetadataStatus('')
      return
    }

    let active = true
    setMetadataStatus('Waiting to detect')
    const timer = window.setTimeout(() => {
      setMetadataStatus('Detecting metadata')
      detectSecMetadata(url)
        .then((metadata) => {
          if (!active) return
          setSecForm((current) => {
            if (current.url.trim() !== url) return current
            return {
              ...current,
              ticker: metadata.ticker || current.ticker,
              company: metadata.company || current.company,
              form_type: metadata.form_type || current.form_type,
              filing_date: metadata.filing_date || current.filing_date
            }
          })
          setMetadataStatus(metadata.ticker ? `Detected ${metadata.ticker}` : 'Metadata detected')
        })
        .catch(() => {
          if (active) setMetadataStatus('Metadata not detected')
        })
    }, SEC_METADATA_DEBOUNCE_MS)

    return () => {
      active = false
      window.clearTimeout(timer)
    }
  }, [secForm.url])

  const selectedChunk = useMemo(() => {
    if (!chat || !selectedCitation) return null
    return chat.retrieved_chunks.find((chunk) => chunk.chunk_id === selectedCitation.chunk_id) ?? null
  }, [chat, selectedCitation])

  const activeSource = useMemo(() => {
    if (sourceTab === 'demo') {
      return { ticker: 'ACME', company: 'Acme Corp', formType: '10-K' }
    }
    if (sourceTab === 'upload') {
      return {
        ticker: uploadForm.ticker.toUpperCase(),
        company: uploadForm.company,
        formType: uploadForm.form_type.toUpperCase()
      }
    }
    return {
      ticker: secForm.ticker.toUpperCase(),
      company: secForm.company,
      formType: secForm.form_type.toUpperCase()
    }
  }, [secForm.company, secForm.form_type, secForm.ticker, sourceTab, uploadForm])

  const questionSuggestions = useMemo(
    () => buildQuestionSuggestions(activeSource.company, activeSource.ticker),
    [activeSource.company, activeSource.ticker]
  )

  useEffect(() => {
    setSuggestionIndex(0)
  }, [activeSource.company, activeSource.ticker])

  useEffect(() => {
    const timer = window.setInterval(() => {
      setSuggestionIndex((current) => (current + 1) % questionSuggestions.length)
    }, QUESTION_ROTATION_MS)
    return () => window.clearInterval(timer)
  }, [questionSuggestions.length])

  const isBusy = Boolean(loading)
  const canIndexSec = Boolean(secForm.url.trim())
  const canUpload = Boolean(
    uploadFile &&
      uploadForm.ticker.trim() &&
      uploadForm.company.trim() &&
      uploadForm.form_type.trim() &&
      uploadForm.filing_date.trim()
  )
  const citationStatus = chat
    ? chat.verification.passed
      ? 'Pass'
      : 'Review'
    : 'Pending'
  const citationStatusClass = chat ? (chat.verification.passed ? 'pass' : 'warn') : ''
  const responseMethod = retrievalMethod(chat?.mode ?? mode)
  const costDisplay = formatCost(evalSummary)
  const rotatingQuestion = questionSuggestions[suggestionIndex % questionSuggestions.length]
  const activeTickerFilter = activeSource.ticker
  const activeSearchFilters =
    searchScope === 'active' && activeTickerFilter.trim()
      ? { ticker: activeTickerFilter.trim().toUpperCase() }
      : undefined

  async function handleIngest() {
    await runAction('ingest', async () => {
      const result = await ingestSample()
      setStatus(await getIngestionStatus())
      setIndexedAt(formatClock())
      setSourceTab('demo')
      setLastSource(`ACME sample - ${result.chunks_indexed} chunks`)
    })
  }

  async function handleSecUrlIngest() {
    await runAction('sec-url', async () => {
      const result = await ingestSecUrl(secForm)
      setStatus(await getIngestionStatus())
      setIndexedAt(formatClock())
      setSourceTab('sec')
      setSecForm((current) => ({
        ...current,
        ticker: result.ticker ?? current.ticker,
        company: result.company ?? current.company,
        form_type: result.form_type ?? current.form_type,
        filing_date: result.filing_date ?? current.filing_date
      }))
      setLastSource(`${result.ticker ?? secForm.ticker.toUpperCase()} SEC filing`)
    })
  }

  async function handleUpload() {
    if (!uploadFile) return
    await runAction('upload', async () => {
      const result = await uploadDocument({ ...uploadForm, file: uploadFile })
      setStatus(await getIngestionStatus())
      setIndexedAt(formatClock())
      setSourceTab('upload')
      setLastSource(`${result.ticker ?? uploadForm.ticker.toUpperCase()} upload`)
    })
  }

  async function handleAsk() {
    await runAction('chat', async () => {
      const started = performance.now()
      const result = await sendChat(question, mode, activeSearchFilters)
      setLatencyMs(Math.round(performance.now() - started))
      setChat(result)
      setSelectedCitation(result.citations[0] ?? null)
    })
  }

  async function handleEval() {
    await runAction('eval', async () => {
      setEvalSummary(await runEval())
    })
  }

  async function handleRaft() {
    await runAction('raft', async () => {
      setRaftSummary(await runRaftExperiment())
    })
  }

  async function runAction(name: string, action: () => Promise<void>) {
    setError('')
    setLoading(name)
    try {
      await action()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error')
    } finally {
      setLoading('')
    }
  }

  function handleQuestionKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault()
      if (!isBusy && question.trim()) {
        void handleAsk()
      }
    }
  }

  function cycleQuestion(direction: -1 | 1) {
    setSuggestionIndex((current) => {
      const next = current + direction
      return (next + questionSuggestions.length) % questionSuggestions.length
    })
  }

  return (
    <main className="workspace">
      <header className="app-header">
        <div>
          <h1>FinLLM Research Agent</h1>
          <p className="subtitle">
            Retrieval-augmented financial research with citations, tool use, and evaluation.
          </p>
        </div>
        <div className="header-actions">
          <button onClick={handleEval} disabled={isBusy} title="Run local evals">
            <BarChart3 size={14} /> Run eval
          </button>
          <button onClick={handleRaft} disabled={isBusy} title="Generate RAFT examples">
            <FileSearch size={14} /> Generate RAFT data
          </button>
        </div>
      </header>

      {error && <div className="notice error">{error}</div>}

      <section className="summary-grid" aria-label="Research status">
        <Metric
          icon={<Database size={15} />}
          label="Corpus"
          value={`${status.chunks_indexed} chunks`}
          sub={indexedAt ? `indexed ${indexedAt}` : 'not yet indexed'}
        />
        <Metric
          icon={<Clock3 size={15} />}
          label="Latency"
          value={`${latencyMs} ms`}
          sub="last query"
        />
        <Metric
          icon={<ShieldCheck size={15} />}
          label="Citations"
          value={citationStatus}
          sub={chat ? `${chat.citations.length} markers` : 'awaiting query'}
        />
        <Metric
          icon={<BarChart3 size={15} />}
          label="Confidence"
          value={chat ? `${Math.round(chat.confidence * 100)}%` : '-'}
          sub={chat ? `mode ${chat.mode}` : 'no response'}
        />
      </section>

      <section className="app-grid">
        <aside className="source-column">
          <PanelTitle title="Sources" detail={lastSource} />
          {isBusy && <div className="inline-status">{LOADING_LABELS[loading]}</div>}
          <div className="source-tabs" role="tablist" aria-label="Source type">
            {SOURCE_TABS.map((tab) => (
              <button
                key={tab.id}
                className={sourceTab === tab.id ? 'active' : ''}
                onClick={() => setSourceTab(tab.id)}
                type="button"
              >
                {tab.label}
              </button>
            ))}
          </div>

          {sourceTab === 'demo' && (
            <div className="source-block">
              <div className="source-kicker">
                <strong>ACME 10-K fixture</strong>
                <span>Built-in eval source</span>
              </div>
              <button className="secondary-action full-button" onClick={handleIngest} disabled={isBusy}>
                <Database size={15} /> Load demo sample
              </button>
            </div>
          )}

          {sourceTab === 'sec' && (
            <div className="source-block">
              <label className="field full">
                <span>SEC URL</span>
                <input
                  value={secForm.url}
                  onChange={(event) => setSecForm({ ...secForm, url: event.target.value })}
                />
              </label>
              {metadataStatus && <div className="metadata-status">{metadataStatus}</div>}
              <div className="field-row compact">
                <TextField label="Ticker" value={secForm.ticker} onChange={(ticker) => setSecForm({ ...secForm, ticker })} />
                <TextField label="Form" value={secForm.form_type} onChange={(form_type) => setSecForm({ ...secForm, form_type })} />
              </div>
              <TextField label="Company" value={secForm.company} onChange={(company) => setSecForm({ ...secForm, company })} />
              <TextField label="Filing date" value={secForm.filing_date} onChange={(filing_date) => setSecForm({ ...secForm, filing_date })} />
              <button className="primary-action full-button" onClick={handleSecUrlIngest} disabled={isBusy || !canIndexSec}>
                <Link2 size={15} /> Index SEC filing
              </button>
            </div>
          )}

          {sourceTab === 'upload' && (
            <div className="source-block">
              <label className="field full">
                <span>Upload</span>
                <input
                  type="file"
                  accept=".txt,.text,.pdf,text/plain,application/pdf"
                  onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)}
                />
              </label>
              <div className="file-name">{uploadFile?.name ?? 'No file selected'}</div>
              <div className="field-row compact">
                <TextField label="Ticker" value={uploadForm.ticker} onChange={(ticker) => setUploadForm({ ...uploadForm, ticker })} />
                <TextField label="Form" value={uploadForm.form_type} onChange={(form_type) => setUploadForm({ ...uploadForm, form_type })} />
              </div>
              <TextField label="Company" value={uploadForm.company} onChange={(company) => setUploadForm({ ...uploadForm, company })} />
              <TextField label="Filing date" value={uploadForm.filing_date} onChange={(filing_date) => setUploadForm({ ...uploadForm, filing_date })} />
              <button className="secondary-action full-button" onClick={handleUpload} disabled={isBusy || !canUpload}>
                <Upload size={15} /> Upload filing
              </button>
            </div>
          )}

          <div className="source-summary">
            <span>Query scope</span>
            <strong>{searchScope === 'active' ? activeTickerFilter || 'Active source' : 'All corpus'}</strong>
          </div>

          <section className="indexed-docs" aria-label="Indexed documents">
            <div>
              <strong>Indexed documents</strong>
              <span>{status.documents.length} sources</span>
            </div>
            {status.documents.length ? (
              <div className="doc-list">
                {status.documents.map((document) => (
                  <article
                    key={`${document.ticker}-${document.form_type}-${document.filing_date}-${document.source}`}
                  >
                    <div>
                      <strong>{document.ticker ?? 'DOC'}</strong>
                      <span>{document.chunk_count} chunks</span>
                    </div>
                    <p>{document.company ?? 'Unknown company'}</p>
                    <small>
                      {document.form_type ?? 'Filing'} - {document.filing_date ?? 'No date'}
                    </small>
                  </article>
                ))}
              </div>
            ) : (
              <p className="empty-state">No documents indexed yet.</p>
            )}
          </section>
        </aside>

        <section className="research-column">
          <div className="query-panel">
            <div className="query-bar">
              <PanelTitle title="Question" detail={modeLabel(mode)} />
              <div className="query-controls">
                <div className="mode-toggle" role="group" aria-label="Research mode">
                  {MODES.map((option) => (
                    <button
                      key={option.id}
                      className={mode === option.id ? 'active' : ''}
                      onClick={() => setMode(option.id)}
                      disabled={isBusy}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
                <div className="scope-toggle" role="group" aria-label="Search scope">
                  {SEARCH_SCOPES.map((option) => (
                    <button
                      key={option.id}
                      className={searchScope === option.id ? 'active' : ''}
                      onClick={() => setSearchScope(option.id)}
                      disabled={isBusy}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="question-deck">
              <button
                className="question-cycle"
                onClick={() => setQuestion(rotatingQuestion)}
                disabled={isBusy}
                type="button"
              >
                <span>Suggested question</span>
                <strong>{rotatingQuestion}</strong>
              </button>
              <div className="deck-actions">
                <button className="icon-button" onClick={() => cycleQuestion(-1)} disabled={isBusy} type="button">
                  <ChevronLeft size={15} />
                </button>
                <button className="icon-button" onClick={() => cycleQuestion(1)} disabled={isBusy} type="button">
                  <ChevronRight size={15} />
                </button>
              </div>
            </div>

            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleQuestionKeyDown}
              placeholder="Ask a research question grounded in indexed filings..."
            />

            <div className="query-actions">
              <div className="sample-row">
                {questionSuggestions.slice(0, 4).map((sample) => (
                  <button key={sample} onClick={() => setQuestion(sample)} disabled={isBusy}>
                    {sample}
                  </button>
                ))}
              </div>
              <button className="primary-action submit-button" onClick={handleAsk} disabled={isBusy || !question.trim()}>
                <Send size={14} /> Submit
              </button>
            </div>
          </div>

          <section className="answer-panel">
            <PanelTitle title="Response" detail={responseMethod} detailClass="method-badge" />
            <pre>
              {chat?.answer ??
                'No response yet.'}
            </pre>
            {chat && (
              <div className="citation-strip" aria-label="Citations">
                {chat.citations.map((citation) => (
                  <button
                    key={citation.marker}
                    className={selectedCitation?.marker === citation.marker ? 'active' : ''}
                    onClick={() => setSelectedCitation(citation)}
                  >
                    <strong>{citation.marker}</strong>
                    <span>{citation.section ?? citation.ticker ?? 'Source'}</span>
                  </button>
                ))}
              </div>
            )}
          </section>

          <section className="evidence-panel">
            <PanelTitle title="Retrieved evidence" detail={`${chat?.retrieved_chunks.length ?? 0} chunks`} />
            <div className="evidence-list">
              {chat?.retrieved_chunks.length ? (
                chat.retrieved_chunks.map((chunk) => (
                  <article key={chunk.chunk_id} className="evidence-row">
                    <div>
                      <strong>
                        {chunk.metadata.ticker ?? 'DOC'} - {chunk.metadata.section ?? 'Unknown section'}
                      </strong>
                      <span className="row-meta">
                        {chunk.chunk_id} - {chunk.score.toFixed(3)}
                      </span>
                    </div>
                    <p>{clamp(chunk.text, EVIDENCE_PREVIEW_CHARS)}</p>
                  </article>
                ))
              ) : (
                <p className="empty-state">
                  Evidence will appear here after a question is submitted. Each chunk shows section,
                  retrieval score, and a preview. Click a citation marker for the full passage.
                </p>
              )}
            </div>
          </section>
        </section>

        <aside className="review-column">
          <section className="review-panel">
            <PanelTitle title="Citation" detail={selectedCitation?.marker ?? 'None'} />
            {selectedCitation ? (
              <>
                <dl>
                  <dt>Section</dt>
                  <dd>{selectedCitation.section ?? 'Unknown'}</dd>
                  <dt>Source</dt>
                  <dd>{selectedCitation.source}</dd>
                  <dt>Chunk</dt>
                  <dd>{selectedCitation.chunk_id}</dd>
                </dl>
                <p className="snippet">{selectedChunk?.text ?? selectedCitation.snippet}</p>
              </>
            ) : (
              <p className="empty-state">
                Select a citation marker above to inspect the source chunk and its metadata.
              </p>
            )}
          </section>

          <section className="review-panel">
            <PanelTitle title="Tools" detail={`${chat?.tool_calls.length ?? 0}`} />
            {chat?.tool_calls.length ? (
              <div className="tool-list">
                {chat.tool_calls.map((call, index) => (
                  <article key={`${call.name}-${index}`}>
                    <div>
                      <strong>{call.name}</strong>
                      <span className={`tool-status${call.ok ? '' : ' failed'}`}>
                        {call.ok ? 'ok' : 'failed'}
                      </span>
                    </div>
                    <code>{JSON.stringify(call.output ?? call.error)}</code>
                  </article>
                ))}
              </div>
            ) : (
              <p className="empty-state">
                Tool calls (calculator, ratios, market data, backtest) will appear here when the agent
                invokes them.
              </p>
            )}
          </section>

          <section className="review-panel">
            <PanelTitle title="Evaluation" detail={evalSummary ? 'Complete' : 'Not run'} />
            <StatLine
              label="Regression"
              value={evalSummary ? `${Math.round(evalSummary.regression_pass_rate * 100)}%` : 'Not run'}
              tone={evalSummary ? (evalSummary.regression_pass_rate >= 0.8 ? 'pass' : 'warn') : ''}
            />
            <StatLine label="Citation check" value={citationStatus} tone={citationStatusClass} />
            <StatLine label="API cost estimate" value={costDisplay} />
            {evalSummary?.mode_results.map((result) => (
              <div className="mode-result" key={result.mode}>
                <strong>{result.mode}</strong>
                <span>precision {Math.round(result.retrieval_precision * 100)}%</span>
                <span>recall@5 {Math.round(result.retrieval_recall_at_5 * 100)}%</span>
                <span>mrr {result.retrieval_mrr.toFixed(2)}</span>
                <span>citations {Math.round(result.citation_correctness * 100)}%</span>
                <span>{result.avg_latency_ms.toFixed(1)} ms</span>
                <span>${result.estimated_cost_usd.toFixed(5)}</span>
              </div>
            ))}
            {raftSummary && (
              <div className="mode-result">
                <strong>RAFT / LoRA</strong>
                <span>{raftSummary.raft_examples} examples</span>
                <span>{raftSummary.lora_records} records</span>
                <span>{String(raftSummary.training_report.status)}</span>
              </div>
            )}
          </section>

          <p className="disclaimer">{chat?.disclaimer ?? 'Research analysis only; not financial advice.'}</p>
        </aside>
      </section>
    </main>
  )
}

function PanelTitle({
  title,
  detail,
  detailClass
}: {
  title: string
  detail?: string
  detailClass?: string
}) {
  return (
    <div className="panel-title">
      <h2>{title}</h2>
      {detail && <span className={detailClass}>{detail}</span>}
    </div>
  )
}

function Metric({
  icon,
  label,
  value,
  sub
}: {
  icon: React.ReactNode
  label: string
  value: string | number
  sub?: string
}) {
  return (
    <div className="metric">
      <div className="metric-head">
        {icon}
        <span>{label}</span>
      </div>
      <strong>{value}</strong>
      {sub && <small>{sub}</small>}
    </div>
  )
}

function TextField({
  label,
  value,
  onChange
}: {
  label: string
  value: string
  onChange: (value: string) => void
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  )
}

function StatLine({ label, value, tone = '' }: { label: string; value: string; tone?: string }) {
  return (
    <div className={`stat-line${tone ? ` ${tone}` : ''}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function modeLabel(mode: string) {
  return MODES.find((option) => option.id === mode)?.label ?? mode
}

function retrievalMethod(mode: string): string {
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

function buildQuestionSuggestions(company: string, ticker: string): string[] {
  const label = company.trim() || ticker.trim() || 'the company'
  const tickerLabel = ticker.trim() || label
  return [
    `What are the main risk factors ${label} discloses?`,
    `Summarize ${label}'s revenue and margin drivers with citations.`,
    `What does ${label} disclose about liquidity, debt, and capital allocation?`,
    `Extract notable events, accounting changes, or operational updates for ${tickerLabel}.`,
    `Generate a cited research thesis for ${label}, separating facts from inference.`,
    `What evidence is available for demand, supply, or customer concentration risk at ${label}?`
  ]
}

function isSecFilingCandidate(url: string): boolean {
  return (
    url.startsWith('https://www.sec.gov/') &&
    (url.includes('/Archives/edgar/data/') || url.includes('/ix?doc='))
  )
}

function clamp(text: string, max: number): string {
  if (text.length <= max) return text
  return `${text.slice(0, max).trimEnd()}...`
}

function formatClock(): string {
  const now = new Date()
  const hh = String(now.getHours()).padStart(2, '0')
  const mm = String(now.getMinutes()).padStart(2, '0')
  return `${hh}:${mm}`
}

function formatCost(summary: EvalSummary | null): string {
  if (!summary || summary.mode_results.length === 0) return 'Not estimated'
  const total = summary.mode_results.reduce((acc, m) => acc + m.estimated_cost_usd, 0)
  const avg = total / summary.mode_results.length
  if (avg <= 0) return 'Local'
  return `est $${avg.toFixed(5)}`
}
