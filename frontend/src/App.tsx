import {
  BarChart3,
  CheckCircle2,
  Clock3,
  Database,
  Link2,
  Play,
  Send,
  ShieldCheck
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { getIngestionStatus, ingestSample, ingestSecUrl, runEval, sendChat } from './api'
import type { ChatResponse, Citation, EvalSummary } from './types'

const SAMPLE_QUESTIONS = [
  'What risk factors did Acme disclose?',
  'Why did Acme margins expand?',
  'Generate a cited research thesis for Acme.'
]

const MODES = [
  { id: 'basic_rag', label: 'Basic' },
  { id: 'rag_rerank', label: 'Rerank' },
  { id: 'self_verify', label: 'Verify' }
]

export function App() {
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[0])
  const [mode, setMode] = useState('rag_rerank')
  const [status, setStatus] = useState({ chunks_indexed: 0 })
  const [chat, setChat] = useState<ChatResponse | null>(null)
  const [evalSummary, setEvalSummary] = useState<EvalSummary | null>(null)
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null)
  const [secForm, setSecForm] = useState({
    url: 'https://www.sec.gov/ix?doc=/Archives/edgar/data/0001045810/000104581026000021/nvda-20260125.htm',
    ticker: 'NVDA',
    company: 'NVIDIA',
    form_type: '10-K',
    filing_date: '2026-01-25'
  })
  const [ingestedSource, setIngestedSource] = useState('')
  const [loading, setLoading] = useState('')
  const [error, setError] = useState('')
  const [latencyMs, setLatencyMs] = useState(0)

  useEffect(() => {
    getIngestionStatus().then(setStatus).catch(() => setStatus({ chunks_indexed: 0 }))
  }, [])

  const selectedChunk = useMemo(() => {
    if (!chat || !selectedCitation) return null
    return chat.retrieved_chunks.find((chunk) => chunk.chunk_id === selectedCitation.chunk_id) ?? null
  }, [chat, selectedCitation])

  const verificationLabel = chat
    ? chat.verification.passed
      ? 'Verified'
      : 'Review'
    : 'Pending'

  const confidence = chat ? `${Math.round(chat.confidence * 100)}%` : '0%'

  async function handleIngest() {
    await runAction('ingest', async () => {
      const result = await ingestSample()
      setStatus({ chunks_indexed: result.chunks_indexed })
      setIngestedSource('ACME sample indexed')
    })
  }

  async function handleSecUrlIngest() {
    await runAction('sec-url', async () => {
      const result = await ingestSecUrl(secForm)
      setStatus(await getIngestionStatus())
      setIngestedSource(`${result.ticker ?? secForm.ticker.toUpperCase()} indexed`)
    })
  }

  async function handleAsk() {
    await runAction('chat', async () => {
      const started = performance.now()
      const result = await sendChat(question, mode)
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

  return (
    <main className="workspace">
      <header className="topbar">
        <div className="brand-block">
          <p className="eyebrow">FinLLM</p>
          <h1>Research Workbench</h1>
          <span>Grounded answers over filings, with inspectable evidence.</span>
        </div>
        <div className="actions">
          <button
            className="secondary-action"
            title="Index sample filing"
            onClick={handleIngest}
            disabled={loading === 'ingest'}
          >
            <Database size={16} /> Index sample
          </button>
          <button
            className="secondary-action"
            title="Run evaluation"
            onClick={handleEval}
            disabled={loading === 'eval'}
          >
            <BarChart3 size={16} /> Run eval
          </button>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="metrics-strip">
        <Metric icon={<Database size={17} />} label="Indexed" value={status.chunks_indexed} />
        <Metric icon={<Clock3 size={17} />} label="Latency" value={`${latencyMs} ms`} />
        <Metric icon={<ShieldCheck size={17} />} label="Citation check" value={verificationLabel} />
        <Metric icon={<BarChart3 size={17} />} label="Confidence" value={confidence} />
      </section>

      <section className="ingest-panel">
        <div className="ingest-heading">
          <div>
            <h2>Document ingestion</h2>
            <p>SEC archive source and filing metadata.</p>
          </div>
          <span>{ingestedSource || 'SEC URLs only'}</span>
        </div>
        <div className="ingest-grid">
          <label className="url-field">
            <span>SEC filing URL</span>
            <input
              value={secForm.url}
              onChange={(event) => setSecForm({ ...secForm, url: event.target.value })}
            />
          </label>
          <label>
            <span>Ticker</span>
            <input
              value={secForm.ticker}
              onChange={(event) => setSecForm({ ...secForm, ticker: event.target.value })}
            />
          </label>
          <label>
            <span>Company</span>
            <input
              value={secForm.company}
              onChange={(event) => setSecForm({ ...secForm, company: event.target.value })}
            />
          </label>
          <label>
            <span>Form</span>
            <input
              value={secForm.form_type}
              onChange={(event) => setSecForm({ ...secForm, form_type: event.target.value })}
            />
          </label>
          <label>
            <span>Filing date</span>
            <input
              value={secForm.filing_date}
              onChange={(event) => setSecForm({ ...secForm, filing_date: event.target.value })}
            />
          </label>
          <button
            className="primary-action ingest-submit"
            title="Index SEC filing URL"
            onClick={handleSecUrlIngest}
            disabled={loading === 'sec-url' || !secForm.url.trim()}
          >
            <Link2 size={16} /> Index SEC URL
          </button>
        </div>
      </section>

      <section className="layout">
        <div className="primary-panel">
          <div className="query-header">
            <div>
              <h2>Query</h2>
              <p>Mode controls retrieval and verification depth.</p>
            </div>
            <div className="mode-toggle" role="group" aria-label="Research mode">
              {MODES.map((option) => (
                <button
                  key={option.id}
                  className={mode === option.id ? 'active' : ''}
                  onClick={() => setMode(option.id)}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <textarea value={question} onChange={(event) => setQuestion(event.target.value)} />

          <div className="query-footer">
            <div className="sample-row">
              {SAMPLE_QUESTIONS.map((sample) => (
                <button key={sample} onClick={() => setQuestion(sample)}>
                  <Play size={13} /> {sample}
                </button>
              ))}
            </div>
            <button
              className="primary-action"
              title="Submit research query"
              onClick={handleAsk}
              disabled={loading === 'chat' || !question.trim()}
            >
              <Send size={16} /> Submit
            </button>
          </div>

          <section className="answer">
            <div className="section-heading">
              <h2>Response</h2>
              <span>{chat?.mode ?? mode}</span>
            </div>
            <pre>{chat?.answer ?? 'Index the sample filing, then submit a research question.'}</pre>
            {chat && (
              <div className="citation-row">
                {chat.citations.map((citation) => (
                  <button
                    key={citation.marker}
                    className={selectedCitation?.marker === citation.marker ? 'active' : ''}
                    onClick={() => setSelectedCitation(citation)}
                  >
                    {citation.marker}
                    <span>{citation.section ?? citation.ticker ?? 'Source'}</span>
                  </button>
                ))}
              </div>
            )}
          </section>

          <section className="chunks">
            <div className="section-heading">
              <h2>Retrieved Evidence</h2>
              <span>{chat?.retrieved_chunks.length ?? 0} chunks</span>
            </div>
            {chat?.retrieved_chunks.map((chunk) => (
              <article key={chunk.chunk_id} className="chunk-row">
                <div>
                  <strong>{chunk.metadata.ticker} · {chunk.metadata.section ?? 'Unknown section'}</strong>
                  <span>{chunk.chunk_id} · score {chunk.score.toFixed(3)}</span>
                </div>
                <p>{chunk.text}</p>
              </article>
            ))}
          </section>
        </div>

        <aside className="source-panel">
          <div className="section-heading">
            <h2>Source</h2>
            <span>{selectedCitation?.marker ?? 'No citation'}</span>
          </div>
          {selectedCitation ? (
            <>
              <dl>
                <dt>Section</dt>
                <dd>{selectedCitation.section}</dd>
                <dt>Source</dt>
                <dd>{selectedCitation.source}</dd>
                <dt>Chunk</dt>
                <dd>{selectedCitation.chunk_id}</dd>
              </dl>
              <p className="snippet">{selectedChunk?.text ?? selectedCitation.snippet}</p>
            </>
          ) : (
              <p className="muted">Select a citation to inspect the source chunk.</p>
          )}

          <div className="section-heading compact">
            <h2>Evaluation</h2>
            <CheckCircle2 size={16} />
          </div>
          <div className="eval-box">
            <span>Regression pass rate</span>
            <strong>{evalSummary ? `${Math.round(evalSummary.regression_pass_rate * 100)}%` : 'Not run'}</strong>
          </div>
          <div className="eval-box">
            <span>Citation check</span>
            <strong>{chat ? (chat.verification.passed ? 'Pass' : 'Review') : 'Waiting'}</strong>
          </div>
          <div className="eval-box">
            <span>Estimated cost</span>
            <strong>$0.00 local</strong>
          </div>
          <p className="muted">{chat?.disclaimer ?? 'Research outputs are not financial advice.'}</p>
        </aside>
      </section>
    </main>
  )
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string | number }) {
  return (
    <div className="metric">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}
