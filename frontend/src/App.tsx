import {
  BarChart3,
  CheckCircle2,
  Clock3,
  Database,
  FileSearch,
  Link2,
  Play,
  Send,
  ShieldCheck,
  Upload
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import {
  getIngestionStatus,
  ingestSample,
  ingestSecUrl,
  runEval,
  runRaftExperiment,
  sendChat,
  uploadDocument
} from './api'
import type { ChatResponse, Citation, EvalSummary, RaftExperimentResult } from './types'

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

const LOADING_LABELS: Record<string, string> = {
  ingest: 'Indexing',
  'sec-url': 'Indexing SEC',
  upload: 'Uploading',
  chat: 'Researching',
  eval: 'Evaluating',
  raft: 'Generating'
}

export function App() {
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[0])
  const [mode, setMode] = useState('rag_rerank')
  const [status, setStatus] = useState({ chunks_indexed: 0 })
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
  const [error, setError] = useState('')
  const [latencyMs, setLatencyMs] = useState(0)

  useEffect(() => {
    getIngestionStatus().then(setStatus).catch(() => setStatus({ chunks_indexed: 0 }))
  }, [])

  const selectedChunk = useMemo(() => {
    if (!chat || !selectedCitation) return null
    return chat.retrieved_chunks.find((chunk) => chunk.chunk_id === selectedCitation.chunk_id) ?? null
  }, [chat, selectedCitation])

  const isBusy = Boolean(loading)
  const canIndexSec = Boolean(
    secForm.url.trim() &&
      secForm.ticker.trim() &&
      secForm.company.trim() &&
      secForm.form_type.trim() &&
      secForm.filing_date.trim()
  )
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

  async function handleIngest() {
    await runAction('ingest', async () => {
      const result = await ingestSample()
      setStatus(await getIngestionStatus())
      setLastSource(`ACME sample · ${result.chunks_indexed} chunks`)
    })
  }

  async function handleSecUrlIngest() {
    await runAction('sec-url', async () => {
      const result = await ingestSecUrl(secForm)
      setStatus(await getIngestionStatus())
      setLastSource(`${result.ticker ?? secForm.ticker.toUpperCase()} SEC filing`)
    })
  }

  async function handleUpload() {
    if (!uploadFile) return
    await runAction('upload', async () => {
      const result = await uploadDocument({ ...uploadForm, file: uploadFile })
      setStatus(await getIngestionStatus())
      setLastSource(`${result.ticker ?? uploadForm.ticker.toUpperCase()} upload`)
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

  return (
    <main className="workspace">
      <header className="app-header">
        <div>
          <p className="eyebrow">FinLLM</p>
          <h1>Research Agent</h1>
        </div>
        <div className="header-actions">
          <StatusPill label={isBusy ? LOADING_LABELS[loading] : 'Ready'} />
          <button onClick={handleIngest} disabled={isBusy} title="Index ACME sample filing">
            <Database size={16} /> Index sample
          </button>
          <button onClick={handleEval} disabled={isBusy} title="Run local evals">
            <BarChart3 size={16} /> Run eval
          </button>
          <button onClick={handleRaft} disabled={isBusy} title="Generate RAFT examples">
            <FileSearch size={16} /> RAFT
          </button>
        </div>
      </header>

      {error && <div className="notice error">{error}</div>}

      <section className="summary-grid" aria-label="Research status">
        <Metric icon={<Database size={16} />} label="Corpus" value={`${status.chunks_indexed} chunks`} />
        <Metric icon={<Clock3 size={16} />} label="Latency" value={`${latencyMs} ms`} />
        <Metric icon={<ShieldCheck size={16} />} label="Citations" value={citationStatus} />
        <Metric icon={<BarChart3 size={16} />} label="Confidence" value={chat ? `${Math.round(chat.confidence * 100)}%` : '0%'} />
      </section>

      <section className="app-grid">
        <aside className="source-column">
          <PanelTitle title="Sources" detail={lastSource} />
          <div className="source-block">
            <label className="field full">
              <span>SEC URL</span>
              <input
                value={secForm.url}
                onChange={(event) => setSecForm({ ...secForm, url: event.target.value })}
              />
            </label>
            <div className="field-row compact">
              <TextField label="Ticker" value={secForm.ticker} onChange={(ticker) => setSecForm({ ...secForm, ticker })} />
              <TextField label="Form" value={secForm.form_type} onChange={(form_type) => setSecForm({ ...secForm, form_type })} />
            </div>
            <TextField label="Company" value={secForm.company} onChange={(company) => setSecForm({ ...secForm, company })} />
            <TextField label="Filing date" value={secForm.filing_date} onChange={(filing_date) => setSecForm({ ...secForm, filing_date })} />
            <button className="primary-action full-button" onClick={handleSecUrlIngest} disabled={isBusy || !canIndexSec}>
              <Link2 size={16} /> Index SEC filing
            </button>
          </div>

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
              <Upload size={16} /> Upload filing
            </button>
          </div>
        </aside>

        <section className="research-column">
          <div className="query-panel">
            <div className="query-bar">
              <PanelTitle title="Question" detail={modeLabel(mode)} />
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
            </div>

            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} />

            <div className="query-actions">
              <div className="sample-row">
                {SAMPLE_QUESTIONS.map((sample) => (
                  <button key={sample} onClick={() => setQuestion(sample)} disabled={isBusy}>
                    <Play size={13} /> {sample}
                  </button>
                ))}
              </div>
              <button className="primary-action submit-button" onClick={handleAsk} disabled={isBusy || !question.trim()}>
                <Send size={16} /> Submit
              </button>
            </div>
          </div>

          <section className="answer-panel">
            <PanelTitle title="Response" detail={chat?.mode ?? modeLabel(mode)} />
            <pre>{chat?.answer ?? 'No response yet.'}</pre>
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
              {chat?.retrieved_chunks.map((chunk) => (
                <article key={chunk.chunk_id} className="evidence-row">
                  <div>
                    <strong>{chunk.metadata.ticker ?? 'DOC'} · {chunk.metadata.section ?? 'Unknown section'}</strong>
                    <span>{chunk.chunk_id} · {chunk.score.toFixed(3)}</span>
                  </div>
                  <p>{chunk.text}</p>
                </article>
              )) ?? <p className="empty-state">No evidence retrieved.</p>}
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
              <p className="empty-state">Select a citation.</p>
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
                      <span>{call.ok ? 'ok' : 'failed'}</span>
                    </div>
                    <code>{JSON.stringify(call.output ?? call.error)}</code>
                  </article>
                ))}
              </div>
            ) : (
              <p className="empty-state">No tool calls.</p>
            )}
          </section>

          <section className="review-panel">
            <PanelTitle title="Evaluation" detail={evalSummary ? 'Complete' : 'Not run'} />
            <StatLine label="Regression" value={evalSummary ? `${Math.round(evalSummary.regression_pass_rate * 100)}%` : 'Not run'} />
            <StatLine label="Citation check" value={citationStatus} />
            <StatLine label="Cost" value="$0.00 local" />
            {evalSummary?.mode_results.map((result) => (
              <div className="mode-result" key={result.mode}>
                <strong>{result.mode}</strong>
                <span>retrieval {Math.round(result.retrieval_precision * 100)}%</span>
                <span>citations {Math.round(result.citation_correctness * 100)}%</span>
                <span>{result.avg_latency_ms.toFixed(1)} ms</span>
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

function PanelTitle({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="panel-title">
      <h2>{title}</h2>
      {detail && <span>{detail}</span>}
    </div>
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

function StatLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-line">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function StatusPill({ label }: { label: string }) {
  return (
    <span className="status-pill">
      <CheckCircle2 size={14} /> {label}
    </span>
  )
}

function modeLabel(mode: string) {
  return MODES.find((option) => option.id === mode)?.label ?? mode
}
