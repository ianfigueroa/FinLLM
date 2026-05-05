import { BarChart3, Clock3, Database, FileSearch, Play, Send } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { getIngestionStatus, ingestSample, runEval, sendChat } from './api'
import type { ChatResponse, Citation, EvalSummary } from './types'

const SAMPLE_QUESTIONS = [
  'What risk factors did Acme disclose?',
  'Why did Acme margins expand?',
  'Generate a cited research thesis for Acme.'
]

export function App() {
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[0])
  const [mode, setMode] = useState('rag_rerank')
  const [status, setStatus] = useState({ chunks_indexed: 0 })
  const [chat, setChat] = useState<ChatResponse | null>(null)
  const [evalSummary, setEvalSummary] = useState<EvalSummary | null>(null)
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null)
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

  async function handleIngest() {
    await runAction('ingest', async () => {
      const result = await ingestSample()
      setStatus({ chunks_indexed: result.chunks_indexed })
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
        <div>
          <p className="eyebrow">FinLLM Research Agent</p>
          <h1>Cited financial research console</h1>
        </div>
        <div className="actions">
          <button title="Ingest sample filing" onClick={handleIngest} disabled={loading === 'ingest'}>
            <Database size={17} /> Ingest
          </button>
          <button title="Run evaluation" onClick={handleEval} disabled={loading === 'eval'}>
            <BarChart3 size={17} /> Eval
          </button>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="metrics-strip">
        <Metric icon={<Database size={18} />} label="Indexed chunks" value={status.chunks_indexed} />
        <Metric icon={<Clock3 size={18} />} label="Last latency" value={`${latencyMs} ms`} />
        <Metric icon={<FileSearch size={18} />} label="Mode" value={mode} />
        <Metric icon={<BarChart3 size={18} />} label="Est. cost" value="$0.00 local" />
      </section>

      <section className="layout">
        <div className="primary-panel">
          <div className="question-row">
            <select value={mode} onChange={(event) => setMode(event.target.value)}>
              <option value="basic_rag">Basic RAG</option>
              <option value="rag_rerank">RAG + reranker</option>
              <option value="self_verify">Self verification</option>
            </select>
            <button title="Ask question" onClick={handleAsk} disabled={loading === 'chat' || !question.trim()}>
              <Send size={17} /> Ask
            </button>
          </div>

          <textarea value={question} onChange={(event) => setQuestion(event.target.value)} />

          <div className="sample-row">
            {SAMPLE_QUESTIONS.map((sample) => (
              <button key={sample} onClick={() => setQuestion(sample)}>
                <Play size={14} /> {sample}
              </button>
            ))}
          </div>

          <section className="answer">
            <h2>Answer</h2>
            <pre>{chat?.answer ?? 'Ingest the sample filing, then ask a research question.'}</pre>
            {chat && (
              <div className="citation-row">
                {chat.citations.map((citation) => (
                  <button
                    key={citation.marker}
                    className={selectedCitation?.marker === citation.marker ? 'active' : ''}
                    onClick={() => setSelectedCitation(citation)}
                  >
                    {citation.marker}
                  </button>
                ))}
              </div>
            )}
          </section>

          <section className="chunks">
            <h2>Retrieved chunks</h2>
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
          <h2>Source inspector</h2>
          {selectedCitation ? (
            <>
              <p className="marker">{selectedCitation.marker}</p>
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

          <h2>Evaluation</h2>
          <div className="eval-box">
            <span>Regression pass rate</span>
            <strong>{evalSummary ? `${Math.round(evalSummary.regression_pass_rate * 100)}%` : 'Not run'}</strong>
          </div>
          <div className="eval-box">
            <span>Citation check</span>
            <strong>{chat ? (chat.verification.passed ? 'Pass' : 'Review') : 'Waiting'}</strong>
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
