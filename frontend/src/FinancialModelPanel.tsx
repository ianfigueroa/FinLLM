import { Download, LineChart, Table2 } from 'lucide-react'
import { useMemo, useState } from 'react'

import { buildThreeStatementModel } from './api'
import type { StatementLine, ThreeStatementModelResult } from './types'

interface FinancialModelPanelProps {
  ticker: string
  company: string
}

export function FinancialModelPanel({ ticker, company }: FinancialModelPanelProps) {
  const [projectionYears, setProjectionYears] = useState(3)
  const [revenueGrowth, setRevenueGrowth] = useState(5)
  const [model, setModel] = useState<ThreeStatementModelResult | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const years = useMemo(() => modelYears(model), [model])
  const hasModel = Boolean(model)

  async function handleBuildModel() {
    if (!ticker.trim()) return
    setError('')
    setLoading(true)
    try {
      setModel(
        await buildThreeStatementModel({
          ticker: ticker.trim().toUpperCase(),
          projection_years: projectionYears,
          revenue_growth: revenueGrowth / 100
        })
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Model build failed')
    } finally {
      setLoading(false)
    }
  }

  function handleDownloadCsv() {
    if (!model) return
    const csv = modelToCsv(model)
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${model.ticker.toLowerCase()}-three-statement-model.csv`
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.URL.revokeObjectURL(url)
  }

  return (
    <section className="model-panel">
      <div className="model-header">
        <div>
          <h2>Financial model</h2>
          <p>{company || ticker || 'Select a ticker'} 3-statement scaffold from indexed evidence.</p>
        </div>
        <div className="model-actions">
          <label>
            Years
            <input
              min={1}
              max={5}
              type="number"
              value={projectionYears}
              onChange={(event) => setProjectionYears(Number(event.target.value))}
            />
          </label>
          <label>
            Growth %
            <input
              max={50}
              min={-50}
              step={0.5}
              type="number"
              value={revenueGrowth}
              onChange={(event) => setRevenueGrowth(Number(event.target.value))}
            />
          </label>
          <button onClick={handleBuildModel} disabled={loading || !ticker.trim()}>
            <LineChart size={14} /> {loading ? 'Building' : 'Build model'}
          </button>
          <button onClick={handleDownloadCsv} disabled={!hasModel}>
            <Download size={14} /> CSV
          </button>
        </div>
      </div>

      {error && <div className="model-error">{error}</div>}

      {model ? (
        <div className="model-body">
          <div className="model-kpis">
            <Metric label="Confidence" value={`${Math.round(model.confidence * 100)}%`} />
            <Metric label="Sources" value={String(model.sources.length)} />
            <Metric
              label="Revenue growth"
              value={formatPercent(model.assumptions.revenue_growth)}
            />
          </div>

          <div className="statement-stack">
            {Object.entries(model.statements).map(([statementKey, statement]) => (
              <div className="statement-card" key={statementKey}>
                <div className="statement-title">
                  <strong>{statement.label}</strong>
                  <span>{Object.keys(statement.lines).length} lines</span>
                </div>
                <div className="statement-table-wrap">
                  <table className="statement-table">
                    <thead>
                      <tr>
                        <th>Line item</th>
                        {years.map((year) => (
                          <th key={year}>{year}</th>
                        ))}
                        <th>Source</th>
                      </tr>
                    </thead>
                    <tbody>
                      {statementRows(statement.lines, statementKey, model).map((row) => (
                        <tr key={row.key}>
                          <td>{row.label}</td>
                          {years.map((year) => (
                            <td key={year} className={row.isProjection(year) ? 'projection' : ''}>
                              {formatNumber(row.value(year))}
                            </td>
                          ))}
                          <td>{row.source}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>

          <div className="model-footnotes">
            <div>
              <strong>Model sources</strong>
              {model.sources.slice(0, 4).map((source) => (
                <p key={`${source.marker}-${source.chunk_id}`}>
                  <span>{source.marker}</span> {source.section ?? source.chunk_id}: {source.excerpt}
                </p>
              ))}
            </div>
            <div>
              <strong>Limitations</strong>
              {model.limitations.map((limitation) => (
                <p key={limitation}>{limitation}</p>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="model-empty">
          <Table2 size={18} />
          Build a model after indexing a filing. The app extracts supported financial lines,
          projects the next years, and keeps source chunks attached.
        </div>
      )}
    </section>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function modelYears(model: ThreeStatementModelResult | null): string[] {
  if (!model) return []
  const years = new Set<string>()
  Object.values(model.statements).forEach((statement) => {
    Object.values(statement.lines).forEach((line) => {
      Object.keys(line.historical).forEach((year) => years.add(year))
    })
  })
  Object.keys(model.projections).forEach((year) => years.add(year))
  return [...years].sort()
}

function statementRows(
  lines: Record<string, StatementLine>,
  statementKey: string,
  model: ThreeStatementModelResult
) {
  return Object.entries(lines).map(([key, line]) => ({
    key,
    label: line.label,
    source: line.sources[0]?.marker ?? 'derived',
    value: (year: string) => line.historical[year] ?? model.projections[year]?.[statementKey]?.[key],
    isProjection: (year: string) => Boolean(model.projections[year])
  }))
}

function modelToCsv(model: ThreeStatementModelResult): string {
  const years = modelYears(model)
  const rows = [['Statement', 'Line item', ...years, 'Source']]
  Object.entries(model.statements).forEach(([statementKey, statement]) => {
    statementRows(statement.lines, statementKey, model).forEach((row) => {
      rows.push([
        statement.label,
        row.label,
        ...years.map((year) => String(row.value(year) ?? '')),
        row.source
      ])
    })
  })
  return rows.map((row) => row.map(csvCell).join(',')).join('\n')
}

function csvCell(value: string): string {
  return `"${value.replace(/"/g, '""')}"`
}

function formatNumber(value: number | undefined): string {
  if (value === undefined) return '-'
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 })
}

function formatPercent(value: number | undefined): string {
  if (value === undefined) return '-'
  return `${(value * 100).toFixed(1)}%`
}
