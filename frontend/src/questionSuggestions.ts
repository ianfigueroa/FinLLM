export function buildQuestionSuggestions(company: string, ticker: string): string[] {
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

export function isSecFilingCandidate(url: string): boolean {
  return (
    url.startsWith('https://www.sec.gov/') &&
    (url.includes('/Archives/edgar/data/') || url.includes('/ix?doc='))
  )
}
