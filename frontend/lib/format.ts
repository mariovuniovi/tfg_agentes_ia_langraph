export function formatBytes(bytes: number): string {
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(0)} KB`
}

export function formatK(n: number): string {
  return n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n)
}

export function formatCost(usd: number | null | undefined): string {
  if (usd === null || usd === undefined) return 'Unknown'
  if (usd === 0) return '$0.00000'
  if (usd < 0.01) return '$' + usd.toFixed(5)
  return '$' + usd.toFixed(4)
}

export function formatMetricValue(v: number): string {
  if (!Number.isFinite(v)) return '—'
  return Math.abs(v) >= 1 ? v.toFixed(3) : v.toFixed(4)
}

export function formatRunTime(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  const pad = (n: number) => String(n).padStart(2, '0')
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}`
  )
}

export function buildRunCsv(
  metrics: Record<string, number>,
  params: Record<string, string>,
): string {
  const cell = (v: string) => `"${v.replace(/"/g, '""')}"`
  const rows = ['"type","key","value"']
  for (const [k, v] of Object.entries(metrics).sort(([a], [b]) => a.localeCompare(b))) {
    rows.push([cell('metric'), cell(k), cell(String(v))].join(','))
  }
  for (const [k, v] of Object.entries(params).sort(([a], [b]) => a.localeCompare(b))) {
    rows.push([cell('param'), cell(k), cell(v)].join(','))
  }
  return rows.join('\n')
}
