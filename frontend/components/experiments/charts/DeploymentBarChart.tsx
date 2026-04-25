'use client'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, LabelList, ResponsiveContainer } from 'recharts'

function downloadCSV(metrics: Record<string, number>) {
  const csv = 'metric,value\n' + Object.entries(metrics).map(([k, v]) => `${k},${v}`).join('\n')
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
  const a = document.createElement('a')
  a.href = url
  a.download = 'deployment-metrics.csv'
  a.click()
  URL.revokeObjectURL(url)
}

export function DeploymentBarChart({ metrics }: { metrics: Record<string, number> }) {
  const data = Object.entries(metrics)
    .sort(([, a], [, b]) => b - a)
    .map(([name, value]) => ({ name, value }))
  if (!data.length) return null
  return (
    <div>
      <div className="mb-1 flex justify-end">
        <button
          onClick={() => downloadCSV(metrics)}
          className="rounded px-2 py-1 text-xs font-medium text-amber-600 hover:bg-amber-50"
        >
          Export CSV
        </button>
      </div>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={data} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis type="number" tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={100} />
          <Tooltip />
          <Bar dataKey="value" fill="#1e3a5f">
            <LabelList dataKey="value" position="right" style={{ fontSize: 10 }} formatter={(v: any) => v.toFixed(3)} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
