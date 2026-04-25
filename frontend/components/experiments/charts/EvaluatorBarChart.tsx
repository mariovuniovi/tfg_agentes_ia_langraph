'use client'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, LabelList, ResponsiveContainer } from 'recharts'

export function EvaluatorBarChart({ metrics }: { metrics: Record<string, number> }) {
  const data = Object.entries(metrics).map(([name, value]) => ({ name, value }))
  if (!data.length) return null
  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={data} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10 }} />
        <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={80} />
        <Tooltip />
        <Bar dataKey="value" fill="#D97706">
          <LabelList dataKey="value" position="right" style={{ fontSize: 10 }} formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v ?? ''))} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
