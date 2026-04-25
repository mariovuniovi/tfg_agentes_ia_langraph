'use client'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import type { MetricSeries } from '@/types/api'

const STROKE_DASH: Record<string, string> = { solid: '0', dashed: '6 3', dotted: '2 2' }
const COLORS = ['#1e3a5f', '#D97706', '#16a34a', '#7c3aed']

export function TrainerLineChart({ series }: { series: MetricSeries[] }) {
  if (!series.length) return <p className="text-xs text-slate-400">No training metrics</p>

  const maxLen = Math.max(...series.map((s) => s.steps.length))
  const data = Array.from({ length: maxLen }, (_, i) => {
    const point: Record<string, number> = { step: i }
    series.forEach((s) => { if (s.steps[i] !== undefined) point[s.name] = s.values[i] })
    return point
  })

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="step" tick={{ fontSize: 11 }} />
        <YAxis tick={{ fontSize: 11 }} />
        <Tooltip />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {series.map((s, i) => (
          <Line
            key={s.name}
            type="monotone"
            dataKey={s.name}
            stroke={COLORS[i % COLORS.length]}
            strokeDasharray={STROKE_DASH[s.line_style]}
            dot={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
