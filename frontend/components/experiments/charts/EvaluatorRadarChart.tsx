'use client'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from 'recharts'

export function EvaluatorRadarChart({ metrics }: { metrics: Record<string, number> }) {
  const data = Object.entries(metrics).map(([metric, value]) => ({ metric, value }))
  if (!data.length) return null
  return (
    <ResponsiveContainer width="100%" height={200}>
      <RadarChart data={data}>
        <PolarGrid />
        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10 }} />
        <Radar dataKey="value" stroke="#1e3a5f" fill="#1e3a5f" fillOpacity={0.3} />
        <Tooltip />
      </RadarChart>
    </ResponsiveContainer>
  )
}
