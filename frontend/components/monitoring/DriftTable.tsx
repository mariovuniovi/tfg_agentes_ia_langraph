import type { ColumnDriftResult } from '@/types/api'

export function DriftTable({ columns }: { columns: ColumnDriftResult[] }) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="border-b border-slate-200 text-left text-slate-500">
          <th className="pb-1 pr-4 font-medium">Column</th>
          <th className="pb-1 pr-4 font-medium">Score</th>
          <th className="pb-1 pr-4 font-medium">Method</th>
          <th className="pb-1 font-medium">Drift</th>
        </tr>
      </thead>
      <tbody>
        {columns.map((col) => (
          <tr key={col.column} className="border-b border-slate-100">
            <td className="py-1 pr-4 font-mono">{col.column}</td>
            <td className="py-1 pr-4">{col.score.toFixed(3)}</td>
            <td className="py-1 pr-4 text-slate-500">{col.method}</td>
            <td className="py-1">
              {col.drift_detected
                ? <span className="text-red-500">✗</span>
                : <span className="text-green-600">✓</span>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
