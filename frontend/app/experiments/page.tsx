'use client'
import { useState } from 'react'
import { RunSidebar } from '@/components/experiments/RunSidebar'
import { ChartPanel } from '@/components/experiments/ChartPanel'
import type { RunOut } from '@/types/api'

export default function ExperimentsPage() {
  const [selectedRun, setSelectedRun] = useState<RunOut | null>(null)
  return (
    <div className="flex h-[calc(100vh-80px)] gap-4">
      <div className="w-64 shrink-0 overflow-hidden">
        <RunSidebar selectedRunId={selectedRun?.run_id ?? null} onSelectRun={setSelectedRun} />
      </div>
      <div className="flex-1 overflow-hidden rounded-lg border border-slate-200 bg-white p-4">
        <ChartPanel run={selectedRun} />
      </div>
    </div>
  )
}
