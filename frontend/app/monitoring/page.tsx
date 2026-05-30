'use client'
import { useState } from 'react'
import { LatestReport } from '@/components/monitoring/LatestReport'
import { AdHocForm } from '@/components/monitoring/AdHocForm'

const TABS = ['Latest Report', 'Ad-hoc Analysis'] as const
type Tab = typeof TABS[number]

export default function MonitoringPage() {
  const [tab, setTab] = useState<Tab>('Latest Report')
  return (
    <div className="max-w-3xl space-y-4">
      <div className="flex gap-1 border-b border-zinc-200">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              tab === t
                ? 'border-b-2 border-indigo-600 text-zinc-900'
                : 'text-zinc-500 hover:text-zinc-700'
            }`}
          >
            {t}
          </button>
        ))}
      </div>
      <div className="rounded-lg border border-zinc-200 bg-white p-4">
        {tab === 'Latest Report' ? <LatestReport /> : <AdHocForm />}
      </div>
    </div>
  )
}
