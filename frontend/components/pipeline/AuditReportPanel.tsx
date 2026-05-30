'use client'
import { useState } from 'react'
import type { AuditReportEventData } from '@/types/api'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'

function Section({ title, defaultOpen, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  return (
    <details open={defaultOpen} className="border-t border-[var(--color-border)] py-2 [&_summary]:cursor-pointer">
      <summary className="text-xs font-semibold text-[var(--color-fg)]">{title}</summary>
      <div className="mt-1 text-xs text-zinc-600">{children}</div>
    </details>
  )
}

export function AuditReportPanel({ data }: { data: AuditReportEventData }) {
  const [showJson, setShowJson] = useState(false)
  const a = data.audit
  return (
    <Card
      title="Audit Report"
      actions={data.evaluation_passed
        ? <Badge variant="success">eligible</Badge>
        : <Badge variant="info">candidate rejected</Badge>}
    >
      <dl className="mb-3 grid grid-cols-[160px_1fr] gap-y-1 text-xs">
        <dt className="text-zinc-500">Champion model</dt>
        <dd className="font-mono text-zinc-800">{data.champion_model}</dd>
        <dt className="text-zinc-500">Deterministic eval</dt>
        <dd>{data.evaluation_passed
          ? <span className="text-emerald-700">✓ passed</span>
          : <span className="text-sky-700">✗ not passed</span>}
        </dd>
      </dl>

      {a.summary && <Section title="Summary">{a.summary}</Section>}
      {a.why_champion_won && <Section title="Why this model won">{a.why_champion_won}</Section>}
      {a.planner_alignment && <Section title="Planner alignment">{a.planner_alignment}</Section>}
      {(a.deviations_from_planner_expectations?.length ?? 0) > 0 && (
        <Section title="Deviations from planner expectations">
          <ul className="list-disc pl-4">{a.deviations_from_planner_expectations!.map((d, i) => <li key={i}>{d}</li>)}</ul>
        </Section>
      )}
      {(a.evidence_consistency_warnings?.length ?? 0) > 0 && (
        <Section title="Evidence consistency warnings">
          <ul className="list-disc pl-4">{a.evidence_consistency_warnings!.map((w, i) => <li key={i}>{w}</li>)}</ul>
        </Section>
      )}
      {(a.risks_and_warnings?.length ?? 0) > 0 && (
        <Section title="Risks & warnings" defaultOpen>
          <ul className="space-y-1">{a.risks_and_warnings!.map((r, i) => (
            <li key={i} className="rounded bg-amber-50 px-2 py-1 text-amber-700">⚠ {r}</li>
          ))}</ul>
        </Section>
      )}
      {(a.human_review_notes?.length ?? 0) > 0 && (
        <Section title="Human review notes">
          <ul className="list-disc pl-4">{a.human_review_notes!.map((n, i) => <li key={i}>{n}</li>)}</ul>
        </Section>
      )}

      <div className="mt-3 border-t border-[var(--color-border)] pt-2">
        <button
          type="button"
          onClick={() => setShowJson(v => !v)}
          className="cursor-pointer text-xs text-zinc-500 hover:text-zinc-700"
        >
          {showJson ? 'Hide' : 'View full audit JSON'}
        </button>
        {showJson && (
          <pre className="mt-1 overflow-x-auto rounded bg-zinc-50 p-2 font-mono text-xs text-zinc-700">
            {JSON.stringify(data, null, 2)}
          </pre>
        )}
      </div>
    </Card>
  )
}
