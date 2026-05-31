'use client'
import { NodeTypeBadge, type NodeType } from '@/components/ui/NodeTypeBadge'
import type { StageKey, StageStatus } from '@/lib/stage-derive'

const STAGES: Array<{ key: StageKey; label: string; type: NodeType }> = [
  { key: 'data_validation',  label: 'Data Validation',  type: 'agent' },
  { key: 'dataset_approval', label: 'Dataset Approval', type: 'hitl' },
  { key: 'model_planning',   label: 'Model Planning',   type: 'agent' },
  { key: 'training',         label: 'Training',         type: 'deterministic' },
  { key: 'evaluation',       label: 'Evaluation',       type: 'deterministic' },
  { key: 'audit_report',     label: 'Audit Report',     type: 'llm' },
  { key: 'deploy_approval',  label: 'Deploy Approval',  type: 'hitl' },
  { key: 'deploy',           label: 'Deploy',           type: 'deterministic' },
]

const ICON: Record<StageStatus, string> = {
  pending:       '·',
  running:       '◐',
  completed:     '✓',
  waiting_human: '⏱',
  failed:        '✗',
  skipped:       '—',
}

const COLOR: Record<StageStatus, string> = {
  pending:       'text-[var(--color-fg-subtle)]',
  running:       'text-[var(--color-accent)]',
  completed:     'text-[var(--color-success)]',
  waiting_human: 'text-[var(--color-warning)]',
  failed:        'text-[var(--color-danger)]',
  skipped:       'text-[var(--color-fg-subtle)] opacity-60',
}

export function PipelineStepper({
  stages,
}: {
  stages: Record<StageKey, StageStatus>
}) {
  return (
    <ol className="flex flex-wrap items-stretch gap-2 rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] p-2">
      {STAGES.map(({ key, label, type }) => {
        const status = stages[key]
        return (
          <li
            key={key}
            data-testid={`stage-${key}`}
            className={`flex min-w-[140px] flex-1 flex-col gap-1 rounded border border-[var(--color-border)] px-3 py-2 text-xs ${COLOR[status]}`}
          >
            <div className="flex items-center gap-1.5">
              <span className="text-base leading-none">{ICON[status]}</span>
              <span className="font-medium text-[var(--color-fg)]">{label}</span>
            </div>
            <NodeTypeBadge type={type} />
          </li>
        )
      })}
    </ol>
  )
}
