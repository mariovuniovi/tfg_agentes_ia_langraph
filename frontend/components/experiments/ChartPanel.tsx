import type { RunOut } from '@/types/api'
import { Card } from '@/components/ui/Card'
import { TrainerLineChart } from './charts/TrainerLineChart'
import { EvaluatorRadarChart } from './charts/EvaluatorRadarChart'
import { EvaluatorBarChart } from './charts/EvaluatorBarChart'
import { DeploymentBarChart } from './charts/DeploymentBarChart'

export function ChartPanel({ run }: { run: RunOut | null }) {
  if (!run) {
    return (
      <Card title="Metrics" className="h-full">
        <div className="flex h-full items-center justify-center text-zinc-400">
          Select a run to view charts
        </div>
      </Card>
    )
  }
  return (
    <Card title="Metrics" className="overflow-y-auto">
      <div className="space-y-6">
        <section>
          <h3 className="mb-2 text-sm font-semibold text-zinc-900">Trainer — Loss & Accuracy</h3>
          <TrainerLineChart series={run.metric_series} />
        </section>
        <section>
          <h3 className="mb-2 text-sm font-semibold text-zinc-900">Evaluator Metrics</h3>
          <div className="grid grid-cols-2 gap-4">
            <EvaluatorRadarChart metrics={run.metrics} />
            <EvaluatorBarChart metrics={run.metrics} />
          </div>
        </section>
        <section>
          <h3 className="mb-2 text-sm font-semibold text-zinc-900">Deployment Comparison</h3>
          <DeploymentBarChart metrics={run.metrics} />
        </section>
      </div>
    </Card>
  )
}
