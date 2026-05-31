'use client'
import { PipelineHealthCard } from '@/components/observability/PipelineHealthCard'
import { LlmActivityCard } from '@/components/observability/LlmActivityCard'
import { ToolUsageCard } from '@/components/observability/ToolUsageCard'
import { TokenCostCard } from '@/components/observability/TokenCostCard'

export default function ObservabilityPage() {
  return (
    <div className="space-y-3 p-3">
      <PipelineHealthCard />
      <LlmActivityCard />
      <TokenCostCard />
      <ToolUsageCard />
    </div>
  )
}
