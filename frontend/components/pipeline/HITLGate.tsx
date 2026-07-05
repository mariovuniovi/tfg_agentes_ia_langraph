'use client'
import { useRunStore } from '@/stores/run-store'
import { useApprove } from '@/hooks/use-approve'
import { DeploymentApprovalCard } from '@/components/pipeline/DeploymentApprovalCard'
import type { DeployerInterrupt } from '@/types/api'

export function HITLGate({ runId }: { runId: string | null }) {
  const hitlPending = useRunStore((s) => s.hitlPending)
  const interruptValue = useRunStore((s) => s.interruptValue)
  const { approve, isPending } = useApprove(runId)

  if (!hitlPending) return null
  if ((interruptValue as { type?: string })?.type !== 'deployer') return null

  return (
    <DeploymentApprovalCard
      runId={runId}
      interrupt={interruptValue as unknown as DeployerInterrupt}
      onApprove={(decision) => approve(decision)}
      isPending={isPending}
    />
  )
}
