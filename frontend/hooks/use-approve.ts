import { useMutation } from '@tanstack/react-query'
import { approveRun } from '@/lib/api'
import { useRunStore } from '@/stores/run-store'

export function useApprove(runId: string | null) {
  const clearHITL = useRunStore((s) => s.clearHITL)

  const mutation = useMutation({
    mutationFn: (decision: 'approve' | 'reject') => {
      if (!runId) throw new Error('no run id')
      return approveRun(runId, { decision, reason: '' })
    },
    onSuccess: () => clearHITL(),
  })

  return {
    approve: (decision: 'approve' | 'reject') => mutation.mutateAsync(decision),
    isPending: mutation.isPending,
    isError: mutation.isError,
  }
}
