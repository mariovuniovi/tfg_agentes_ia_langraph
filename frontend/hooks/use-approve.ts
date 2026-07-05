import { useMutation } from '@tanstack/react-query'
import { approveRun } from '@/lib/api'
import { useRunStore } from '@/stores/run-store'

export function useApprove(runId: string | null) {
  const clearHITL = useRunStore((s) => s.clearHITL)

  const mutation = useMutation({
    mutationFn: ({ decision, comment }: { decision: 'approve' | 'reject'; comment?: string }) => {
      if (!runId) throw new Error('no run id')
      return approveRun(runId, { decision, comment: comment ?? '' })
    },
    onSuccess: () => clearHITL(),
  })

  return {
    approve: (decision: 'approve' | 'reject', comment?: string) =>
      mutation.mutateAsync({ decision, comment }),
    isPending: mutation.isPending,
    isError: mutation.isError,
  }
}
