import { create } from 'zustand'
import type { PipelineEvent, RunStatus } from '@/types/api'

interface RunState {
  runId: string | null
  status: RunStatus | 'idle'
  events: PipelineEvent[]
  interruptValue: Record<string, unknown> | null
  hitlPending: boolean
  stagedFiles: File[]
  setRunId: (id: string) => void
  appendEvent: (event: PipelineEvent) => void
  setHITL: (value: Record<string, unknown>) => void
  clearHITL: () => void
  setStatus: (status: RunStatus | 'idle') => void
  setStagedFiles: (files: File[]) => void
  reset: () => void
}

const initial = {
  runId: null,
  status: 'idle' as const,
  events: [],
  interruptValue: null,
  hitlPending: false,
  stagedFiles: [] as File[],
}

export const useRunStore = create<RunState>((set) => ({
  ...initial,
  setRunId: (id) => set({ runId: id, status: 'running' }),
  appendEvent: (event) => set((s) => ({ events: [...s.events, event] })),
  setHITL: (value) => set({ hitlPending: true, interruptValue: value, status: 'awaiting_approval' }),
  clearHITL: () => set({ hitlPending: false }),
  setStatus: (status) => set({ status }),
  setStagedFiles: (files) => set({ stagedFiles: files }),
  reset: () => set(initial),
}))
