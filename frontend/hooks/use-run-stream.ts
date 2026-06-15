import { useEffect, useRef } from 'react'
import { toast } from 'sonner'
import { useRunStore } from '@/stores/run-store'
import { fetchRunStatus } from '@/lib/api'
import type { PipelineEvent } from '@/types/api'

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL ?? 'ws://localhost:8000'
const MAX_RETRIES = 3

export function useRunStream(runId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const retriesRef = useRef(0)
  const status = useRunStore((s) => s.status)

  useEffect(() => {
    if (!runId) return
    const currentStatus = useRunStore.getState().status
    if (currentStatus === 'complete' || currentStatus === 'failed') return

    function connect() {
      const ws = new WebSocket(`${WS_BASE}/ws/${runId}`)
      wsRef.current = ws

      ws.onmessage = (e) => {
        const event: PipelineEvent = JSON.parse(e.data as string)
        const store = useRunStore.getState()
        // On reconnect the server replays the full event log from seq 0; skip events
        // we have already processed so replays don't duplicate or re-fire side effects.
        if (typeof event.seq === 'number' && event.seq <= store.lastSeq) return
        const { appendEvent, setHITL, setStatus } = store
        appendEvent(event)
        if (event.type === 'planner_validation_error') {
          const attempt = (event.data.attempt as number) ?? 0
          const willRetry = event.data.will_retry as boolean | undefined
          toast.warning(`Planner output failed validation (attempt ${attempt})`, {
            description: willRetry ? 'Retrying with corrective feedback…' : 'No attempts left.',
          })
        }
        if (event.type === 'planner_retry') {
          toast.info('Retrying the model planner', {
            description: 'The previous plan did not pass deterministic validation.',
          })
        }
        if (event.type === 'hitl_request') {
          setHITL(event.data)
        }
        if (event.type === 'run_complete') {
          const error = event.data.error as string | undefined
          if (error) {
            toast.error('Pipeline failed', { description: error })
            setStatus('failed')
          } else {
            toast.success('Pipeline complete')
            setStatus('complete')
          }
          ws.close()
        }
      }

      ws.onerror = () => {
        if (retriesRef.current >= MAX_RETRIES) return
        retriesRef.current++
        const delay = Math.pow(2, retriesRef.current) * 200
        setTimeout(async () => {
          const state = await fetchRunStatus(runId!)
          const { setRunId, setHITL, setStatus } = useRunStore.getState()
          setRunId(state.run_id)
          if (state.status === 'awaiting_approval' && state.interrupt_value) {
            setHITL(state.interrupt_value)
          } else {
            setStatus(state.status)
          }
          connect()
        }, delay)
      }
    }

    connect()
    return () => {
      wsRef.current?.close()
      retriesRef.current = 0
    }
  }, [runId])
}
