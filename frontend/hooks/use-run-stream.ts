import { useEffect, useRef } from 'react'
import { useRunStore } from '@/stores/run-store'
import { fetchRunStatus } from '@/lib/api'
import type { PipelineEvent } from '@/types/api'

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL ?? 'ws://localhost:8000'
const MAX_RETRIES = 3

export function useRunStream(runId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const retriesRef = useRef(0)

  useEffect(() => {
    if (!runId) return

    function connect() {
      const ws = new WebSocket(`${WS_BASE}/ws/${runId}`)
      wsRef.current = ws

      ws.onmessage = (e) => {
        const event: PipelineEvent = JSON.parse(e.data as string)
        const { appendEvent, setHITL, setStatus } = useRunStore.getState()
        appendEvent(event)
        if (event.type === 'hitl_request') {
          setHITL(event.data)
        }
        if (event.type === 'run_complete') {
          setStatus('complete')
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
