import { Badge, type BadgeVariant } from './Badge'

export type NodeType = 'agent' | 'llm' | 'deterministic' | 'hitl'

const MAP: Record<NodeType, { variant: BadgeVariant; label: string }> = {
  agent:         { variant: 'llm',     label: 'Agent' },
  llm:           { variant: 'llm',     label: 'LLM' },
  deterministic: { variant: 'info',    label: 'Deterministic' },
  hitl:          { variant: 'warning', label: 'HITL' },
}

export function NodeTypeBadge({ type }: { type: NodeType }) {
  const { variant, label } = MAP[type]
  return <Badge variant={variant}>{label}</Badge>
}
