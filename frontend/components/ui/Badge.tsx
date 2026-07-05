import type { ReactNode } from 'react'

export type BadgeVariant = 'success' | 'warning' | 'danger' | 'info' | 'llm' | 'neutral' | 'accent'

const STYLES: Record<BadgeVariant, string> = {
  success: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
  warning: 'bg-amber-50  text-amber-700  ring-amber-200',
  danger:  'bg-red-50    text-red-700    ring-red-200',
  info:    'bg-sky-50    text-sky-700    ring-sky-200',
  llm:     'bg-violet-50 text-violet-700 ring-violet-200',
  accent:  'bg-indigo-50 text-indigo-700 ring-indigo-200',
  neutral: 'bg-zinc-100  text-zinc-700   ring-zinc-200',
}

interface BadgeProps {
  variant?: BadgeVariant
  children: ReactNode
  className?: string
}

export function Badge({ variant = 'neutral', children, className = '' }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${STYLES[variant]} ${className}`}
    >
      {children}
    </span>
  )
}
