'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from '@/lib/api'

const TABS = [
  { label: 'Pipeline',      href: '/pipeline' },
  { label: 'Experiments',   href: '/experiments' },
  { label: 'Observability', href: '/observability' },
  { label: 'Monitoring',    href: '/monitoring' },
]

export function TopNav() {
  const pathname = usePathname()
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 30_000,
    retry: false,
  })

  const unhealthy = health && (!health.mlflow || !health.graph)

  return (
    <nav className="flex items-center gap-1 border-b border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-2">
      {TABS.map(({ label, href }) => {
        const active = pathname.startsWith(href)
        return (
          <Link
            key={href}
            href={href}
            className={`rounded px-3 py-1.5 text-sm font-medium transition-colors ${
              active
                ? 'bg-indigo-50 text-indigo-700'
                : 'text-[var(--color-fg-muted)] hover:bg-zinc-100 hover:text-[var(--color-fg)]'
            }`}
          >
            {label}
          </Link>
        )
      })}
      {unhealthy && (
        <span className="ml-auto rounded-full bg-red-50 px-2 py-0.5 text-xs font-medium text-red-700 ring-1 ring-inset ring-red-200">
          Backend unhealthy
        </span>
      )}
    </nav>
  )
}
