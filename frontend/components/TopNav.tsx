'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from '@/lib/api'

const TABS = [
  { label: 'Pipeline', href: '/pipeline' },
  { label: 'Experiments', href: '/experiments' },
  { label: 'Monitoring', href: '/monitoring' },
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
    <nav className="flex items-center gap-1 bg-navy px-4 py-2">
      {TABS.map(({ label, href }) => {
        const active = pathname.startsWith(href)
        return (
          <Link
            key={href}
            href={href}
            className={`rounded px-4 py-2 text-sm font-medium transition-colors ${
              active
                ? 'bg-navy text-white ring-1 ring-white/30'
                : 'text-slate-300 hover:bg-white/10 hover:text-white'
            }`}
          >
            {label}
          </Link>
        )
      })}
      {unhealthy && (
        <span className="ml-auto rounded-full bg-red-500 px-3 py-1 text-xs font-medium text-white">
          Backend unhealthy
        </span>
      )}
    </nav>
  )
}
