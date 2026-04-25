import type { Metadata } from 'next'
import './globals.css'
import { TopNav } from '@/components/TopNav'
import { Providers } from '@/components/Providers'
import { Toaster } from 'sonner'

export const metadata: Metadata = { title: 'MLOps Dashboard' }

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Fira+Sans:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen bg-slate-50 font-sans text-slate-900">
        <Providers>
          <TopNav />
          <main className="p-6">{children}</main>
          <Toaster position="bottom-right" richColors />
        </Providers>
      </body>
    </html>
  )
}
