export function formatBytes(bytes: number): string {
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(0)} KB`
}

export function formatK(n: number): string {
  return n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n)
}

export function formatCost(usd: number): string {
  if (usd === 0) return '$0.00'
  if (usd < 0.01) return '$' + usd.toFixed(5)
  return '$' + usd.toFixed(4)
}
