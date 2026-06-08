/** 字号密度的 runtime 切换（暗色主题已移除，UI 固定日间配色）。
 *
 * tokens.css 里定义了字号密度 layer：
 *   - .density-tight / .density-loose → 紧凑 / 宽松字号 + 间距（无类即默认）
 *
 * 这个模块负责在 boot 时（main.tsx 里 `initTheme()`）确保不残留暗色类、
 * 并应用持久化的密度选择。
 */

export type Density = 'tight' | 'default' | 'loose'

const KEY_DENSITY = 'studio.density'

const DEFAULT_DENSITY: Density = 'default'

function safeGet(key: string): string | null {
  try { return localStorage.getItem(key) } catch { return null }
}
function safeSet(key: string, val: string) {
  try { localStorage.setItem(key, val) } catch { /* ignore */ }
}

// ── density ────────────────────────────────────────────────────────────────
export function getStoredDensity(): Density {
  const v = safeGet(KEY_DENSITY)
  return v === 'tight' || v === 'loose' ? v : 'default'
}

export function setStoredDensity(d: Density): void {
  safeSet(KEY_DENSITY, d)
}

export function applyDensity(d: Density): void {
  const root = document.documentElement
  root.classList.remove('density-tight', 'density-loose')
  if (d === 'tight') root.classList.add('density-tight')
  else if (d === 'loose') root.classList.add('density-loose')
  // 'default' 不加类
}

// ── boot ───────────────────────────────────────────────────────────────────
export function initTheme(): void {
  // 强制日间配色：清掉任何历史残留的暗色类。
  document.documentElement.classList.remove('theme-dark')
  applyDensity(getStoredDensity() || DEFAULT_DENSITY)
}
