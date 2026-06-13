/** 字号密度（暗色主题已移除，UI 固定日间配色）。
 *
 * redesign 原型定稿：界面固定 density-tight（紧凑），密度切换已移除
 * （设计迭代 chat2 #8："Сделай по умолчанию плотный вариант интерфейса,
 * переключения плотности также не должно быть"）。
 *
 * 这个模块只负责 boot 时（main.tsx 里 `initTheme()`）清理历史残留类并
 * 固定应用 density-tight。
 */

export type Density = 'tight' | 'default' | 'loose'

export function applyDensity(d: Density): void {
  const root = document.documentElement
  root.classList.remove('density-tight', 'density-loose')
  if (d === 'tight') root.classList.add('density-tight')
  else if (d === 'loose') root.classList.add('density-loose')
}

// ── boot ───────────────────────────────────────────────────────────────────
export function initTheme(): void {
  // 强制日间配色：清掉任何历史残留的暗色类。
  document.documentElement.classList.remove('theme-dark')
  // 密度固定紧凑（原型定稿，无切换）。
  applyDensity('tight')
}
