import { useTranslation } from 'react-i18next'

/** 视图模式 tab：单图 / XY 矩阵。
 *
 * 用户决策：双图对比合并进 XY 模式内部（selectedIndices=2 时自动切到
 * compare sub-view，不再单独占顶部 tab）。 */

export type ViewMode = 'single' | 'xy'

export default function ViewModeTabs({
  mode, onModeChange,
}: {
  mode: ViewMode
  onModeChange: (m: ViewMode) => void
}) {
  const { t } = useTranslation()
  // Segmented control（redesign 原型 Segmented）：sunken 轨道 + 白色 active pill。
  const tab = (m: ViewMode, label: string) => (
    <button
      onClick={() => onModeChange(m)}
      role="tab"
      aria-selected={mode === m}
      className={`border-none px-3 py-[5px] text-xs rounded-[calc(var(--r-md)-2px)] transition-all duration-100 cursor-pointer ${
        mode === m
          ? 'bg-surface text-fg-primary font-semibold shadow-sm'
          : 'bg-transparent text-fg-secondary font-medium hover:text-fg-primary'
      }`}
    >
      {label}
    </button>
  )
  return (
    <div className="inline-flex items-center gap-0.5 bg-sunken rounded-md p-[3px]" role="tablist">
      {tab('single', t('generate.singleMode'))}
      {tab('xy', t('generate.xyMode'))}
    </div>
  )
}
