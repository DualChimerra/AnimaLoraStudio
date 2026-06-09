import type { ReactNode } from 'react'

interface Props {
  title: string
  subtitle?: string
  /** 小标题上方的 mono caption（"PROJECT" / "GENERATE" 等），对齐 redesign 原型。 */
  eyebrow?: string
  /** eyebrow 用 accent 色（默认 tertiary 灰）。 */
  accentEyebrow?: boolean
  /** Tab 导航条；如果传了 tabs 则 subtitle 不渲染（tab 取代 description 位置）。 */
  tabs?: ReactNode
  actions?: ReactNode
  /** 右上角 slot —— 跟 title 顶部对齐的独立位置（脱离 actions 行）。
   *  专用于 PhaseHeaderNav 等"位置必须固定在右上"的辅助导航。 */
  topRight?: ReactNode
  sticky?: boolean
}

export default function PageHeader({ title, subtitle, eyebrow, accentEyebrow, tabs, actions, topRight, sticky }: Props) {
  return (
    <div className={`px-7 pt-[22px] pb-[18px] bg-canvas border-b border-subtle ${sticky ? 'sticky top-0 z-[5]' : 'relative'}`}>
      {topRight && (
        <div className="absolute top-3 right-6 z-[1]">{topRight}</div>
      )}
      <div className="flex items-end gap-[18px] flex-wrap">
        <div className="flex-1 min-w-0">
          {eyebrow && (
            <div className="caption mb-[7px]" style={{ color: accentEyebrow ? 'var(--accent)' : 'var(--fg-tertiary)' }}>{eyebrow}</div>
          )}
          <h1 className="m-0 text-3xl font-bold tracking-[-0.02em] leading-[1.1]">{title}</h1>
          {/* tabs 在主标题下方取代 subtitle 位置；两者互斥（tabs 优先）。 */}
          {tabs ? (
            <div className="mt-[14px]">{tabs}</div>
          ) : (
            subtitle && (
              <p className="mt-[7px] text-fg-secondary text-md max-w-[760px] m-0">{subtitle}</p>
            )
          )}
        </div>
        {actions && (
          <div className="flex gap-2.5 items-center">{actions}</div>
        )}
      </div>
    </div>
  )
}
