import type { ReactNode } from 'react'
import PageHeader from './PageHeader'

interface Props {
  idx: number | string
  title: string
  subtitle?: string
  /** 主标题上方的 caption；不传时由 idx 派生（数字 idx → "Step N"）。 */
  eyebrow?: string
  accentEyebrow?: boolean
  actions?: ReactNode
  topRight?: ReactNode
  children: ReactNode
}

export default function StepShell({ idx, title, subtitle, eyebrow, accentEyebrow, actions, topRight, children }: Props) {
  // 数字 idx → "Step N" eyebrow，匹配 redesign 原型的流水线步骤标题。
  const derivedEyebrow = eyebrow ?? (typeof idx === 'number' || /^\d+$/.test(String(idx)) ? `Step ${idx}` : undefined)
  return (
    <div className="fade-in flex flex-col h-full">
      <PageHeader
        title={title}
        eyebrow={derivedEyebrow}
        accentEyebrow={accentEyebrow}
        subtitle={subtitle}
        actions={actions}
        topRight={topRight}
        sticky
      />
      {/* flex column container: overflow:hidden stops page scroll; children use flex:1 to fill */}
      <div className="flex-1 min-h-0 p-6 flex flex-col overflow-hidden">
        {children}
      </div>
    </div>
  )
}
