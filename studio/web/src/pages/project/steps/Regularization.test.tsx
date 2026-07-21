import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ExcludeTags } from './Regularization'

// 上游 v0.20.2 (#454) 的排除 tag 溢出修复。上游同 PR 的 SourceSegmented
// pill-radio 用例本 fork 不适用：来源选择器是 fork 自己的 segmented control，
// 不走 pill-radio / vs-channel-radio 那套样式，也就没有对应的回归。
describe('regularization exclusion chips', () => {
  it('keeps a long natural-language tag inside a single-height chip', () => {
    const longTag = 'a single girl with very long pale lavender hair faces toward the viewer while the background extends across the entire frame'
    render(
      <ExcludeTags
        trainTags={[{ tag: longTag, count: 1 }]}
        excluded={new Set()}
        onToggle={vi.fn()}
      />,
    )

    const text = screen.getByText(longTag)
    const textContainer = text.parentElement
    const chip = text.closest('button')

    expect(chip).toHaveClass('h-6', 'max-w-full', 'overflow-hidden', 'whitespace-nowrap')
    expect(textContainer).toHaveClass('min-w-0', 'truncate', 'text-left')
    expect(textContainer).toHaveAttribute('title', longTag)
  })
})
