import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import '../i18n'

vi.mock('../api/client', () => ({
  api: {
    addModelSource: vi.fn(),
    removeModelSource: vi.fn(),
  },
}))

import { api, type ModelSourceRow } from '../api/client'
import { ToastProvider } from './Toast'
import { AddLocalModelButton, LocalModelRows } from './LocalModelSources'

function row(over: Partial<ModelSourceRow> = {}): ModelSourceRow {
  return {
    kind: 'local',
    candidate: { kind: 'local', path: 'G:/m/ft.safetensors' },
    value: 'G:/m/ft.safetensors',
    label: 'ft.safetensors',
    description: 'G:/m/ft.safetensors',
    download_id: null,
    download_variant: null,
    status_key: null,
    exists: true,
    size: 3_000_000,
    size_estimate: 0,
    is_current: false,
    removable: true,
    deletable: false,
    extra: {},
    ...over,
  }
}

beforeEach(() => { vi.clearAllMocks() })

const FAMILIES = [
  { value: 'anima', label: 'Anima' },
  { value: 'krea2', label: 'Krea 2' },
]

function renderRows(props: Partial<Parameters<typeof LocalModelRows>[0]> = {}) {
  return render(
    <ToastProvider>
      <LocalModelRows
        domain="anima"
        rows={[row({ kind: 'preset', value: '1.0', candidate: null }), row()]}
        radioName="anima_variant"
        onSelect={() => {}}
        onChanged={() => {}}
        {...props}
      />
    </ToastProvider>,
  )
}

describe('LocalModelRows', () => {
  it('renders only local candidates and selects one', async () => {
    const onSelect = vi.fn()
    renderRows({ onSelect })
    // preset 行归各自的下载卡渲染，这里只出本地那条
    expect(screen.getAllByRole('radio')).toHaveLength(1)
    await userEvent.click(screen.getByRole('radio'))
    expect(onSelect).toHaveBeenCalledWith('G:/m/ft.safetensors')
  })

  it('missing file cannot be selected', () => {
    renderRows({ rows: [row({ exists: false })] })
    expect(screen.getByRole('radio')).toBeDisabled()
  })

  it('unregister removes the candidate and refreshes', async () => {
    const onChanged = vi.fn()
    vi.mocked(api.removeModelSource).mockResolvedValue({} as never)
    renderRows({ onChanged })
    await userEvent.click(screen.getByRole('button'))
    await waitFor(() => expect(onChanged).toHaveBeenCalled())
    expect(api.removeModelSource).toHaveBeenCalledWith(
      'anima', { kind: 'local', path: 'G:/m/ft.safetensors' },
    )
  })

  it('changing the working mode re-registers under the other family', async () => {
    vi.mocked(api.addModelSource).mockResolvedValue({} as never)
    vi.mocked(api.removeModelSource).mockResolvedValue({} as never)
    const selectInDomain = vi.fn()
    renderRows({
      rows: [row({ is_current: true })],
      familyOptions: FAMILIES,
      selectInDomain,
    })
    await userEvent.selectOptions(screen.getByRole('combobox'), 'krea2')
    await waitFor(() => expect(api.addModelSource).toHaveBeenCalledWith(
      'krea2', { kind: 'local', path: 'G:/m/ft.safetensors' },
    ))
    expect(api.removeModelSource).toHaveBeenCalledWith(
      'anima', { kind: 'local', path: 'G:/m/ft.safetensors' },
    )
    // 换模式前正被选中 → 在新族里补选，用户不用再点一次
    expect(selectInDomain).toHaveBeenCalledWith('krea2', 'G:/m/ft.safetensors')
  })

  it('mode dropdown is hidden for asset domains (vae / text encoder)', () => {
    renderRows({ domain: 'vae' })
    expect(screen.queryByRole('combobox')).toBeNull()
  })
})

describe('AddLocalModelButton', () => {
  it('registers the picked path as a local candidate', async () => {
    vi.mocked(api.addModelSource).mockResolvedValue({} as never)
    const onChanged = vi.fn()
    render(
      <ToastProvider>
        <AddLocalModelButton
          domain="vae"
          shape="file"
          initialPath="G:/models"
          onChanged={onChanged}
        />
      </ToastProvider>,
    )
    await userEvent.click(screen.getByRole('button'))
    // PathPicker 的底部按钮：直接选中当前目录（浏览请求在 client mock 里没有，
    // 组件仍会渲染出这个按钮）
    await userEvent.click(await screen.findByText(/Select current directory|选择当前目录/))
    await waitFor(() => expect(api.addModelSource).toHaveBeenCalled())
    expect(vi.mocked(api.addModelSource).mock.calls[0][0]).toBe('vae')
    expect(onChanged).toHaveBeenCalled()
  })
})
