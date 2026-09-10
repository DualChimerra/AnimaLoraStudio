// LocalModelSources.tsx —— 设置页「用自己的权重」这一条线的复用件。
//
// 后端把每类权重的候选拍平成 catalog.model_sources[domain]（见
// studio/services/models/catalog.py 的 _source_row）：内置 preset 行 + 用户
// 注册的 local 行同构。本文件只渲染 **local 行** 与「从文件夹选一个」的入口，
// preset 行仍由各自的下载卡渲染（它们要显示下载按钮 / 分片进度）。
//
// domain 与选中值字段的对应关系（三处调用点各自负责写回）：
//   anima / krea2  → secrets.models.selected[family]     主模型权重（.safetensors）
//   vae            → secrets.models.selected_vae         VAE 权重（.safetensors）
//   anima_te / krea2_te → secrets.models.selected_te[family]  文本编码器目录
//
// 路径都是**服务端机器上的绝对路径**：本地模式下就是用户这台电脑，云端模式
// 下是容器里的盘 —— 提示语按 runtime mode 切换，避免云端用户填本机 D:\ 路径。
import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { api, type ModelSourceRow } from '../api/client'
import { useToast } from './Toast'
import { useRuntimeModeOptional } from '../lib/RuntimeMode'
import PathPicker from './PathPicker'

/** 本地权重的形态：单文件（主模型 / VAE）还是 transformers 目录（文本编码器）。 */
export type LocalSourceShape = 'file' | 'dir'

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}

interface RowsProps {
  /** catalog.model_sources 的键（anima / krea2 / vae / anima_te / krea2_te）。 */
  domain: string
  /** 该 domain 的全部候选行；本组件只渲染 kind === 'local' 的。 */
  rows: ModelSourceRow[]
  /** 单选组名（同一张卡里的 preset radio 要用同一个 name 才互斥）。 */
  radioName: string
  /** 选中一行：调用方写回自己的选中值字段（selected / selected_vae / selected_te）。 */
  onSelect: (value: string) => void | Promise<void>
  /** 注册 / 注销后刷新 catalog（选中值可能被服务端回退默认）。 */
  onChanged: () => void | Promise<void>
  /** 主模型行额外提供「工作模式」下拉：把这条权重改挂到另一个模型族。 */
  familyOptions?: { value: string; label: string }[]
  /** 换模式后把这条权重在**新族**里重新选中（仅当它换之前正被选中）。
   *  没有它的话用户改完模式还得去另一张卡再点一次单选。 */
  selectInDomain?: (domain: string, value: string) => void | Promise<void>
}

/** 已注册的本地权重行（单选 + 删除 + 可选的模式切换）。 */
export function LocalModelRows({
  domain, rows, radioName, onSelect, onChanged, familyOptions, selectInDomain,
}: RowsProps) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [busy, setBusy] = useState<string | null>(null)
  const local = rows.filter((r) => r.kind === 'local')
  if (local.length === 0) return null

  const unregister = async (row: ModelSourceRow) => {
    if (!row.candidate) return
    setBusy(row.value)
    try {
      await api.removeModelSource(domain, row.candidate)
      toast(t('settings.localModelRemoved', { name: row.label }), 'success')
      await onChanged()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(null)
    }
  }

  // 模式切换 = 换个 domain 重新登记同一条路径。原 domain 若正选中它，服务端
  // 会把该族的选中值回退官方 variant（_selected_value_reset），所以切完要
  // 重新拉 catalog 才能看到真实状态。
  const changeFamily = async (row: ModelSourceRow, nextDomain: string) => {
    if (!row.candidate || nextDomain === domain) return
    setBusy(row.value)
    try {
      await api.addModelSource(nextDomain, row.candidate)
      // 注销原候选时服务端会把原族的选中值回退官方 variant（那条权重已不在
      // 该族名下）——所以「原来正选中」的语义要在注销前读，注销后补选。
      const wasCurrent = row.is_current
      await api.removeModelSource(domain, row.candidate)
      if (wasCurrent && selectInDomain) await selectInDomain(nextDomain, row.value)
      toast(t('settings.localModelFamilyChanged', {
        name: row.label,
        family: familyOptions?.find((f) => f.value === nextDomain)?.label ?? nextDomain,
      }), 'success')
      await onChanged()
    } catch (e) {
      toast(String(e), 'error')
      await onChanged()
    } finally {
      setBusy(null)
    }
  }

  return (
    <ul className="list-none m-0 p-0 flex flex-col gap-1">
      {local.map((row) => (
        <li
          key={row.value}
          className={`flex items-center gap-2 text-xs px-1.5 py-1 rounded-sm ${
            row.is_current
              ? 'bg-accent-soft border border-accent'
              : 'bg-transparent border border-transparent'
          }`}
        >
          <input
            type="radio"
            name={radioName}
            checked={row.is_current}
            disabled={!row.exists || busy === row.value}
            onChange={() => void onSelect(row.value)}
            className="shrink-0"
            style={{ accentColor: 'var(--accent)' }}
            title={row.exists
              ? t('settings.selectLocalModel')
              : t('settings.localModelMissing')}
          />
          <code
            className="font-mono text-fg-primary w-32 shrink-0 truncate"
            title={row.value}
          >{row.label}</code>
          <span className="badge badge-neutral text-[10px] shrink-0">
            {t('settings.localBaseModel')}
          </span>
          <span className={`text-xs px-1.5 py-0.5 rounded-sm font-mono ${
            row.exists ? 'bg-ok-soft text-ok' : 'bg-err-soft text-err'
          }`}>
            {row.exists ? `✓ ${fmtBytes(row.size)}` : t('settings.localModelMissing')}
          </span>
          <span style={{ flex: 1 }} />
          {familyOptions && familyOptions.length > 1 && (
            <select
              value={domain}
              disabled={busy === row.value}
              onChange={(e) => void changeFamily(row, e.target.value)}
              className="input input-mono shrink-0"
              // .input 是 width:100% 的表单基类；这里在行内当小控件用，
              // 显式收窄免得把同一行的注销按钮挤出卡片
              style={{ padding: '2px 4px', fontSize: 'var(--t-xs)', width: 96 }}
              title={t('settings.localModelFamilyHint')}
            >
              {familyOptions.map((f) => (
                <option key={f.value} value={f.value}>{f.label}</option>
              ))}
            </select>
          )}
          <button
            onClick={() => void unregister(row)}
            disabled={busy === row.value}
            className="btn btn-ghost btn-sm shrink-0"
            title={t('settings.localModelRemoveHint')}
          >
            {t('settings.localModelRemove')}
          </button>
        </li>
      ))}
    </ul>
  )
}

interface AddProps {
  domain: string
  /** 单文件（主模型 / VAE）还是目录（文本编码器）。 */
  shape: LocalSourceShape
  /** PathPicker 的起始目录（一般传 catalog.models_root）。 */
  initialPath?: string
  onChanged: () => void | Promise<void>
}

/** 「从文件夹里选一个」入口：PathPicker 选中 → 注册成 local 候选。 */
export function AddLocalModelButton({
  domain, shape, initialPath, onChanged,
}: AddProps) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const runtime = useRuntimeModeOptional()
  const [picking, setPicking] = useState(false)
  const [saving, setSaving] = useState(false)

  const register = async (path: string) => {
    setPicking(false)
    setSaving(true)
    try {
      await api.addModelSource(domain, { kind: 'local', path })
      toast(t('settings.localModelAdded', { path }), 'success')
      await onChanged()
    } catch (e) {
      // 后端校验（后缀 / config.json 缺失 / 路径不存在）的报错原样透出，
      // 用户据此改选。
      toast(String(e), 'error')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex items-center gap-2 mt-1.5">
      <button
        onClick={() => setPicking(true)}
        disabled={saving}
        className="btn btn-secondary btn-sm"
      >
        {saving
          ? t('common.saving')
          : shape === 'dir'
            ? t('settings.addLocalModelDir')
            : t('settings.addLocalModelFile')}
      </button>
      <span className="text-fg-tertiary text-xs">
        {runtime?.mode === 'colab'
          ? t('settings.localModelPathHintColab')
          : t('settings.localModelPathHintLocal')}
      </span>
      {picking && (
        <PathPicker
          initialPath={initialPath}
          dirOnly={shape === 'dir'}
          onPick={(p) => void register(p)}
          onClose={() => setPicking(false)}
        />
      )}
    </div>
  )
}
