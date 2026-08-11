// RuntimeModeGate.tsx —— 首次进应用时的「Colab 还是 Local」选择框（本 fork）。
//
// 为什么是一次性阻断式的：两种模式下**合理的默认值互相冲突**（绑定地址、要不
// 要开浏览器、studio_data 放哪、要不要提醒同步到 Drive）。以前靠 CLI flag 各自
// 记，用户装完不知道要传什么；现在进应用先答一次，落盘后不再问。
//
// 探测结果只做预选，不代替用户按 —— 自建 JupyterHub / docker 里跑本地训练这类
// 场景探测必然会猜错，硬替用户选反而更难纠正。卡片上把探测依据摆出来。
//
// 被 `ALS_RUNTIME_MODE` 钉死时本组件不渲染（`needsPick` 为 false），Colab
// notebook 的启动 cell 设了它，云端用户开箱即用不会被拦。
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import type { RuntimeMode } from '../api/client'
import { useRuntimeMode } from '../lib/RuntimeMode'

function formatBytes(n: number | null | undefined): string {
  if (!n || n <= 0) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1 }
  return `${v >= 10 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`
}

export default function RuntimeModeGate() {
  const { needsPick, info, setMode } = useRuntimeMode()
  const { t } = useTranslation()
  const [choice, setChoice] = useState<RuntimeMode | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // 探测结果做预选。info 是异步到的，所以在 effect 里补而不是 useState 初值。
  useEffect(() => {
    if (info && choice === null) setChoice(info.detected)
  }, [info, choice])

  if (!needsPick || !info) return null

  const env = info.environment
  const confirm = async () => {
    if (!choice) return
    setSaving(true)
    setError(null)
    try {
      await setMode(choice)
    } catch (e) {
      setError(String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    // 刻意没有 backdrop onClick 关闭、也没有 × —— 这是必答题，答完就再也不出现。
    <div
      className="fixed inset-0 z-[60] bg-black/60 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="runtime-mode-title"
    >
      <div className="bg-elevated border border-dim rounded-lg shadow-xl w-[640px] max-w-full max-h-[90vh] flex flex-col overflow-hidden">
        <header className="px-5 pt-5 pb-3 shrink-0">
          <h2 id="runtime-mode-title" className="m-0 text-base font-semibold text-fg-primary">
            {t('runtimeMode.title')}
          </h2>
          <p className="mt-1 mb-0 text-xs text-fg-tertiary">
            {t('runtimeMode.subtitle')}
          </p>
        </header>

        <div className="px-5 pb-4 flex flex-col gap-3 overflow-y-auto">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <ModeCard
              mode="local"
              selected={choice === 'local'}
              recommended={info.detected === 'local'}
              onSelect={() => setChoice('local')}
            />
            <ModeCard
              mode="colab"
              selected={choice === 'colab'}
              recommended={info.detected === 'colab'}
              onSelect={() => setChoice('colab')}
            />
          </div>

          {/* 探测依据：用户据此判断预选对不对，也是选错后回来自查的入口。 */}
          <div className="rounded-md border border-subtle bg-surface px-3 py-2 text-xs text-fg-secondary flex flex-col gap-1">
            <div className="text-fg-tertiary">{t('runtimeMode.detectedAs', {
              mode: t(`runtimeMode.${info.detected}.name`),
            })}</div>
            <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono">
              <span>{env.platform} · Python {env.python}</span>
              {env.gpu ? <span>GPU: {env.gpu}</span> : <span>{t('runtimeMode.noGpu')}</span>}
              <span>{t('runtimeMode.diskFree', {
                free: formatBytes(env.disk_free),
                total: formatBytes(env.disk_total),
              })}</span>
            </div>
            <div className="font-mono break-all">
              studio_data: {env.studio_data}
            </div>
          </div>

          {error && (
            <div className="rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-danger">
              {error}
            </div>
          )}
        </div>

        <footer className="px-5 py-3 border-t border-subtle flex items-center gap-2 shrink-0">
          <span className="text-xs text-fg-tertiary flex-1">
            {t('runtimeMode.changeLater')}
          </span>
          <button
            className="btn btn-primary text-sm"
            disabled={!choice || saving}
            onClick={() => void confirm()}
          >
            {saving ? t('runtimeMode.saving') : t('runtimeMode.confirm')}
          </button>
        </footer>
      </div>
    </div>
  )
}

function ModeCard({ mode, selected, recommended, onSelect }: {
  mode: RuntimeMode
  selected: boolean
  recommended: boolean
  onSelect: () => void
}) {
  const { t } = useTranslation()
  const bullets = t(`runtimeMode.${mode}.bullets`, { returnObjects: true }) as string[]
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={`text-left rounded-md border p-3 flex flex-col gap-2 transition-colors ${
        selected
          ? 'border-accent bg-accent/10'
          : 'border-subtle bg-surface hover:border-dim'
      }`}
    >
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-fg-primary">
          {t(`runtimeMode.${mode}.name`)}
        </span>
        {recommended && (
          <span className="text-[10px] uppercase tracking-wide rounded px-1.5 py-0.5 bg-accent/20 text-accent">
            {t('runtimeMode.recommended')}
          </span>
        )}
      </div>
      <p className="m-0 text-xs text-fg-secondary">{t(`runtimeMode.${mode}.summary`)}</p>
      <ul className="m-0 pl-4 text-xs text-fg-tertiary flex flex-col gap-0.5">
        {Array.isArray(bullets) && bullets.map((b, i) => <li key={i}>{b}</li>)}
      </ul>
    </button>
  )
}
