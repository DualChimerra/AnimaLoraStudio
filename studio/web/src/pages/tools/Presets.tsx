import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  type ApiError,
  type ConfigData,
  type PresetSummary,
  type SchemaResponse,
} from '../../api/client'
import ConfigSkeleton from '../../components/ConfigSkeleton'
import { useDialog } from '../../components/Dialog'
import PageHeader from '../../components/PageHeader'
import PathPicker from '../../components/PathPicker'
import SchemaForm from '../../components/SchemaForm'
import { useToast } from '../../components/Toast'
import { useSettingsDrawer } from '../../lib/SettingsDrawer'
import { useAdvancedMode } from '../../lib/useAdvancedMode'
import {
  PRESET_NAME_RE,
  defaultsFromSchema,
  loadPresetDescriptions,
  savePresetDescriptions,
} from '../../lib/preset-helpers'

// ── TOML 生成（键按字母排序，值尽量保留原始类型） ──────────────────────────
function toTomlValue(v: unknown): string {
  if (v === null || v === undefined) return ''
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number') return String(v)
  if (Array.isArray(v)) return '[' + v.map(toTomlValue).join(', ') + ']'
  if (typeof v === 'object') {
    const lines: string[] = []
    for (const [k, vv] of Object.entries(v as Record<string, unknown>)) {
      lines.push(`  ${k} = ${toTomlValue(vv)}`)
    }
    return '{\n' + lines.join('\n') + '\n}'
  }
  const s = String(v)
  if (/[\n"'#[\]{}]/.test(s)) return `'''\n${s}\n'''`
  if (s.includes(' ') || s === '' || /[^\w.\-]/.test(s)) return `"${s}"`
  return s
}

function generateToml(config: ConfigData): string {
  const keys = Object.keys(config).sort()
  return keys.map((k) => `${k} = ${toTomlValue(config[k])}`).join('\n')
}

// 相对时间（与 Overview 同款，prototype Presets 表格 / 详情用）。
function fmtAgo(ts: number): string {
  const sec = Math.max(0, Date.now() / 1000 - ts)
  if (sec < 60) return 'just now'
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`
  return `${Math.floor(sec / 86400)}d ago`
}

// 表格列 / 详情卡用：从一份 config 派生 optimizer / rank / resolution。
function cfgOptimizer(c?: ConfigData): string {
  return (c?.optimizer_type as string | undefined) ?? '—'
}
function cfgRank(c?: ConfigData): string {
  const v = c?.lora_rank
  return v === undefined || v === null ? '—' : String(v)
}
function cfgAlpha(c?: ConfigData): string {
  const v = c?.lora_alpha
  return v === undefined || v === null ? '—' : String(v)
}
function cfgRes(c?: ConfigData): string {
  const v = c?.resolution
  return v === undefined || v === null ? '—' : String(v)
}

// 预设名校验 / 描述存储 / schema 默认值 抽到 lib/preset-helpers.ts，
// 跟 Train 页面「新建预设」内联表单共享，避免两份维护。

// 上传冲突时,后端 409 body 透传到这里;用户决定覆盖 / 另存为 / 取消。
interface ConflictState {
  config: ConfigData
  desc: string
  suggestedName: string
}

type ConflictChoice =
  | { kind: 'overwrite' }
  | { kind: 'saveAs'; name: string }
  | { kind: 'cancel' }

export default function PresetsPage() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const settingsDrawer = useSettingsDrawer()

  // ── backend state ──
  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [presets, setPresets] = useState<PresetSummary[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [busy, setBusy] = useState(false)
  const [autoSyncPaths, setAutoSyncPaths] = useState<boolean>(true)
  // 4 个模型字段当前 Settings 算出的绝对路径（reset 按钮 + 新建预设默认值）
  const [modelPathDefaults, setModelPathDefaults] = useState<Record<string, string>>({})
  // prototype 表格列（Optimizer/Rank/Res）需要每个 preset 的 config —— 列表 API
  // 只回 name/path/updated，这里按需拉全部 config 填表格（preset 数量很少）。
  const [configCache, setConfigCache] = useState<Record<string, ConfigData>>({})

  // 已保存快照，用于 dirty 判定
  const savedJsonRef = useRef<string | null>(null)
  const [droppedFields, setDroppedFields] = useState<string[]>([])
  const [defaultedFields, setDefaultedFields] = useState<string[]>([])

  // 描述
  const [descriptions, setDescriptions] = useState<Record<string, string>>(loadPresetDescriptions)
  const [descDraft, setDescDraft] = useState('')
  const [descDirty, setDescDirty] = useState(false)

  // 新建模式输入
  const [newName, setNewName] = useState('')
  const [newNameError, setNewNameError] = useState('')
  const isNew = selected === null

  // ── 上传冲突 dialog 状态 + 命令式 resolver ──
  const [conflict, setConflict] = useState<ConflictState | null>(null)
  const conflictResolveRef = useRef<((c: ConflictChoice) => void) | null>(null)
  const askConflict = (state: ConflictState): Promise<ConflictChoice> =>
    new Promise((resolve) => {
      conflictResolveRef.current = resolve
      setConflict(state)
    })
  const resolveConflict = (choice: ConflictChoice) => {
    setConflict(null)
    const r = conflictResolveRef.current
    conflictResolveRef.current = null
    r?.(choice)
  }

  // ── UI 状态 ──
  // editorOpen：prototype 把整套 schema 编辑收进「Edit config / New preset」模态。
  const [editorOpen, setEditorOpen] = useState(false)
  const [tomlOpen, setTomlOpen] = useState(false)
  const [exportDialogOpen, setExportDialogOpen] = useState(false)
  const [showImportPathPicker, setShowImportPathPicker] = useState(false)
  const [advancedMode, toggleAdvancedMode] = useAdvancedMode()
  const newNameInputRef = useRef<HTMLInputElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // 4 个模型字段（用于新建预设默认值 / reset 按钮）。同 Train.tsx 的 GLOBAL_MODEL_FIELDS。
  const MODEL_PATH_FIELDS = useMemo(() => [
    'transformer_path', 'vae_path', 'text_encoder_path', 't5_tokenizer_path',
  ], [])

  // ── 加载 schema + 预设列表 + Settings toggle + 模型路径默认 ──
  useEffect(() => {
    api.schema().then(setSchema).catch((e) => toast(t('presets.loadSchemaFailed', { error: e }), 'error'))
    refreshList()
    api.getSecrets().then((s) => setAutoSyncPaths(s.models?.auto_sync_paths ?? true)).catch(() => {})
    api.getModelPathDefaults().then(setModelPathDefaults).catch(() => {})
  }, [t, toast])

  const refreshList = () => {
    api.listPresets().then((list) => {
      setPresets(list)
      // 拉每个 preset 的 config 填表格列（best-effort，失败列显示 —）。
      list.forEach((p) => {
        api.getPreset(p.name)
          .then((c) => setConfigCache((m) => ({ ...m, [p.name]: c })))
          .catch(() => {})
      })
    }).catch(() => setPresets([]))
  }

  // ── 选 preset 切换 ──
  useEffect(() => {
    if (!selected) {
      if (schema) {
        const defaults = { ...defaultsFromSchema(schema), ...modelPathDefaults }
        setConfig(defaults)
        savedJsonRef.current = JSON.stringify(defaults)
        setNewName('')
        setDescDraft('')
        setDescDirty(false)
        setDroppedFields([])
        setDefaultedFields([])
      } else {
        setConfig(null)
        savedJsonRef.current = null
        setNewName('')
        setDescDraft('')
        setDescDirty(false)
        setDroppedFields([])
        setDefaultedFields([])
      }
      setNewNameError('')
      return
    }
    api.getPresetWithWarnings(selected).then(({ config: data, dropped_fields, defaulted_fields }) => {
      setConfig(data)
      savedJsonRef.current = JSON.stringify(data)
      setDroppedFields(dropped_fields)
      setDefaultedFields(defaulted_fields)
      setDescDraft(descriptions[selected] ?? '')
      setDescDirty(false)
    }).catch((e) => {
      toast(t('presets.loadFailed', { error: e }), 'error')
      setSelected(null)
    })
    // modelPathDefaults 故意排除：late-arrival 由下一个 useEffect 处理
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, schema, descriptions, t, toast])

  // modelPathDefaults 异步晚到时，新建模式下用户没改过就地覆盖 4 字段为绝对路径。
  useEffect(() => {
    if (selected !== null) return
    if (!schema || !config) return
    if (Object.keys(modelPathDefaults).length === 0) return
    const currentJson = JSON.stringify(config)
    if (currentJson !== savedJsonRef.current) return
    let needsUpdate = false
    for (const f of MODEL_PATH_FIELDS) {
      if (modelPathDefaults[f] && config[f] !== modelPathDefaults[f]) {
        needsUpdate = true
        break
      }
    }
    if (!needsUpdate) return
    const next = { ...config, ...modelPathDefaults }
    setConfig(next)
    savedJsonRef.current = JSON.stringify(next)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelPathDefaults, selected, schema])

  // ── 首次拿到列表后：自动选最近一个，省一次「切换」点击 ──
  const autoSelectedRef = useRef(false)
  useEffect(() => {
    if (autoSelectedRef.current) return
    if (presets.length > 0 && selected === null) {
      autoSelectedRef.current = true
      setSelected(presets[0].name)
    } else if (presets.length === 0 && schema) {
      autoSelectedRef.current = true
    }
  }, [presets, selected, schema])

  // 编辑器开着时 Esc 关闭（dirty 时仍可关 —— 改动留在内存，跟切 preset 一致）。
  useEffect(() => {
    if (!editorOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setEditorOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [editorOpen])

  // ── 派生 ──
  const dirty = useMemo(() => {
    if (!config) return false
    return JSON.stringify(config) !== savedJsonRef.current
  }, [config])
  const hasAnyChange = dirty || descDirty

  // auto_sync_paths ON：预设里 4 模型字段灰显。OFF：可编辑 + 重置按钮。
  const disabledFields = autoSyncPaths ? MODEL_PATH_FIELDS : []
  const disabledHints = useMemo(() => {
    const h: Record<string, React.ReactNode> = {}
    if (autoSyncPaths) {
      const node = (
        <>
          {t('train.globalAutoLockedPrefix')} ·{' '}
          <button
            type="button"
            onClick={() => settingsDrawer.open({ section: 'models' })}
            className="bg-transparent border-none p-0 underline text-warn hover:opacity-80 cursor-pointer"
          >
            {t('train.globalAutoLockedLink')}
          </button>
        </>
      )
      for (const f of MODEL_PATH_FIELDS) h[f] = node
    }
    return h
  }, [t, autoSyncPaths, MODEL_PATH_FIELDS, settingsDrawer])
  const autoHints = useMemo(() => {
    const h: Record<string, string> = {}
    if (!autoSyncPaths) {
      for (const f of MODEL_PATH_FIELDS) h[f] = t('train.globalAutoEditableHint')
    }
    return h
  }, [t, autoSyncPaths, MODEL_PATH_FIELDS])

  const fieldSuffixes = useMemo(() => {
    if (autoSyncPaths) return {}
    if (!config) return {}
    if (Object.keys(modelPathDefaults).length === 0) return {}
    const out: Record<string, React.ReactNode> = {}
    for (const f of MODEL_PATH_FIELDS) {
      const dv = modelPathDefaults[f]
      if (typeof dv !== 'string' || !dv) continue
      out[f] = (
        <button
          type="button"
          onClick={() => setConfig({ ...config, [f]: dv })}
          className="btn btn-ghost btn-sm shrink-0"
          title={t('train.resetToGlobalDefaultTitle')}
        >
          {t('train.resetToGlobalDefault')}
        </button>
      )
    }
    return out
  }, [autoSyncPaths, modelPathDefaults, config, t, MODEL_PATH_FIELDS])

  // ── 操作 ──
  const handleSave = async () => {
    const name = isNew ? newName.trim() : selected
    if (!name) {
      setNewNameError(t('presets.nameRequired'))
      newNameInputRef.current?.focus()
      return
    }
    if (!config) return
    if (isNew) {
      if (!PRESET_NAME_RE.test(name)) { setNewNameError(t('presets.nameInvalid')); return }
      if (presets.find((p) => p.name === name)) { setNewNameError(t('presets.nameExists')); return }
    }
    setBusy(true)
    try {
      await api.savePreset(name, config)
      if (descDraft) {
        const next = { ...descriptions, [name]: descDraft }
        setDescriptions(next); savePresetDescriptions(next)
      } else if (descriptions[name]) {
        const { [name]: _, ...rest } = descriptions
        setDescriptions(rest); savePresetDescriptions(rest)
      }
      savedJsonRef.current = JSON.stringify(config)
      setConfigCache((m) => ({ ...m, [name]: config }))
      setDescDirty(false)
      if (isNew) {
        setSelected(name)
        setNewName('')
        setNewNameError('')
        toast(t('presets.created', { name }), 'success')
      } else {
        toast(t('presets.saved'), 'success')
      }
      setEditorOpen(false)
      refreshList()
    } catch (e) { toast(String(e), 'error') }
    finally { setBusy(false) }
  }

  // "复制副本":Save-As 语义 —— 把当前 config 写到新名字下,refresh + 自动选中。
  const handleDuplicate = async () => {
    if (!config || busy) return
    const baseName = selected ?? 'preset'
    let candidate = `${baseName}-copy`
    let i = 2
    while (presets.find((p) => p.name === candidate)) {
      candidate = `${baseName}-copy-${i++}`
    }
    setBusy(true)
    try {
      await api.savePreset(candidate, config)
      if (descDraft) {
        const next = { ...descriptions, [candidate]: descDraft }
        setDescriptions(next); savePresetDescriptions(next)
      }
      refreshList()
      setSelected(candidate)
      toast(t('presets.duplicated', { name: candidate }), 'success')
    } catch (e) { toast(String(e), 'error') }
    finally { setBusy(false) }
  }

  // 「+ New preset」：进新建模式 + 打开编辑器（schema 默认值由 selected→null effect 预填）。
  const handleNew = () => {
    setSelected(null)
    setEditorOpen(true)
  }

  const handleDelete = async () => {
    if (!selected) return
    if (!(await confirm(t('presets.confirmDelete', { name: selected }), { tone: 'danger', okText: t('common.delete') }))) return
    setBusy(true)
    const target = selected
    api.deletePreset(target).then(() => {
      const { [target]: _, ...rest } = descriptions
      setDescriptions(rest); savePresetDescriptions(rest)
      setConfigCache((m) => { const { [target]: _drop, ...keep } = m; return keep })
      setSelected(null)
      refreshList()
      toast(t('presets.deleted'), 'success')
    }).catch((e) => toast(String(e), 'error')).finally(() => setBusy(false))
  }

  const currentExportName = () => (isNew ? newName.trim() : selected) || 'preset'

  const downloadCurrentPreset = () => {
    if (!config) return
    if (isNew || !selected || hasAnyChange) {
      toast(t('presets.saveBeforeDownload'), 'info')
      return
    }
    const a = document.createElement('a')
    a.href = api.presetDownloadUrl(selected)
    a.download = `${selected}.yaml`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const exportCurrentPresetToDataExports = async () => {
    if (!config) return
    setBusy(true)
    try {
      const result = await api.exportPresetToDataExports(currentExportName(), config)
      toast(t('presets.exportedToDataExports', { filename: result.filename, path: result.path }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  // 「导入」：上传 / server path → 后端校验落盘 → refresh + 选中；409 冲突弹三选一。
  const handleImportedPreset = (name: string) => {
    refreshList()
    setSelected(name)
    toast(t('presets.imported', { name }), 'success')
  }

  const handleImportConflict = async (err: ApiError): Promise<boolean> => {
    if (err.status === 409 && err.detail && typeof err.detail === 'object') {
      const d = err.detail as { config?: ConfigData; suggested_name?: string }
      if (!d.config || !d.suggested_name) { toast(String(err), 'error'); return true }
      const choice = await askConflict({
        config: d.config, desc: '', suggestedName: d.suggested_name,
      })
      if (choice.kind === 'cancel') return true
      const target = choice.kind === 'overwrite' ? d.suggested_name : choice.name
      setBusy(true)
      try {
        await api.savePreset(target, d.config)
        handleImportedPreset(target)
      } catch (saveErr) { toast(String(saveErr), 'error') }
      finally { setBusy(false) }
      return true
    }
    return false
  }

  const handleImportFile = async (f: File) => {
    let imported: { name: string }
    try {
      imported = await api.importPreset(f)
    } catch (e) {
      const err = e as ApiError
      if (await handleImportConflict(err)) return
      toast(String(e), 'error')
      return
    }
    handleImportedPreset(imported.name)
  }

  const handleImportFromPath = async (path: string) => {
    setShowImportPathPicker(false)
    setBusy(true)
    try {
      const imported = await api.importPresetFromPath(path)
      handleImportedPreset(imported.name)
    } catch (e) {
      const err = e as ApiError
      if (await handleImportConflict(err)) return
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const onImportClick = () => fileInputRef.current?.click()

  const saveDisabled =
    busy
    || !config
    || (isNew && !newName.trim())
    || (!isNew && !hasAnyChange)

  const selectedConfig = selected ? configCache[selected] : undefined

  // ── 渲染 ──
  return (
    <div className="fade-in">
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,.yaml,.yml"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) void handleImportFile(f)
          if (fileInputRef.current) fileInputRef.current.value = ''
        }}
      />

      <PageHeader
        title="Presets"
        eyebrow="Training configs"
        subtitle="Global preset pool. Fork configs to or from a version's private config."
        actions={
          <>
            <button onClick={onImportClick} disabled={busy} className="btn btn-secondary btn-sm">
              {t('presets.importUpload')}
            </button>
            <button onClick={() => setShowImportPathPicker(true)} disabled={busy} className="btn btn-secondary btn-sm">
              {t('presets.importPath')}
            </button>
            <button onClick={handleNew} disabled={busy} className="btn btn-primary btn-sm">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>
              <span>{t('presets.newPresetBtn')}</span>
            </button>
          </>
        }
      />

      <div className="px-7 pb-7" style={{ paddingTop: 0 }}>
        <div className="grid items-start gap-5" style={{ gridTemplateColumns: 'minmax(0, 1fr) 360px' }}>

          {/* ── 列表表格 ── */}
          <div className="card overflow-hidden">
            <div
              className="caption"
              style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1.4fr) minmax(0,1fr) 0.8fr 0.8fr minmax(0,1fr)', gap: 12, padding: '12px 20px', borderBottom: '1px solid var(--border-subtle)' }}
            >
              <span>{t('presets.colName')}</span>
              <span>{t('presets.colOptimizer')}</span>
              <span>{t('presets.colRank')}</span>
              <span>{t('presets.colRes')}</span>
              <span>{t('presets.colUpdated')}</span>
            </div>

            {presets.length === 0 ? (
              <div style={{ padding: 40, textAlign: 'center', color: 'var(--fg-tertiary)', fontSize: 'var(--t-sm)' }}>
                {t('presets.empty')}
              </div>
            ) : (
              presets.map((p, i) => {
                const active = p.name === selected
                const c = configCache[p.name]
                return (
                  <button
                    key={p.name}
                    onClick={() => setSelected(p.name)}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'minmax(0,1.4fr) minmax(0,1fr) 0.8fr 0.8fr minmax(0,1fr)',
                      gap: 12,
                      padding: '15px 20px',
                      width: '100%',
                      textAlign: 'left',
                      alignItems: 'center',
                      border: 'none',
                      borderBottom: i < presets.length - 1 ? '1px solid var(--border-subtle)' : 'none',
                      background: active ? 'var(--accent-soft)' : 'transparent',
                      cursor: 'pointer',
                    }}
                  >
                    <span className="mono" style={{ fontWeight: 600, fontSize: 'var(--t-sm)', color: active ? 'var(--accent)' : 'var(--fg-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.name}</span>
                    <span className="mono" style={{ fontSize: 'var(--t-xs)', color: 'var(--fg-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{cfgOptimizer(c)}</span>
                    <span className="mono" style={{ fontSize: 'var(--t-sm)' }}>{cfgRank(c)}</span>
                    <span className="mono" style={{ fontSize: 'var(--t-sm)' }}>{cfgRes(c)}</span>
                    <span style={{ fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)' }}>{fmtAgo(p.updated_at)}</span>
                  </button>
                )
              })
            )}
          </div>

          {/* ── 详情卡 ── */}
          <div className="card" style={{ padding: 22, position: 'sticky', top: 0 }}>
            {selected && config ? (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
                  <span className="mono" style={{ fontSize: 'var(--t-md)', fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{selected}</span>
                </div>
                {descriptions[selected] && (
                  <p style={{ margin: '0 0 14px', fontSize: 'var(--t-sm)', color: 'var(--fg-secondary)' }}>{descriptions[selected]}</p>
                )}
                <div style={{ display: 'flex', flexDirection: 'column', marginTop: descriptions[selected] ? 0 : 12 }}>
                  {([
                    ['scope', 'global'],
                    ['optimizer', cfgOptimizer(selectedConfig ?? config)],
                    ['rank / alpha', `${cfgRank(selectedConfig ?? config)} / ${cfgAlpha(selectedConfig ?? config)}`],
                    ['resolution', cfgRes(selectedConfig ?? config)],
                    ['updated', fmtAgo(presets.find((p) => p.name === selected)?.updated_at ?? Date.now() / 1000)],
                  ] as [string, string][]).map(([k, v], idx, arr) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 0', borderBottom: idx < arr.length - 1 ? '1px solid var(--border-subtle)' : 'none' }}>
                      <span style={{ fontSize: 'var(--t-sm)', color: 'var(--fg-secondary)' }}>{k}</span>
                      <span className="mono" style={{ fontSize: 'var(--t-sm)', fontWeight: 600 }}>{v}</span>
                    </div>
                  ))}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 18 }}>
                  <button className="btn btn-secondary btn-sm" onClick={() => setEditorOpen(true)} disabled={busy || !config}>
                    {t('presets.editConfig')}
                  </button>
                  <button className="btn btn-secondary btn-sm" onClick={handleDuplicate} disabled={busy || !config}>
                    {t('presets.duplicate')}
                  </button>
                  <button className="btn btn-secondary btn-sm" onClick={() => setExportDialogOpen(true)} disabled={busy || !config}>
                    {t('presets.exportYaml')}
                  </button>
                  <button className="btn btn-ghost btn-sm" onClick={handleDelete} disabled={busy} style={{ color: 'var(--err)' }}>
                    {t('common.delete')}
                  </button>
                </div>
              </>
            ) : (
              <div style={{ padding: '24px 4px', textAlign: 'center', color: 'var(--fg-tertiary)', fontSize: 'var(--t-sm)' }}>
                {t('presets.selectHint')}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── 编辑器模态（New / Edit config） ── */}
      {editorOpen && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-start justify-center"
          style={{ background: 'rgba(23,24,26,0.42)', backdropFilter: 'blur(2px)', paddingTop: '6vh', paddingBottom: '6vh' }}
          onMouseDown={(e) => { if (e.target === e.currentTarget) setEditorOpen(false) }}
        >
          <div
            className="card"
            style={{ width: '92%', maxWidth: 760, maxHeight: '88vh', display: 'flex', flexDirection: 'column', boxShadow: 'var(--sh-xl)', overflow: 'hidden' }}
          >
            {/* header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '18px 24px 16px', borderBottom: '1px solid var(--border-subtle)' }}>
              <h2 style={{ margin: 0, fontSize: 'var(--t-xl)', fontWeight: 700 }}>
                {isNew ? t('presets.newPresetBtn') : <>{t('presets.editPrefix')} · <span className="mono">{selected}</span></>}
              </h2>
              <span style={{ flex: 1 }} />
              <span className="inline-flex items-center gap-0.5 bg-sunken rounded-md p-[3px] text-xs">
                <button
                  type="button"
                  onClick={() => advancedMode && toggleAdvancedMode()}
                  className={`border-none px-3 py-1 rounded-[calc(var(--r-md)-2px)] transition-all duration-100 cursor-pointer ${!advancedMode ? 'bg-surface text-fg-primary font-semibold shadow-sm' : 'bg-transparent text-fg-secondary font-medium hover:text-fg-primary'}`}
                >
                  {t('train.simpleMode')}
                </button>
                <button
                  type="button"
                  onClick={() => !advancedMode && toggleAdvancedMode()}
                  className={`border-none px-3 py-1 rounded-[calc(var(--r-md)-2px)] transition-all duration-100 cursor-pointer ${advancedMode ? 'bg-surface text-fg-primary font-semibold shadow-sm' : 'bg-transparent text-fg-secondary font-medium hover:text-fg-primary'}`}
                >
                  {t('train.advancedMode')}
                </button>
              </span>
              <button onClick={() => setEditorOpen(false)} className="btn btn-ghost btn-sm" aria-label={t('common.cancel')}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M6 6l12 12M18 6 6 18" /></svg>
              </button>
            </div>

            {/* scroll body */}
            <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '18px 24px 22px', display: 'flex', flexDirection: 'column', gap: 16 }}>
              {/* name / description */}
              <div style={{ display: 'flex', gap: 12 }}>
                {isNew ? (
                  <label className="flex flex-col gap-1.5" style={{ flex: 1 }}>
                    <span className="text-sm font-medium text-fg-secondary">{t('presets.presetName')}</span>
                    <input
                      ref={newNameInputRef}
                      autoFocus
                      className="input input-mono font-mono"
                      placeholder="my-training-preset"
                      value={newName}
                      onChange={(e) => { setNewName(e.target.value); setNewNameError('') }}
                      disabled={busy}
                    />
                    {newNameError && <span className="text-xs text-err">{newNameError}</span>}
                  </label>
                ) : (
                  <label className="flex flex-col gap-1.5" style={{ flex: 1 }}>
                    <span className="text-sm font-medium text-fg-secondary">{t('presets.nameReadonly')}</span>
                    <div className="py-2 px-3 rounded-md border border-subtle bg-sunken font-mono text-sm text-fg-primary">{selected}</div>
                  </label>
                )}
                <label className="flex flex-col gap-1.5" style={{ flex: 1.5 }}>
                  <span className="text-sm font-medium text-fg-secondary">{t('presets.description')}</span>
                  <input
                    className="input"
                    placeholder={t('presets.descPlaceholder')}
                    value={descDraft}
                    onChange={(e) => { setDescDraft(e.target.value); setDescDirty(true) }}
                    disabled={busy}
                  />
                </label>
              </div>

              {(droppedFields.length > 0 || defaultedFields.length > 0) && (
                <div className="rounded-md border border-warn bg-warn-soft px-3.5 py-2.5 text-xs text-warn space-y-1">
                  <span className="font-semibold">{t('presets.compatNoticeTitle')}</span>
                  {droppedFields.length > 0 && (
                    <div>{t('presets.droppedFieldsBody')}<code className="ml-1 text-[11px] opacity-80">{droppedFields.join(', ')}</code></div>
                  )}
                  {defaultedFields.length > 0 && (
                    <div>{t('presets.defaultedFieldsBody')}<code className="ml-1 text-[11px] opacity-80">{defaultedFields.join(', ')}</code></div>
                  )}
                </div>
              )}

              {!schema || !config ? (
                <div className="h-[200px]"><ConfigSkeleton variant="flat" label={t('presets.loadingConfig')} /></div>
              ) : (
                <SchemaForm
                  schema={schema}
                  values={config}
                  onChange={setConfig}
                  disabledFields={disabledFields}
                  disabledHints={disabledHints}
                  autoHints={autoHints}
                  fieldSuffixes={fieldSuffixes}
                  advancedMode={advancedMode}
                />
              )}

              {/* TOML preview（折叠） */}
              {config && Object.keys(config).length > 0 && (
                <section className={`rounded-md border border-subtle bg-surface ${tomlOpen ? 'px-3.5 py-2.5' : 'px-3.5 py-1.5'}`}>
                  <button
                    type="button"
                    onClick={() => setTomlOpen((v) => !v)}
                    className="w-full flex items-center gap-2 bg-transparent border-none p-0 cursor-pointer text-left"
                  >
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-info shrink-0" />
                    <span className="caption uppercase tracking-[0.06em] text-xs">{t('presets.tomlPreview')}</span>
                    <span className="flex-1" />
                    {tomlOpen && (
                      <button
                        className="btn btn-ghost btn-sm text-xs"
                        onClick={(e) => {
                          e.stopPropagation()
                          navigator.clipboard.writeText(generateToml(config))
                            .then(() => toast(t('presets.copied'), 'success'))
                            .catch(() => toast(t('presets.copyFailed'), 'error'))
                        }}
                      >{t('common.copy')}</button>
                    )}
                    <span className="text-fg-tertiary">{tomlOpen ? '▾' : '▸'}</span>
                  </button>
                  {tomlOpen && (
                    <pre className="m-0 mt-2.5 p-3 bg-sunken rounded-sm font-mono text-xs text-fg-secondary leading-[1.7] whitespace-pre-wrap break-words max-h-80 overflow-auto">
                      {generateToml(config)}
                    </pre>
                  )}
                </section>
              )}
            </div>

            {/* footer */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, padding: '14px 24px', borderTop: '1px solid var(--border-subtle)' }}>
              <button className="btn btn-secondary" onClick={() => setEditorOpen(false)}>{t('common.cancel')}</button>
              <button className="btn btn-primary" onClick={handleSave} disabled={saveDisabled}>
                {isNew ? t('common.create') : t('common.save')}
              </button>
            </div>
          </div>
        </div>
      )}

      {exportDialogOpen && (
        <PresetExportDialog
          onDownload={() => {
            setExportDialogOpen(false)
            downloadCurrentPreset()
          }}
          onDataExports={() => {
            setExportDialogOpen(false)
            void exportCurrentPresetToDataExports()
          }}
          onCancel={() => setExportDialogOpen(false)}
        />
      )}

      {showImportPathPicker && (
        <PathPicker
          dirOnly={false}
          onClose={() => setShowImportPathPicker(false)}
          onPick={(path) => { void handleImportFromPath(path) }}
        />
      )}

      {conflict && (
        <ImportConflictDialog
          suggestedName={conflict.suggestedName}
          existingNames={presets.map((p) => p.name)}
          onDecide={resolveConflict}
        />
      )}
    </div>
  )
}

function PresetExportDialog({
  onDownload,
  onDataExports,
  onCancel,
}: {
  onDownload: () => void
  onDataExports: () => void
  onCancel: () => void
}) {
  const { t } = useTranslation()
  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel() }}
    >
      <div className="bg-elevated border border-subtle rounded-[var(--r-card)] w-[90%] max-w-[420px] p-6 flex flex-col gap-4 shadow-xl">
        <div>
          <h2 className="m-0 text-lg font-semibold text-fg-primary">{t('presets.exportPresetTitle')}</h2>
          <p className="mt-1 mb-0 text-sm text-fg-secondary">{t('presets.exportPresetHint')}</p>
        </div>
        <button type="button" className="card p-4 text-left hover:border-dim" onClick={onDownload}>
          <div className="font-medium text-fg-primary mb-1">{t('presets.exportDownload')}</div>
          <div className="text-xs text-fg-tertiary">{t('presets.exportDownloadHint')}</div>
        </button>
        <button type="button" className="card p-4 text-left hover:border-dim" onClick={onDataExports}>
          <div className="font-medium text-fg-primary mb-1">{t('presets.exportDataExports')}</div>
          <div className="text-xs text-fg-tertiary">{t('presets.exportDataExportsHint')}</div>
        </button>
        <div className="flex justify-end">
          <button type="button" className="btn btn-secondary" onClick={onCancel}>{t('common.cancel')}</button>
        </div>
      </div>
    </div>
  )
}

// ImportConflictDialog —— 上传 preset 名字撞库时弹三选一（覆盖 / 另存为 / 取消）。
function ImportConflictDialog({
  suggestedName,
  existingNames,
  onDecide,
}: {
  suggestedName: string
  existingNames: string[]
  onDecide: (c: ConflictChoice) => void
}) {
  const { t } = useTranslation()
  const [newName, setNewName] = useState(() => {
    let i = 2
    let cand = `${suggestedName}-${i}`
    while (existingNames.includes(cand)) cand = `${suggestedName}-${++i}`
    return cand
  })
  const [error, setError] = useState('')
  const inputRef = useRef<HTMLInputElement | null>(null)
  useEffect(() => {
    requestAnimationFrame(() => { inputRef.current?.focus(); inputRef.current?.select() })
  }, [])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onDecide({ kind: 'cancel' }) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onDecide])

  const submitSaveAs = (e?: React.FormEvent) => {
    e?.preventDefault()
    const v = newName.trim()
    if (!v) { setError(t('presets.nameRequired')); return }
    if (!PRESET_NAME_RE.test(v)) { setError(t('presets.nameInvalid')); return }
    if (existingNames.includes(v)) { setError(t('presets.nameExists')); return }
    onDecide({ kind: 'saveAs', name: v })
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onDecide({ kind: 'cancel' }) }}
    >
      <form
        onSubmit={submitSaveAs}
        className="bg-elevated border border-subtle rounded-[var(--r-card)] w-[90%] max-w-[480px] p-6 flex flex-col gap-4 shadow-xl"
      >
        <h2 className="m-0 text-lg font-semibold text-fg-primary">
          {t('presets.importConflictTitle', { name: suggestedName })}
        </h2>
        <p className="m-0 text-sm text-fg-secondary">
          {t('presets.importConflictBody')}
        </p>
        <label className="flex flex-col gap-1.5">
          <span className="text-sm text-fg-secondary">{t('presets.importSaveAsLabel')}</span>
          <input
            ref={inputRef}
            className="input input-mono font-mono"
            value={newName}
            onChange={(e) => { setNewName(e.target.value); if (error) setError('') }}
          />
          {error && <span className="text-xs text-err">{error}</span>}
        </label>
        <div className="flex gap-2 justify-end mt-1">
          <button type="button" onClick={() => onDecide({ kind: 'cancel' })} className="btn btn-secondary">
            {t('common.cancel')}
          </button>
          <button
            type="button"
            onClick={() => onDecide({ kind: 'overwrite' })}
            className="btn btn-warn"
            title={t('presets.importOverwriteTitle', { name: suggestedName })}
          >
            {t('presets.importOverwrite')}
          </button>
          <button type="submit" className="btn btn-primary">
            {t('presets.importSaveAs')}
          </button>
        </div>
      </form>
    </div>
  )
}
