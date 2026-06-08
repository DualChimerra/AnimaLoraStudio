import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from 'react'
import type { TFunction } from 'i18next'
import { Trans, useTranslation } from 'react-i18next'
import {
  api,
  DEFAULT_WD14_MODELS,
  type LLMPreset,
  type FlashAttnStatus,
  type XformersStatus,
  type ModelDownloadStatus,
  type ModelsCatalog,
  type Secrets,
  type SecretsPatch,
  type TorchCuTag,
  type TorchStatus,
} from '../../api/client'
import { useDialog } from '../../components/Dialog'
import { InfoButton } from '../../components/InfoButton'
import PageHeader from '../../components/PageHeader'
import { useToast } from '../../components/Toast'
import { useSettingsData } from '../../lib/SettingsData'
import { useSettingsDrawer } from '../../lib/SettingsDrawer'

const MASK = '***'

type Section =
  | 'gelbooru'
  | 'danbooru'
  | 'download'
  | 'huggingface'
  | 'wandb'
  | 'modelscope'
  | 'llm_tagger'
  | 'wd14'
  | 'cltagger'
  | 'models'
  | 'queue'
  | 'generate'
  | 'proxy'

// Settings 现在只有「训练」一组配置（打标 / 测试 / 外观 / 系统 tab 已移除）。
// 右侧 sticky 导航的 section index。
const TRAINING_SECTIONS: { id: string; labelKey: string }[] = [
  { id: 'download-source', labelKey: 'settings.modelSource' },
  { id: 'queue', labelKey: 'settings.queueSchedule' },
  { id: 'pytorch', labelKey: 'settings.torch' },
  { id: 'flash-attn', labelKey: 'settings.flashAttn' },
  { id: 'xformers', labelKey: 'settings.xformers' },
  { id: 'models', labelKey: 'settings.trainingModels' },
]

// fallback 预设：仅在 GET /api/secrets 失败时充当占位，真实 prompt 由后端 builtin
// json 文件提供。命中此 fallback 然后 PUT 回去不会破坏 builtin（后端 validator
// 会再补全 builtin defaults）。
function _makeFallbackPreset(id: string, label: string, output_format: 'json' | 'text', extra: Partial<LLMPreset> = {}): LLMPreset {
  return {
    id,
    label,
    builtin: true,
    base_url: '',
    api_key: '',
    model: '',
    model_ids: [],
    endpoint: 'chat_completions',
    messages: [
      { type: 'text', role: 'system', content: '' },
      { type: 'image', role: 'user', content: '' },
    ],
    output_format,
    temperature: 0.2,
    max_tokens: 700,
    max_side: 1280,
    jpeg_quality: 85,
    max_image_mb: 5,
    timeout: 60,
    max_retries: 3,
    concurrency: 1,
    requests_per_second: 0,
    max_requests_per_minute: 0,
    ...extra,
  }
}

const DEFAULT_LLM_PRESETS: LLMPreset[] = [
  _makeFallbackPreset('style_json', 'Style LoRA JSON', 'json'),
  _makeFallbackPreset('general_json', 'General LoRA JSON', 'json'),
  _makeFallbackPreset('txt_tags', 'TXT tag list', 'json'),
  _makeFallbackPreset('joycaption', 'JoyCaption (vLLM local)', 'text', {
    base_url: 'http://localhost:8000/v1',
    model: 'fancyfeast/llama-joycaption-beta-one-hf-llava',
    temperature: 0.6,
    max_tokens: 300,
  }),
]

const EMPTY: Secrets = {
  gelbooru: {
    user_id: '',
    api_key: '',
    save_tags: false,
    convert_to_png: true,
    remove_alpha_channel: true,
  },
  danbooru: { username: '', api_key: '', account_type: 'free' },
  download: {
    exclude_tags: [],
    parallel_workers: 4,
    api_rate_per_sec: 2,
    cdn_rate_per_sec: 5,
  },
  huggingface: { token: '', endpoint: '' },
  wandb: {
    enabled: false,
    api_key: '',
    project: 'AnimaLoraStudio',
    entity: '',
    base_url: '',
    mode: 'online',
    log_samples: true,
    sample_max_side: 1216,
    sample_every_n_steps: 0,
    upload_model: false,
    upload_model_policy: 'last',
    upload_state_manual: false,
    upload_state_manual_policy: 'last',
    upload_state_auto: false,
    upload_state_auto_policy: 'last',
  },
  modelscope: { token: '' },
  download_source: 'huggingface',
  llm_tagger: {
    current_preset: 'style_json',
    presets: [...DEFAULT_LLM_PRESETS],
  },
  wd14: {
    model_id: 'SmilingWolf/wd-eva02-large-tagger-v3',
    model_ids: [...DEFAULT_WD14_MODELS],
    local_dir: null,
    threshold_general: 0.35,
    threshold_character: 0.85,
    blacklist_tags: [],
    batch_size: 8,
  },
  cltagger: {
    model_id: 'cella110n/cl_tagger',
    model_path: 'cl_tagger_1_02/model.onnx',
    tag_mapping_path: 'cl_tagger_1_02/tag_mapping.json',
    local_dir: null,
    threshold_general: 0.35,
    threshold_character: 0.6,
    add_copyright_tag: true,
    add_meta_tag: false,
    add_model_tag: false,
    add_rating_tag: false,
    add_quality_tag: false,
    blacklist_tags: [],
    batch_size: 8,
  },
  models: { root: null, selected_anima: '1.0', selected_upscaler: '4x-AnimeSharp', auto_sync_paths: true },
  queue: { allow_gpu_during_train: false },
  generate: { preview_every_n_steps: 3, attention_backend: 'auto' },
  proxy: {
    enabled: false,
    http_proxy: '',
    https_proxy: '',
    no_proxy: '',
  }
}

const textInputClass = 'w-full px-2 py-1 outline-none rounded-sm bg-sunken border border-subtle text-sm text-fg-primary focus:border-accent'

const MODEL_DESCRIPTION_KEYS: Record<string, string> = {
  anima_main: 'settings.modelDescriptions.animaMain',
  anima_vae: 'settings.modelDescriptions.animaVae',
  qwen3: 'settings.modelDescriptions.qwen3',
  t5_tokenizer: 'settings.modelDescriptions.t5Tokenizer',
  wd14: 'settings.modelDescriptions.wd14',
  cltagger: 'settings.modelDescriptions.cltagger',
}

function translatedCatalogText(keys: Record<string, string>, id: string, fallback: string | undefined, t: TFunction): string {
  const key = keys[id]
  return key ? t(key, { defaultValue: fallback ?? '' }) : (fallback ?? '')
}

export default function SettingsPage() {
  const { t } = useTranslation()
  // 共享数据层（SettingsDataProvider）：secrets / catalog / SSE / downloadBusy 都在根级常驻，
  // 本组件 mount/unmount（抽屉开关）不再触发重拉。`server` 别名保留是为了让下方
  // 大段表单代码改动最小。
  const {
    secrets: server,
    secretsError,
    setSecrets: setServer,
    catalog,
    catalogError,
    reloadCatalog,
    downloadBusy,
    startDownload,
  } = useSettingsData()
  const [draft, setDraft] = useState<Secrets>(EMPTY)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const { toast } = useToast()
  const drawer = useSettingsDrawer()
  // 右侧 section index 用：sticky nav 的 IntersectionObserver root + 滚动平移容器
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  // 第一次拿到 secrets 时把 draft 同步过来；之后 server 变化（save 后）不再
  // 覆盖 draft，避免抹掉用户的未保存编辑（save 里会自己 setDraft(next)）。
  const draftInitRef = useRef(false)
  useEffect(() => {
    if (server && !draftInitRef.current) {
      setDraft(server)
      draftInitRef.current = true
    }
  }, [server])
  // 数据层 fetch secrets 失败时把错误透出到本组件 error 状态，复用底部错误条。
  useEffect(() => { if (secretsError) setError(secretsError) }, [secretsError])

  const dirty = useMemo(
    () => server !== null && JSON.stringify(server) !== JSON.stringify(draft),
    [server, draft]
  )

  // 抽屉关闭前用这个 ref 询问"是否 dirty"；ref 每次 render 刷新，
  // 注册的函数只挂载一次，避免 effect churn。
  const dirtyRef = useRef(false)
  dirtyRef.current = dirty
  useEffect(() => {
    drawer.registerDirtyGuard(() => dirtyRef.current)
    return () => drawer.registerDirtyGuard(null)
  }, [drawer])

  // 抽屉以 open({ section }) 打开时跳到对应 section（取代旧的 ?section= URL 参数）。
  // sectionRequest 带 nonce，相同 section 重复 open 也会触发 effect 重跑。
  const drawerSectionReq = drawer.sectionRequest
  useEffect(() => {
    if (!drawerSectionReq) return
    const section = drawerSectionReq.section
    const t1 = setTimeout(() => {
      const el = document.getElementById(section)
      el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 50)
    return () => clearTimeout(t1)
  }, [drawerSectionReq])

  const update = <S extends Section, K extends keyof Secrets[S]>(
    section: S,
    key: K,
    value: Secrets[S][K]
  ) => {
    setDraft((prev) => ({
      ...prev,
      [section]: { ...prev[section], [key]: value },
    }))
  }

  /** 更新 Secrets 顶层非对象字段（如 download_source）。 */
  const updateTop = <K extends keyof Secrets>(key: K, value: Secrets[K]) => {
    setDraft((prev) => ({ ...prev, [key]: value }))
  }

  const save = async () => {
    if (!server) return
    const patch = buildPatch(draft, server)
    setSaving(true)
    setError(null)
    try {
      const next = await api.updateSecrets(patch)
      setServer(next)
      setDraft(next)
      // 候选 model_ids 改了之后，catalog 里的 wd14 variants 需要刷新
      void reloadCatalog()
      toast(t('settings.saved'), 'success')
    } catch (e) {
      setError(String(e))
      toast(t('settings.saveFailed'), 'error')
    } finally {
      setSaving(false)
    }
  }

  if (error && !server) {
    return (
      <div className="text-err font-mono text-sm p-4 bg-err-soft rounded-md">
        {error}
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <PageHeader
        title={t('settings.title')}
        sticky
        topRight={drawer.isOpen ? (
          <button
            onClick={() => void drawer.close()}
            title={t('settings.drawerClose')}
            aria-label={t('settings.drawerClose')}
            className="w-7 h-7 grid place-items-center text-fg-tertiary bg-transparent border-none rounded-sm cursor-pointer hover:bg-overlay hover:text-fg-primary transition-colors"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M6 6l12 12M18 6l-12 12" />
            </svg>
          </button>
        ) : undefined}
        actions={
          <button
            onClick={save}
            disabled={!dirty || saving}
            className={dirty ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm'}
          >
            {saving ? t('common.saving') : t('common.save')}
          </button>
        }
      />

      <div ref={scrollContainerRef} className="p-6 pb-12 flex-1 overflow-y-auto">
      <div className="grid gap-10 max-w-[1400px]" style={{ gridTemplateColumns: 'minmax(0,1fr) 200px' }}>
      <div className="flex flex-col gap-8 min-w-0">

      {error && (
        <div className="p-3 rounded-md bg-err-soft border border-err text-err text-sm font-mono">
          {error}
        </div>
      )}

      <SettingsSection id="download-source" title={t('settings.modelSource')}>
        <SettingsField
          label={t('settings.downloadSource')}
          helpTooltip={
            <p>{t('settings.downloadSourceHelp')}</p>
          }
        >
          <DownloadSourceSelect
            value={draft.download_source}
            onChange={(v) => updateTop('download_source', v)}
          />
        </SettingsField>

        {/* 下方按当前下载源条件渲染对应凭证配置。HF/ModelScope token 都保留在
         * secrets 里（即便切换源也不丢失），只是 UI 一次只露面一份。 */}
        {draft.download_source === 'huggingface' ? (
          <>
            <SettingsField
              label="token"
              helpTooltip={
                <p>{t('settings.hfTokenHelp')}</p>
              }
            >
              <SensitiveInput
                value={draft.huggingface.token}
                serverValue={server?.huggingface.token ?? ''}
                onChange={(v) => update('huggingface', 'token', v)}
              />
            </SettingsField>
            <SettingsField
              label="endpoint"
              helpTooltip={<p>{t('settings.hfEndpointHelp')}</p>}
            >
              <HFEndpointSelect
                value={draft.huggingface.endpoint}
                onChange={(v) => update('huggingface', 'endpoint', v)}
              />
            </SettingsField>
          </>
        ) : (
          <SettingsField
            label="token"
            helpTooltip={
              <>
                <p>{t('settings.modelscopeTokenHelp')}</p>
                <p><Trans i18nKey="settings.modelscopeInstallHelp" components={{ code: <code /> }} /></p>
              </>
            }
          >
            <SensitiveInput
              value={draft.modelscope.token}
              serverValue={server?.modelscope.token ?? ''}
              onChange={(v) => update('modelscope', 'token', v)}
            />
          </SettingsField>
        )}
      </SettingsSection>

      <SettingsSection id="queue" title={t('settings.queueSchedule')}>
        <SettingsField label={t('settings.allowGpuDuringTrain')}>
          <div className="flex items-center gap-3">
            <Bool value={draft.queue.allow_gpu_during_train} onChange={(v) => update('queue', 'allow_gpu_during_train', v)} />
            <span className="text-xs text-warn">
              {t('settings.allowGpuDuringTrainHint')}
            </span>
          </div>
        </SettingsField>
      </SettingsSection>

      <PyTorchSection />

      <FlashAttentionSection />

      <XformersSection />

      <ModelsSection
        catalog={catalog}
        busy={downloadBusy}
        start={startDownload}
        reloadCatalog={reloadCatalog}
        catalogError={catalogError}
        t={t}
      />

    </div>

    <SectionIndex sections={TRAINING_SECTIONS} scrollContainer={scrollContainerRef} />
    </div>
    </div>
    </div>
  )
}

// ── Section / Field ────────────────────────────────────────────────────────

function SettingsSection({
  id, title, headerExtras, children,
}: {
  id?: string
  title: string
  headerExtras?: React.ReactNode  // 可选 slot：渲染在 h2 右侧（紧贴），给 ⓘ tooltip 之类用
  children: React.ReactNode
}) {
  const titleEl = <h2 className="text-sm font-semibold text-fg-primary">{title}</h2>
  return (
    <section id={id} className="rounded-md border border-subtle bg-surface p-4 flex flex-col gap-3 scroll-mt-24">
      {headerExtras ? (
        <div className="flex items-center gap-2 mb-0.5">
          {titleEl}
          {headerExtras}
        </div>
      ) : (
        <div className="mb-0.5">{titleEl}</div>
      )}
      {children}
    </section>
  )
}

/**
 * 右侧 sticky section 目录。基于 IntersectionObserver 在 scrollContainer 视口内
 * 跟踪当前可见 section，并提供点击平滑滚动。
 *
 * rootMargin 调整为顶部 -20%、底部 -70%：让"当前可见"判定集中在视口偏上区域，
 * 滚动时高亮跟随更自然（用户视线在 viewport 上 1/3 处）。
 */
function SectionIndex({
  sections,
  scrollContainer,
}: {
  sections: { id: string; labelKey: string }[]
  scrollContainer: RefObject<HTMLDivElement>
}) {
  const { t } = useTranslation()
  const [active, setActive] = useState<string>(sections[0]?.id ?? '')

  useEffect(() => {
    // 切换 tab 后重置 active 到第一条
    setActive(sections[0]?.id ?? '')
  }, [sections])

  useEffect(() => {
    const root = scrollContainer.current
    if (!root || sections.length === 0) return
    // jsdom（vitest 环境）没有 IntersectionObserver；非浏览器环境直接跳过。
    if (typeof IntersectionObserver === 'undefined') return
    const observers: IntersectionObserver[] = []
    // 收集 (id, top) 用来在 onIntersect 时挑当前最靠上的可见 section
    const visible = new Set<string>()
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) visible.add(e.target.id)
          else visible.delete(e.target.id)
        }
        // 按 sections 顺序取第一个可见的作为 active
        const next = sections.find((s) => visible.has(s.id))
        if (next) setActive(next.id)
      },
      { root, rootMargin: '-20% 0px -70% 0px', threshold: 0 },
    )
    sections.forEach((s) => {
      const el = document.getElementById(s.id)
      if (el) obs.observe(el)
    })
    observers.push(obs)
    return () => observers.forEach((o) => o.disconnect())
  }, [sections, scrollContainer])

  const onJump = (id: string) => {
    const el = document.getElementById(id)
    if (!el) return
    el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setActive(id)
  }

  return (
    <aside className="hidden lg:block">
      <nav className="sticky top-4 flex flex-col gap-0.5">
        <div className="caption mb-2 px-2">{t('settings.pageIndex')}</div>
        {sections.map((s) => (
          <button
            key={s.id}
            onClick={() => onJump(s.id)}
            className={`text-left text-xs px-2 py-1.5 rounded-sm transition-colors border-l-2 ${
              active === s.id
                ? 'border-accent text-accent bg-accent-soft/40'
                : 'border-transparent text-fg-tertiary hover:text-fg-secondary hover:bg-overlay/40'
            }`}
          >
            {t(s.labelKey)}
          </button>
        ))}
      </nav>
    </aside>
  )
}

function SettingsField({ label, desc, helpTooltip, children }: {
  label: string
  desc?: string
  /** 可选 ⓘ tooltip slot，渲染在 label 旁边。中长说明（≥20 字 / 详细用法）
   *  适合放这里，避免 inline desc 把字段名行撑得过长。一般和 desc 二选一。 */
  helpTooltip?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="grid grid-cols-[240px_1fr] gap-3 items-start">
      <div className="flex flex-col gap-0.5 pt-1.5">
        <div className="flex items-center gap-2 min-w-0">
          <label className="text-xs text-fg-secondary font-mono leading-none">{label}</label>
          {helpTooltip && <InfoButton>{helpTooltip}</InfoButton>}
        </div>
        {desc && <p className="text-[10px] text-fg-tertiary m-0 leading-snug">{desc}</p>}
      </div>
      <div className="min-w-0">{children}</div>
    </div>
  )
}

function Bool({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <input
      type="checkbox"
      checked={value}
      onChange={(e) => onChange(e.target.checked)}
      className="w-4 h-4"
      style={{ accentColor: 'var(--accent)' }}
    />
  )
}

function SensitiveInput({ value, serverValue, onChange }: {
  value: string; serverValue: string; onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  const [localValue, setLocalValue] = useState(value)

  useEffect(() => {
    setLocalValue(value)
  }, [value])

  const masked = localValue === MASK

  const handleBlur = () => {
    if (localValue !== value) {
      onChange(localValue)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.currentTarget.blur()
    }
  }

  return (
    <input
      type="password"
      value={masked ? '' : localValue}
      placeholder={serverValue === MASK ? t('settings.sensitiveSavedPlaceholder') : ''}
      onChange={(e) => setLocalValue(e.target.value || MASK)}
      onBlur={handleBlur}
      onKeyDown={handleKeyDown}
      autoComplete="new-password"
      data-lpignore="true"
      data-1p-ignore
      data-form-type="other"
      className={textInputClass}
    />
  )
}


// ── HFEndpointSelect ────────────────────────────────────────────────────────
//
// HF 模型下载 endpoint 选择器：preset + 自定义 URL 输入。
// 0.8.2 hotfix：hf-mirror.com preset 暂时隐藏（服务端 redirect 改动后所有
// huggingface_hub 版本均失败，详见 docs/todo/hf-mirror-recheck.md）。endpoint
// 字段本身仍接受任意 URL，用户可通过「自定义 URL」粘贴 hf-mirror / sjtug /
// 腾讯镜像 / 自建反代。复活后把 preset 加回来即可。

const HF_ENDPOINT_PRESETS: { value: string; label: string; hintKey: string }[] = [
  { value: '', label: 'huggingface.co', hintKey: 'settings.hfOfficialHint' },
  { value: '__custom__', label: 'Custom URL...', hintKey: 'settings.hfCustomHint' },
]

function HFEndpointSelect({ value, onChange }: {
  value: string; onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  const isPreset = HF_ENDPOINT_PRESETS.some(p => p.value !== '__custom__' && p.value === value)
  const [mode, setMode] = useState<'preset' | 'custom'>(isPreset ? 'preset' : 'custom')
  const selectedPreset = isPreset
    ? value
    : (mode === 'custom' ? '__custom__' : '')

  return (
    <div className="flex flex-col gap-1.5">
      <select
        value={selectedPreset}
        onChange={(e) => {
          const v = e.target.value
          if (v === '__custom__') {
            setMode('custom')
            // 不清当前值，让用户在下方输入
          } else {
            setMode('preset')
            onChange(v)
          }
        }}
        className={`${textInputClass} max-w-md`}
      >
        {HF_ENDPOINT_PRESETS.map(p => (
          <option key={p.value} value={p.value}>
            {p.label}{p.hintKey ? ` — ${t(p.hintKey)}` : ''}
          </option>
        ))}
      </select>
      {mode === 'custom' && (
        <input
          type="text"
          value={value && !isPreset ? value : ''}
          placeholder="https://your-mirror.example.com"
          onChange={(e) => onChange(e.target.value.trim())}
          className={`${textInputClass} max-w-md`}
        />
      )}
    </div>
  )
}

// ── DownloadSourceSelect ────────────────────────────────────────────────────

function DownloadSourceSelect({ value, onChange }: {
  value: string; onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`${textInputClass} max-w-xs`}
    >
      <option value="huggingface">{t('settings.downloadSourceHuggingface')}</option>
      <option value="modelscope">{t('settings.downloadSourceModelscope')}</option>
    </select>
  )
}

// 顶层非 object 字段（string / number / bool），直接比较后塞入 patch。
const TOP_LEVEL_SCALARS: (keyof Secrets)[] = ['download_source']

function buildPatch(draft: Secrets, server: Secrets): SecretsPatch {
  const out: Record<string, unknown> = {}
  for (const key of Object.keys(draft) as (keyof Secrets)[]) {
    if (TOP_LEVEL_SCALARS.includes(key)) {
      if (draft[key] !== server[key]) out[key] = draft[key]
      continue
    }
    const sub: Record<string, unknown> = {}
    const d = draft[key] as unknown as Record<string, unknown>
    const s = server[key] as unknown as Record<string, unknown>
    for (const k of Object.keys(d)) {
      const dv = d[k]
      const sv = s[k]
      if (dv === MASK) continue
      if (JSON.stringify(dv) !== JSON.stringify(sv)) sub[k] = dv
    }
    if (Object.keys(sub).length) out[key] = sub
  }
  return out as SecretsPatch
}

// ── Models Section ─────────────────────────────────────────────────────────

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function ModelsSection({ catalog, busy, start, reloadCatalog, catalogError, t }: {
  catalog: ModelsCatalog | null
  busy: Set<string>
  start: (model_id: string, variant?: string) => Promise<void>
  reloadCatalog: () => Promise<void>
  catalogError: string | null
  t: TFunction
}) {
  const { toast } = useToast()
  const [rootDraft, setRootDraft] = useState<string>('')
  const [serverRoot, setServerRoot] = useState<string | null>(null)
  const [savingRoot, setSavingRoot] = useState(false)
  const [selectedAnima, setSelectedAnima] = useState<string>('1.0')
  const [autoSyncPaths, setAutoSyncPaths] = useState<boolean>(true)
  const [savingAutoSync, setSavingAutoSync] = useState(false)
  const [secretsLoaded, setSecretsLoaded] = useState(false)

  // 一次性拉一份 secrets 取 models.root + selected_anima + auto_sync_paths
  // （这几项走独立 PUT，不进 SettingsPage 的全局 dirty 流程）。catalog 由父级注入。
  useEffect(() => {
    void api.getSecrets().then((sec) => {
      setServerRoot(sec.models?.root ?? null)
      setSelectedAnima(sec.models?.selected_anima ?? '1.0')
      setAutoSyncPaths(sec.models?.auto_sync_paths ?? true)
      setSecretsLoaded(true)
    }).catch(() => { setSecretsLoaded(true) })
  }, [])

  // secrets + catalog 都到位后，把输入框预填成「已保存值」或「实际默认绝对路径」。
  // 用 prev !== '' 当作"已初始化 / 用户已编辑"的标志，避免覆盖用户输入。
  useEffect(() => {
    if (!secretsLoaded || !catalog) return
    setRootDraft((prev) => (prev !== '' ? prev : (serverRoot ?? catalog.models_root ?? '')))
  }, [secretsLoaded, catalog, serverRoot])

  const pickAnima = async (variant: string) => {
    if (variant === selectedAnima) return
    setSelectedAnima(variant)
    try {
      await api.updateSecrets({ models: { selected_anima: variant } })
      toast(t('settings.mainModelSelected', { name: variant }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
      void reloadCatalog()
    }
  }

  const saveRoot = async () => {
    const v = rootDraft.trim()
    setSavingRoot(true)
    try {
      await api.updateSecrets({ models: { root: v ? v : null } })
      toast(v ? t('settings.modelRootSaved', { path: v }) : t('settings.modelRootDefault'), 'success')
      setServerRoot(v ? v : null)
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setSavingRoot(false)
    }
  }

  const saveAutoSync = async (next: boolean) => {
    setSavingAutoSync(true)
    const prev = autoSyncPaths
    setAutoSyncPaths(next)
    try {
      await api.updateSecrets({ models: { auto_sync_paths: next } })
      toast(next ? t('settings.autoSyncPathsOn') : t('settings.autoSyncPathsOff'), 'success')
    } catch (e) {
      setAutoSyncPaths(prev)
      toast(String(e), 'error')
    } finally {
      setSavingAutoSync(false)
    }
  }

  const rootDirty = rootDraft.trim() !== (serverRoot ?? '')
  const error = catalogError

  return (
    <SettingsSection id="models" title={t('settings.trainingModelsOneClick')}>
      <SettingsField label={t('settings.modelsRoot')}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input
            type="text"
            value={rootDraft}
            onChange={(e) => setRootDraft(e.target.value)}
            className={`${textInputClass} flex-1`}                                  />
          <button onClick={saveRoot} disabled={!rootDirty || savingRoot} className="btn btn-primary btn-sm"
            title={rootDirty ? t('settings.savePathConfig') : t('settings.notModified')}>
            {savingRoot ? t('common.saving') : t('settings.savePath')}
          </button>
          <button onClick={() => setRootDraft(serverRoot ?? (catalog?.models_root ?? ''))} disabled={!rootDirty || savingRoot}
            className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm"
            style={{ opacity: !rootDirty ? 0.3 : 1 }}
          >↻</button>
        </div>
      </SettingsField>

      <SettingsField
        label={t('settings.autoSyncPathsLabel')}
        helpTooltip={<p>{t('settings.autoSyncPathsHelp')}</p>}
      >
        <label className="flex items-center gap-2 pt-1.5">
          <input
            type="checkbox"
            checked={autoSyncPaths}
            onChange={(e) => void saveAutoSync(e.target.checked)}
            disabled={savingAutoSync}
            style={{ height: 16, width: 16 }}
          />
        </label>
      </SettingsField>

      {error && <div className="text-err text-xs font-mono">{error}</div>}
      {!catalog ? (
        <p className="text-fg-tertiary text-xs">{t('settings.loadingModelCatalog')}</p>
      ) : (
        <div className="flex flex-col gap-2">
          {/* Anima 主模型 */}
          <ModelGroupCard
            title={catalog.anima_main.name}
            helpTooltip={
              <>
                <p><Trans i18nKey="settings.repoHelp" values={{ desc: translatedCatalogText(MODEL_DESCRIPTION_KEYS, 'anima_main', catalog.anima_main.description, t), repo: catalog.anima_main.repo }} components={{ code: <code /> }} /></p>
                <p><Trans i18nKey="settings.defaultTransformerHelp" components={{ strong: <strong /> }} /></p>
              </>
            }
          >
            <ul className="list-none m-0 p-0 flex flex-col gap-1">
              {catalog.anima_main.variants.map((v) => {
                const key = `anima_main:${v.variant}`
                const dl = catalog.downloads[key]
                const isSel = v.variant === selectedAnima
                const canSelect = v.exists && dl?.status !== 'running'
                return (
                  <li key={v.variant} className={`flex items-center gap-2 text-xs px-1.5 py-1 rounded-sm ${
                    isSel ? 'bg-accent-soft border border-accent' : 'bg-transparent border border-transparent'
                  }`}>
                    <input type="radio" name="anima_variant" checked={isSel} disabled={!canSelect}
                      onChange={() => void pickAnima(v.variant)}
                      className="shrink-0"
                      style={{ accentColor: 'var(--accent)' }}
                      title={canSelect ? t('settings.selectDefaultMainModel') : v.exists ? t('settings.downloadInProgress') : t('settings.downloadRequiredFirst')}
                    />
                    <code className="font-mono text-fg-primary w-32 shrink-0">{v.variant}</code>
                    <ModelStatusBadge exists={v.exists} size={v.size} status={dl?.status} />
                    <span style={{ flex: 1 }} />
                    <DownloadButton exists={v.exists} status={dl?.status} busy={busy.has(key)} onClick={() => void start('anima_main', v.variant)} />
                  </li>
                )
              })}
            </ul>
          </ModelGroupCard>

          {/* VAE */}
          <ModelGroupCard title={catalog.anima_vae.name}>
            <div className="flex items-center gap-2 text-xs">
              <span className="text-fg-tertiary">{translatedCatalogText(MODEL_DESCRIPTION_KEYS, 'anima_vae', catalog.anima_vae.description, t)} · <code>{catalog.anima_vae.repo}</code></span>
              <span style={{ flex: 1 }} />
              <ModelStatusBadge exists={catalog.anima_vae.exists} size={catalog.anima_vae.size} status={catalog.downloads.anima_vae?.status} />
              <DownloadButton exists={catalog.anima_vae.exists} status={catalog.downloads.anima_vae?.status} busy={busy.has('anima_vae')} onClick={() => void start('anima_vae')} />
            </div>
          </ModelGroupCard>

          {/* Qwen3 + T5（CLTagger 已挪到「打标」tab） */}
          {(['qwen3', 't5_tokenizer'] as const).map((id) => {
            const m = catalog[id]
            const dl = catalog.downloads[id]
            const allExist = m.files.every((f) => f.exists)
            const totalSize = m.files.reduce((s, f) => s + f.size, 0)
            return (
              <ModelGroupCard key={id} title={m.name}>
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-fg-tertiary">{translatedCatalogText(MODEL_DESCRIPTION_KEYS, id, m.description, t)} · <code>{m.repo}</code></span>
                  <span style={{ flex: 1 }} />
                  <ModelStatusBadge exists={allExist} size={totalSize} status={dl?.status} fileCount={m.files.length} existsCount={m.files.filter((f) => f.exists).length} />
                  <DownloadButton exists={allExist} status={dl?.status} busy={busy.has(id)} onClick={() => void start(id)} />
                </div>
              </ModelGroupCard>
            )
          })}

          {/* 下载日志 */}
          {Object.values(catalog.downloads).filter((d) => d.status === 'running' || d.status === 'failed').length > 0 && (
            <details className="text-xs">
              <summary className="cursor-pointer text-fg-tertiary">
                {t('settings.downloadLogs', { n: Object.values(catalog.downloads).filter((d) => d.status === 'running' || d.status === 'failed').length })}
              </summary>
              <div className="mt-1 flex flex-col gap-2">
                {Object.values(catalog.downloads).map((d) => (
                  <div key={d.key} className="rounded-sm border border-subtle bg-sunken p-2">
                    <div className="flex items-center gap-2 mb-1">
                      <code className="font-mono text-fg-secondary">{d.key}</code>
                      <ModelStatusBadge exists={d.status === 'done'} size={0} status={d.status} />
                      {d.message && <span className="text-err overflow-hidden text-ellipsis whitespace-nowrap">{d.message}</span>}
                    </div>
                    <pre className="text-xs font-mono text-fg-tertiary max-h-32 overflow-auto whitespace-pre-wrap m-0">
                      {d.log_tail.join('\n') || t('settings.emptyLog')}
                    </pre>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </SettingsSection>
  )
}

function ModelGroupCard({
  title, helpTooltip, children,
}: {
  title: string
  helpTooltip?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="rounded-sm border border-subtle bg-sunken p-2.5">
      <h4 className="text-xs font-semibold text-fg-primary mb-1.5 flex items-center gap-2">
        <span>{title}</span>
        {helpTooltip && <InfoButton>{helpTooltip}</InfoButton>}
      </h4>
      {children}
    </div>
  )
}

function ModelStatusBadge({ exists, size, status, fileCount, existsCount }: {
  exists: boolean; size: number; status?: ModelDownloadStatus['status']; fileCount?: number; existsCount?: number
}) {
  const { t } = useTranslation()
  if (status === 'running') {
    return <StatusLabel bg="bg-warn-soft" fg="text-warn" text={t('settings.downloadInProgress')} pulse />
  }
  if (status === 'failed') {
    return <StatusLabel bg="bg-err-soft" fg="text-err" text={t('status.failed')} />
  }
  if (exists) {
    return <StatusLabel bg="bg-ok-soft" fg="text-ok" text={`✓ ${fmtBytes(size)}${fileCount !== undefined ? ` (${existsCount}/${fileCount})` : ''}`} />
  }
  if (fileCount !== undefined && existsCount! > 0) {
    return <StatusLabel bg="bg-warn-soft" fg="text-warn" text={t('settings.partialFiles', { exists: existsCount, total: fileCount })} />
  }
  return <StatusLabel bg="bg-overlay" fg="text-fg-tertiary" text={t('settings.notDownloaded')} />
}

function StatusLabel({ bg, fg, text, pulse }: { bg: string; fg: string; text: string; pulse?: boolean }) {
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded-sm font-mono ${bg} ${fg}`}
      style={pulse ? { animation: 'pulse 1.5s infinite' } : undefined}
    >{text}</span>
  )
}

function DownloadButton({ exists, status, busy, onClick }: {
  exists: boolean; status?: ModelDownloadStatus['status']; busy: boolean; onClick: () => void
}) {
  const { t } = useTranslation()
  const running = status === 'running' || busy
  if (running) {
    return <button disabled className="btn btn-secondary btn-sm" style={{ opacity: 0.5 }}>...</button>
  }
  return (
    <button onClick={onClick} className={exists ? 'btn btn-secondary btn-sm' : 'btn btn-primary btn-sm'}
      title={exists ? t('settings.redownloadTitle') : t('common.download')}>
      {exists ? t('settings.redownload') : t('settings.downloadAction')}
    </button>
  )
}

// ── PyTorch Section（训练 tab）──────────────────────────────────────────────
//
// 已有 venv 用户的「一键修」入口。PR-4 启动期会 warn「检测到 GPU 但 torch 是
// CPU 版」并给 pip 命令；这里把命令 UI 化，普通用户不用进终端。
//
// 三种状态：
// - cuda_available=True               → ✓ 一切 OK（折叠默认；提供「换 CUDA 版本」高级选项）
// - is_cpu_with_gpu=True               → 红色误装提示 + 显著「重装为 CUDA」主按钮
// - is_cuda_build_unavailable=True     → 黄色驱动警告（pip 修不了，给文档链接）

function PyTorchSection() {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [status, setStatus] = useState<TorchStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const s = await api.getTorchStatus()
      setStatus(s)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const reinstall = async (target: 'auto' | TorchCuTag) => {
    const tag = target === 'auto' ? status?.recommended_cu_tag ?? '?' : target
    // 注册 → 用户 Ctrl+C 重启 → launcher 进程跑 pip。Windows 上 torch.pyd 被
    // 当前 server 进程锁住，没法直接 replace；只能 defer 到 launcher。
    if (!(await dialog.confirm(
      t('settings.confirmRegisterTorch', { tag }),
      { tone: 'warn', okText: t('settings.registerRequest') },
    ))) return
    setBusy(true)
    try {
      const result = await api.reinstallTorch(target)
      // 后端已写 marker，server 进程没真装；提示用户去重启
      toast(result.message, 'success')
    } catch (e) {
      toast(t('settings.registerFailed', { error: String(e) }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const hasIssue = !!error || (status && (status.is_cpu_with_gpu || status.is_cuda_build_unavailable || !status.installed))
  const statusOk = status?.cuda_available && !error
  const statusLabel = error
    ? t('settings.loadFailedShort')
    : !status
      ? t('settings.loadingStatus')
      : !status.installed
        ? t('settings.notInstalledShort')
        : status.is_cpu_with_gpu
          ? t('settings.cpuBuildMisinstalled')
          : !status.cuda_available && status.cuda_build !== 'cpu'
            ? t('settings.cudaUnavailableDriver')
            : status.cuda_available
              ? `CUDA ✓ ${status.cuda_build}`
              : `CPU ${status.cuda_build}`

  return (
    <details id="pytorch" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">PyTorch</h2>
        <span className="text-xs text-fg-tertiary">{t('settings.trainingCoreDependency')}</span>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : status?.is_cpu_with_gpu ? 'text-err' : 'text-warn'}`}>
          {statusLabel}
        </span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !status && <div className="text-xs text-fg-tertiary">{t('settings.loadingStatus')}</div>}

        {status && (<>
          {/* 当前状态卡 */}
          <div className="rounded-sm border border-subtle bg-sunken p-2 flex flex-col gap-1 text-xs">
            <div className="flex gap-4 flex-wrap">
              <span className="text-fg-tertiary">torch: <code className="text-fg-secondary font-mono">{status.version ?? t('settings.notInstalledParen')}</code></span>
              {status.cuda_build && (
                <span className="text-fg-tertiary">build: <code className="text-fg-secondary font-mono">{status.cuda_build}</code></span>
              )}
              {status.cuda_available && status.device_name && (
                <span className="text-fg-tertiary">GPU: <code className="text-fg-secondary font-mono">{status.device_name}</code></span>
              )}
            </div>
            <div className="flex gap-4 flex-wrap">
              <span className="text-fg-tertiary">
                {t('settings.driverLabel')}:{' '}
                <code className="text-fg-secondary font-mono">
                  {status.cuda_detect.driver_version ?? t('settings.notDetected')}
                </code>
              </span>
              {status.cuda_detect.gpu_name && !status.cuda_available && (
                <span className="text-fg-tertiary">
                  {t('settings.systemGpu')}:{' '}
                  <code className="text-fg-secondary font-mono">{status.cuda_detect.gpu_name}</code>
                </span>
              )}
            </div>
          </div>

          {/* 误装：CPU torch + 有 GPU */}
          {status.is_cpu_with_gpu && (
            <div className="rounded-sm border border-err bg-err-soft px-2 py-1.5 text-err text-xs">
              <Trans
                i18nKey="settings.torchCpuWithGpuWarning"
                values={{ tag: status.recommended_cu_tag }}
                components={{ code: <code className="font-mono" /> }}
              />
            </div>
          )}

          {/* CUDA build 但运行时不可用：驱动 / WSL 问题 */}
          {status.is_cuda_build_unavailable && (
            <div className="rounded-sm border border-warn bg-warn-soft px-2 py-1.5 text-warn text-xs">
              <Trans
                i18nKey="settings.torchCudaUnavailableWarning"
                components={{ code: <code className="font-mono" /> }}
              />
            </div>
          )}

          {/* 操作按钮 */}
          <div className="flex gap-1.5 items-center flex-wrap">
            <button
              onClick={() => void reinstall('auto')}
              disabled={busy || !status.cuda_detect.available}
              className={status.is_cpu_with_gpu ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm'}
              title={status.cuda_detect.available
                ? t('settings.autoSelect', { tag: status.recommended_cu_tag })
                : t('settings.noNvidiaDriverCannotCuda')}
            >
              {busy ? t('settings.installing') : status.is_cpu_with_gpu
                ? t('settings.reinstallCudaBuild', { tag: status.recommended_cu_tag })
                : t('settings.reinstallAuto', { tag: status.recommended_cu_tag })}
            </button>
            <button onClick={() => void refresh()} disabled={busy}
              className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm">↻</button>
            <button type="button" onClick={() => setAdvancedOpen(!advancedOpen)}
              className="btn btn-ghost btn-sm text-xs text-fg-tertiary ml-auto">
              {advancedOpen ? '▾' : '▸'} {t('settings.advancedManualCuda')}
            </button>
          </div>

          {/* 手动选版本 */}
          {advancedOpen && (
            <div className="flex flex-col gap-1.5 pt-2 border-t border-subtle text-xs">
              <p className="text-fg-tertiary m-0">
                {t('settings.manualCudaHint')}
              </p>
              <div className="flex gap-1.5 flex-wrap">
                {(['cu128', 'cu126', 'cu124', 'cu118', 'cpu'] as const).map((tag) => (
                  <button
                    key={tag}
                    onClick={() => void reinstall(tag)}
                    disabled={busy}
                    className={`btn btn-secondary btn-sm ${
                      status.cuda_build === tag ? 'border-accent' : ''
                    }`}
                    title={
                      tag === 'cpu'
                        ? t('settings.installCpuBuildHint')
                        : t('settings.installCudaBuildHint', { tag })
                    }
                  >
                    {tag}{status.cuda_build === tag ? ' ✓' : ''}
                  </button>
                ))}
              </div>
            </div>
          )}
        </>)}
      </div>
    </details>
  )
}

// ── Flash Attention Section（训练 tab）─────────────────────────────────────
//
// 训练加速的可选优化。装好 flash_attn 后启动期会自动 set_flash_attn_enabled(True)。
// 本组件给 UI 一键装 wheel 的能力，复用 PR-7a 的 service：状态 + GitHub 候选 + 安装。
//
// 设计要点：
// - install 是同步 pip（几分钟），用 confirm() + busy 状态防误触
// - Python ABI 不一致的 wheel（usable=false）灰显，但保留「强制安装」按钮（
//   极少数情况用户可能在 ABI 兼容子集里跑）
// - GitHub API 限流时 candidates=[] + fetch_error，给手动 URL 输入兜底

function FlashAttentionSection() {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [status, setStatus] = useState<FlashAttnStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [candidatesOpen, setCandidatesOpen] = useState(false)
  const [manualUrl, setManualUrl] = useState('')
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const s = await api.getFlashAttnStatus()
      setStatus(s)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const install = async (url: string | null) => {
    const msg = url ? t('settings.confirmInstallFlashUrl') : t('settings.confirmInstallFlashAuto')
    if (!(await dialog.confirm(msg, { tone: 'warn', okText: t('settings.startInstall') }))) return
    setBusy(true)
    try {
      const result = await api.installFlashAttn(url)
      toast(t('settings.flashAttnInstalled', { version: result.version ?? '?' }), 'success')
      await refresh()
    } catch (e) {
      toast(t('settings.installFailed', { error: String(e) }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const env = status?.env
  const candidates = status?.candidates ?? []
  const fetchError = status?.fetch_error ?? null
  const usable = candidates.filter((c) => c.usable)
  const bestCandidate = usable[0] ?? null
  const hasIssue = !!error || (status && !status.installed)
  const canAutoInstall = !!env?.torch_tag && !!env?.platform && usable.length > 0

  const statusLabel = error
    ? t('settings.loadFailedShort')
    : !status
      ? t('settings.loadingStatus')
      : status.installed
        ? t('settings.installedVersion', { version: status.version ?? '?' })
        : t('settings.notInstalledShort')
  const statusOk = status?.installed && !error

  return (
    <details id="flash-attn" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">Flash Attention</h2>
        <span className="text-xs text-fg-tertiary">{t('settings.trainingAccelerationOptional')}</span>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : 'text-warn'}`}>{statusLabel}</span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !status && <div className="text-xs text-fg-tertiary">{t('settings.loadingStatus')}</div>}

        {status && env && (<>
          {/* 环境信息 */}
          <div className="rounded-sm border border-subtle bg-sunken p-2 flex flex-col gap-1 text-xs">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-fg-tertiary shrink-0">flash_attn:</span>
              <code className="font-mono text-fg-primary">
                {status.installed ? `v${status.version ?? '?'}` : t('settings.notInstalledParen')}
              </code>
              {status.installed && <StatusLabel bg="bg-ok-soft" fg="text-ok" text={t('settings.installed')} />}
            </div>
            <div className="flex gap-4 flex-wrap">
              <span className="text-fg-tertiary">Python: <code className="text-fg-secondary font-mono">{env.python_tag}</code></span>
              <span className="text-fg-tertiary">CUDA: <code className="text-fg-secondary font-mono">{env.cuda_tag ?? t('settings.notDetected')}</code></span>
              <span className="text-fg-tertiary">PyTorch: <code className="text-fg-secondary font-mono">{env.torch_tag ?? t('settings.notDetected')}</code></span>
              <span className="text-fg-tertiary">{t('settings.platform')}: <code className="text-fg-secondary font-mono">{env.platform ?? t('settings.unsupported')}</code></span>
            </div>
          </div>

          {/* GitHub API 失败 */}
          {fetchError && (
            <div className="rounded-sm border border-err bg-err-soft px-2 py-1.5 text-err text-xs">
              {t('settings.githubApiFailed')}
              <code className="block mt-0.5 break-all">{fetchError}</code>
            </div>
          )}

          {/* 没匹配 wheel */}
          {!canAutoInstall && !fetchError && env.platform && env.torch_tag && (
            <div className="rounded-sm border border-warn bg-warn-soft px-2 py-1.5 text-warn text-xs">
              {t('settings.noWheelForPython', { python: env.python_tag })}
            </div>
          )}

          {/* 操作按钮 */}
          <div className="flex gap-1.5 items-center flex-wrap">
            <button
              onClick={() => void install(null)}
              disabled={busy || !canAutoInstall}
              className="btn btn-primary btn-sm"
              title={canAutoInstall
                ? t('settings.autoSelect', { tag: bestCandidate?.name ?? '' })
                : t('settings.noWheelManual')}
            >
              {busy ? t('settings.installing') : status.installed ? t('settings.reinstallAutoMatch') : t('settings.autoMatchInstall')}
            </button>
            <button onClick={() => void refresh()} disabled={busy}
              className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm">↻</button>
            <button type="button" onClick={() => setCandidatesOpen(!candidatesOpen)}
              className="btn btn-ghost btn-sm text-xs text-fg-tertiary ml-auto">
              {candidatesOpen ? '▾' : '▸'} {t('settings.candidateWheels', { n: usable.length })}
            </button>
          </div>

          {/* 候选列表 + 手动 URL */}
          {candidatesOpen && (
            <div className="flex flex-col gap-2 pt-2 border-t border-subtle">
              {candidates.length === 0 ? (
                <p className="text-xs text-fg-tertiary m-0">{t('settings.wheelQueryFailed')}</p>
              ) : (
                <ul className="list-none m-0 p-0 flex flex-col gap-1">
                  {candidates.map((c) => (
                    <li key={c.url} className={`flex items-start gap-2 text-xs px-2 py-1.5 rounded-sm border ${
                      c.usable ? 'border-subtle bg-sunken' : 'border-transparent bg-transparent opacity-50'
                    }`}>
                      <div className="flex flex-col gap-0.5 flex-1 min-w-0">
                        <code className="font-mono text-fg-primary text-[11px] break-all">{c.name}</code>
                        {c.notes.map((n, i) => (
                          <span key={i} className="text-warn text-[10px]">{n}</span>
                        ))}
                      </div>
                      <button
                        onClick={() => void install(c.url)}
                        disabled={busy}
                        className={c.usable ? 'btn btn-primary btn-sm shrink-0' : 'btn btn-secondary btn-sm shrink-0'}
                        title={c.usable ? t('settings.installWheel') : t('settings.wheelAbiIncompatible')}
                      >
                        {c.usable ? t('settings.installAction') : t('settings.forceInstall')}
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              <div className="flex flex-col gap-1 pt-1 border-t border-subtle">
                <p className="text-xs text-fg-tertiary m-0">{t('settings.manualUrl')}</p>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    value={manualUrl}
                    onChange={(e) => setManualUrl(e.target.value)}
                    placeholder="https://github.com/.../flash_attn-...whl"
                    className={`${textInputClass} flex-1`}
                  />
                  <button
                    onClick={() => { if (manualUrl.trim()) void install(manualUrl.trim()) }}
                    disabled={busy || !manualUrl.trim()}
                    className="btn btn-secondary btn-sm shrink-0"
                  >{t('settings.install')}</button>
                </div>
              </div>
            </div>
          )}
        </>)}
      </div>
    </details>
  )
}

// ── xformers Section（训练 tab）─────────────────────────────────────────────
//
// 简化版 attention 加速（替代 flash_attn 的另一选项）。xformers 走 PyPI 直装，
// 不需要 flash_attn 那种 GitHub 候选 wheel 列表。失败时给 stderr 让用户排错。

function XformersSection() {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [status, setStatus] = useState<XformersStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const s = await api.getXformersStatus()
      setStatus(s)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const install = async () => {
    if (
      !(await dialog.confirm(
        t('settings.confirmInstallXformers'),
        { tone: 'warn', okText: t('settings.startInstall') },
      ))
    ) return
    setBusy(true)
    try {
      const r = await api.installXformers()
      toast(t('settings.xformersInstalled', { version: r.version ?? '?' }), 'success')
      await refresh()
    } catch (e) {
      toast(t('settings.installFailed', { error: String(e) }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const statusLabel = error
    ? t('settings.loadFailedShort')
    : !status
      ? t('settings.loadingStatus')
      : status.installed
        ? t('settings.installedVersion', { version: status.version ?? '?' })
        : t('settings.notInstalledShort')
  const statusOk = status?.installed && !error
  const hasIssue = !!error

  return (
    <details id="xformers" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">xformers</h2>
        <span className="text-xs text-fg-tertiary">{t('settings.xformersSubtitle')}</span>
        <InfoButton>
          <p><Trans i18nKey="settings.xformersHelp1" components={{ strong: <strong />, code: <code /> }} /></p>
          <p>{t('settings.xformersHelp2')}</p>
          <p>{t('settings.xformersHelp3')}</p>
        </InfoButton>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : 'text-warn'}`}>{statusLabel}</span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !status && <div className="text-xs text-fg-tertiary">{t('settings.loadingStatus')}</div>}

        {status && (<>
          <div className="rounded-sm border border-subtle bg-sunken p-2 flex items-center gap-2 text-xs">
            <span className="text-fg-tertiary shrink-0">xformers:</span>
            <code className="font-mono text-fg-primary">
              {status.installed ? `v${status.version ?? '?'}` : t('settings.notInstalledParen')}
            </code>
            {status.installed && <StatusLabel bg="bg-ok-soft" fg="text-ok" text={t('settings.installed')} />}
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => void install()}
              disabled={busy}
              className="btn btn-primary btn-sm"
            >
              {busy
                ? t('settings.installing')
                : status.installed
                  ? t('settings.reinstallAutoMatchPlain')
                  : t('settings.installAutoMatchPlain')}
            </button>
            <button
              onClick={() => void refresh()}
              disabled={busy}
              className="btn btn-ghost btn-sm"
              title={t('settings.refreshStatus')}
            >↻</button>
          </div>
        </>)}
      </div>
    </details>
  )
}

