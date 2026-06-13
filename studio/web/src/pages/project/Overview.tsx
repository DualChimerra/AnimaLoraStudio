/** 项目详情页 — Design v2 实装（pvt-detail-v2.jsx）
 *
 *  顶部：Identity strip（glyph + title/version/status + meta caption）
 *  → 横向 VersionRail（pill 行）
 *  → 5 状态 StatusBanner（preparing / training / completed / failed / canceled）
 *  → Tabs (详情 / Tasks / Output)
 *  → 详情 = 2+3 不对称 grid（训练集 hero + 标签分布 hero / 分辨率 + 长宽比 + 正则集）
 *
 *  TopBar (面包屑 + sys stats) 不实装 —— 已被 sidebar/全局区覆盖。
 *  Live 训练进度 (step/total/ETA) 不实装 —— 需 SSE/monitor state 整合，留 follow-up。
 *  "复制配置开新版本" / "调小 batch 重训" 需新后端 API，渲染为占位按钮 toast 提示。
 */
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate, useOutletContext } from 'react-router-dom'
import {
  api,
  type CurationView,
  type ProjectDetail,
  type Task,
  type TaskOutputs,
  type Version,
  type VersionPhase,
} from '../../api/client'
import PageHeader from '../../components/PageHeader'
import VersionStatusBadge from '../../components/VersionStatusBadge'
import BarHistogram from '../../components/BarHistogram'
import { TranslatedTag } from '../../components/tagDisplay/TranslatedTag'
import ImageGrid, { type ImageGridItem } from '../../components/ImageGrid'
import ImagePreviewModal from '../../components/ImagePreviewModal'
import { OutputsTab } from '../QueueDetail'
import { arBucket } from '../../lib/aspectRatio'
import { computePixelHist } from '../../lib/pixelBins'
import { useProjectCtx } from '../../context/ProjectContext'
import { useToast } from '../../components/Toast'

type OverviewTab = 'details' | 'tasks' | 'output'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
  onCreateVersion: (forkFromVid?: number) => void
  creatingVersionBusy: boolean
}

// ── helpers ──────────────────────────────────────────────────────────────

function fmtAgo(ts: number | null | undefined): string {
  if (!ts) return '—'
  const sec = Math.max(0, Date.now() / 1000 - ts)
  if (sec < 60) return 'just now'
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`
  return `${Math.floor(sec / 86400)}d ago`
}

// ── PipelineStrip（redesign 原型：圆形序号 + 连接线 + 标签在下） ──────────

function PipelineStrip({ project, version }: { project: ProjectDetail; version: Version | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const isPreparing = version?.status === 'preparing'
  const ci = version ? PHASE_ORDER_TIMELINE.findIndex((p) => p.id === version.phase) : -1
  const steps: Array<{ key: string; label: string; n: string | null; phaseId: VersionPhase | null }> = [
    { key: 'download', label: t('nav.download'), n: null, phaseId: null },
    ...PHASE_ORDER_TIMELINE.map((p) => ({
      key: PHASE_TO_STEP_LOCAL[p.id], label: t(p.key), n: p.n, phaseId: p.id as VersionPhase | null,
    })),
  ]
  const continueTarget = isPreparing && version
    ? PHASE_ORDER_TIMELINE.find((p) => p.id === version.phase) ?? null
    : null
  return (
    <div className="card" style={{ padding: '18px 22px' }}>
      <div className="caption" style={{ marginBottom: 14 }}>
        Pipeline{version ? ` · ${version.label}` : ''}
      </div>
      <div style={{ display: 'flex', alignItems: 'flex-start' }}>
        {steps.map((s, idx) => {
          const isProject = s.phaseId === null
          const pi = s.phaseId ? PHASE_ORDER_TIMELINE.findIndex((p) => p.id === s.phaseId) : -1
          const done = !isProject && isPreparing && pi < ci
          const current = !isProject && isPreparing && pi === ci
          const allowed = isProject
            ? true
            : version != null && canGoVersionPhase(version, s.phaseId!)
          const to = isProject
            ? `/projects/${project.id}/download`
            : version ? `/projects/${project.id}/v/${version.id}/${s.key}` : null
          return (
            <div key={s.key} style={{ display: 'contents' }}>
              {idx > 0 && (
                <div style={{
                  flex: 1, height: 2, marginTop: 15, borderRadius: 2,
                  background: done || current ? 'var(--accent)' : 'var(--border-default)',
                  opacity: done ? 1 : current ? 0.8 : 0.5,
                }} />
              )}
              <button
                onClick={() => { if (allowed && to) navigate(to) }}
                disabled={!allowed || !to}
                style={{
                  display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 7,
                  background: 'none', border: 'none',
                  cursor: allowed && to ? 'pointer' : 'default',
                  opacity: allowed ? 1 : 0.4, flex: 'none', padding: 0,
                }}
              >
                <span style={{
                  width: 32, height: 32, borderRadius: 100, display: 'grid', placeItems: 'center',
                  fontSize: 12, fontWeight: 700, fontFamily: 'var(--font-mono)',
                  background: done ? 'var(--ok-soft)' : current ? 'var(--accent)' : 'var(--bg-overlay)',
                  color: done ? 'var(--ok)' : current ? 'var(--accent-fg)' : 'var(--fg-tertiary)',
                  boxShadow: current ? '0 0 0 4px var(--accent-soft)' : 'none',
                }}>{done ? '✓' : (s.n ?? '·')}</span>
                <span style={{
                  fontSize: 'var(--t-xs)',
                  color: current ? 'var(--fg-primary)' : 'var(--fg-tertiary)',
                  fontWeight: current ? 600 : 500, whiteSpace: 'nowrap',
                }}>{s.label}</span>
              </button>
            </div>
          )
        })}
      </div>
      {isPreparing && version && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap',
          marginTop: 16, paddingTop: 14, borderTop: '1px solid var(--border-subtle)',
        }}>
          <div>
            <div className="caption" style={{ marginBottom: 3 }}>{t('overview.banner.metaCurrentPhase')}</div>
            <div className="mono" style={{ fontSize: 'var(--t-sm)', fontWeight: 600 }}>
              {t(continueTarget?.key ?? 'nav.curate')}
            </div>
          </div>
          {version.stats && (
            <div>
              <div className="caption" style={{ marginBottom: 3 }}>{t('overview.banner.metaTagged')}</div>
              <div className="mono" style={{ fontSize: 'var(--t-sm)', fontWeight: 600 }}>
                {version.stats.tagged_image_count} / {version.stats.train_image_count}
              </div>
            </div>
          )}
          <span style={{ flex: 1 }} />
          {continueTarget && (
            <button
              onClick={() => navigate(`/projects/${project.id}/v/${version.id}/${PHASE_TO_STEP_LOCAL[continueTarget.id]}`)}
              className="btn btn-primary btn-sm"
            >{t('overview.banner.continueLabel')} {t(continueTarget.key)} →</button>
          )}
        </div>
      )}
    </div>
  )
}

// ── VersionRowCard（redesign 原型：版本列表行卡） ─────────────────────────

function VersionRowCard({
  project, v, selected, onView,
}: {
  project: ProjectDetail
  v: Version
  selected: boolean
  onView: (vid: number) => void
}) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const ctx = useProjectCtx()
  const isActive = v.id === project.active_version_id
  const meta: Array<[string, string]> = [
    ['train', v.stats ? `${v.stats.train_image_count} imgs` : '—'],
    ['reg', v.stats ? `${v.stats.reg_image_count} imgs` : '—'],
    ['phase', v.status === 'preparing' ? t(PHASE_ORDER_TIMELINE.find((p) => p.id === v.phase)?.key ?? 'nav.curate') : t(`versionStatus.${v.status}`)],
    ['created', fmtAgo(v.created_at)],
  ]
  return (
    <div
      className="card"
      onClick={() => onView(v.id)}
      style={{
        padding: 18, cursor: 'pointer',
        borderColor: selected ? 'var(--accent-veil)' : 'var(--border-subtle)',
        boxShadow: selected ? '0 0 0 1px var(--accent-veil)' : 'none',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12, flexWrap: 'wrap' }}>
        <span className="mono" style={{ fontSize: 'var(--t-md)', fontWeight: 700 }}>{v.label}</span>
        <VersionStatusBadge status={v.status} />
        {isActive && <span className="badge badge-accent">active</span>}
        <span style={{ flex: 1 }} />
        <button
          className="btn btn-secondary btn-sm"
          onClick={(e) => { e.stopPropagation(); navigate(`/projects/${project.id}/v/${v.id}/train`) }}
        >{t('overview.versionRow.openTrain')}</button>
        {!isActive && (
          <button
            className="btn btn-ghost btn-sm"
            onClick={(e) => { e.stopPropagation(); ctx?.onSelectVersion(v.id); onView(v.id) }}
          >{t('overview.versionRow.activate')}</button>
        )}
      </div>
      {v.note && <p style={{ margin: '0 0 12px', fontSize: 'var(--t-sm)', color: 'var(--fg-secondary)' }}>{v.note}</p>}
      <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
        {meta.map(([k, val]) => (
          <div key={k}>
            <div className="caption" style={{ marginBottom: 3 }}>{k}</div>
            <div className="mono" style={{ fontSize: 'var(--t-sm)', fontWeight: 600 }}>{val}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── StatusBanner shared bits ─────────────────────────────────────────────

const bannerMetaRow: React.CSSProperties = {
  display: 'flex', gap: 18, flexWrap: 'wrap',
  paddingTop: 10, marginTop: 4,
  borderTop: '1px dashed var(--border-subtle)',
}
const bannerActions: React.CSSProperties = {
  display: 'flex', gap: 6, marginTop: 12, marginLeft: 'auto',
  justifyContent: 'flex-end', flexWrap: 'wrap',
}

function BannerMeta({ k, v }: { k: string; v: string | number }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <span style={{
        fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
        color: 'var(--fg-tertiary)', textTransform: 'uppercase', letterSpacing: '0.06em',
      }}>{k}</span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--t-sm)', color: 'var(--fg-primary)', fontWeight: 600 }}>{v}</span>
    </div>
  )
}

function BannerShell({
  tint, iconChar, iconColor, iconPulse, title, sub, children,
}: {
  tint: 'err' | 'warn' | 'accent' | 'ok'
  iconChar: string
  iconColor: string
  iconPulse?: boolean
  title: string
  sub?: string
  children: ReactNode
}) {
  const tintMap = {
    err:    { bg: 'var(--err-soft)',    border: 'color-mix(in srgb, var(--err) 30%, transparent)' },
    warn:   { bg: 'var(--warn-soft)',   border: 'color-mix(in srgb, var(--warn) 25%, transparent)' },
    accent: { bg: 'var(--accent-soft)', border: 'var(--accent-veil)' },
    ok:     { bg: 'var(--ok-soft)',     border: 'color-mix(in srgb, var(--ok) 25%, transparent)' },
  }
  const tCfg = tintMap[tint]
  return (
    <div className="banner-shell" style={{
      background: tCfg.bg,
      border: '1px solid ' + tCfg.border,
      borderRadius: 'var(--r-lg)',
      display: 'flex', flexDirection: 'column', gap: 4,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span className="banner-shell-icon" style={{
          flexShrink: 0,
          borderRadius: '50%',
          background: 'var(--bg-surface)',
          border: '1px solid ' + tCfg.border,
          display: 'grid', placeItems: 'center',
          color: iconColor, fontWeight: 700,
          animation: iconPulse ? 'pulse 1.6s infinite' : 'none',
        }}>{iconChar}</span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 'var(--t-md)', fontWeight: 600, color: 'var(--fg-primary)' }}>{title}</div>
          {sub && <div style={{ fontSize: 'var(--t-xs)', color: 'var(--fg-secondary)', marginTop: 1 }}>{sub}</div>}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, paddingTop: 6 }}>
        {children}
      </div>
    </div>
  )
}

function BannerProgress({
  now, total, running, muted, fail,
}: { now: number; total: number; running?: boolean; muted?: boolean; fail?: boolean }) {
  const pct = total > 0 ? Math.min(100, (now / total) * 100) : 0
  const color = fail ? 'var(--err)' : muted ? 'var(--fg-disabled)' : 'var(--accent)'
  return (
    <div style={{
      height: 6, borderRadius: 'var(--r-pill)',
      background: 'var(--bg-sunken)',
      overflow: 'hidden', position: 'relative',
    }}>
      <div style={{
        width: `${pct}%`, height: '100%',
        background: color,
        animation: running ? 'pulse 2s infinite' : 'none',
        borderRadius: 'var(--r-pill)',
      }}/>
      {muted && (
        <div style={{
          position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
          background: 'repeating-linear-gradient(45deg, transparent, transparent 4px, rgba(255,255,255,0.04) 4px, rgba(255,255,255,0.04) 8px)',
        }}/>
      )}
    </div>
  )
}

const PHASE_ORDER_TIMELINE: { id: VersionPhase; n: string; key: string }[] = [
  { id: 'curating',      n: '1', key: 'nav.curate' },
  { id: 'preprocessing', n: '2', key: 'nav.preprocess' },
  { id: 'editing',       n: '3', key: 'nav.tagEdit' },
  { id: 'regularizing',  n: '4', key: 'nav.reg' },
  { id: 'ready',         n: '5', key: 'nav.train' },
]

// ── StatusBanner ─────────────────────────────────────────────────────────

function StatusBanner({
  projectId, version, latestTask, onOpenOutput,
}: {
  projectId: number
  version: Version
  latestTask: Task | null
  /** "下载" CTA 切到下方 [Output] tab */
  onOpenOutput: () => void
}) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { toast } = useToast()
  const ctx = useProjectCtx()
  const taskId = latestTask?.id

  // 拉 task outputs —— completed 状态时 version.output_lora_path 可能为空（早期
  // 训练 supervisor 未回填），用 task outputs.files (is_lora) 兜底找产物名。
  const [taskOutputs, setTaskOutputs] = useState<TaskOutputs | null>(null)
  useEffect(() => {
    if (!taskId || version.status !== 'completed') { setTaskOutputs(null); return }
    let cancelled = false
    void api.getTaskOutputs(taskId)
      .then((res) => { if (!cancelled) setTaskOutputs(res) })
      .catch(() => { if (!cancelled) setTaskOutputs(null) })
    return () => { cancelled = true }
  }, [taskId, version.status])

  const goLog = () => taskId && navigate(`/queue/${taskId}#log`)
  const goMonitor = () => taskId && navigate(`/queue/${taskId}#monitor`)

  const fmtTime = (ts: number | null | undefined) =>
    ts ? new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false, dateStyle: 'short', timeStyle: 'short' }) : '—'

  if (version.status === 'canceled') {
    const cancelReason = latestTask?.error_msg || t('overview.banner.canceledReasonDefault')
    return (
      <BannerShell
        tint="err" iconChar="⊘" iconColor="var(--fg-tertiary)"
        title={`${version.label} · ${t('versionStatus.canceled')}`}
        sub={cancelReason}
      >
        <div style={{ ...bannerMetaRow, alignItems: 'center' }}>
          <BannerMeta k={t('overview.banner.metaTime')} v={fmtTime(latestTask?.finished_at)} />
          <span style={{ flex: 1 }} />
          {taskId && <button onClick={goLog} className="btn btn-ghost btn-sm">{t('overview.banner.viewLog')} →</button>}
          <button
            onClick={() => ctx && void ctx.onDeleteVersion(version.id)}
            className="btn btn-secondary btn-sm"
          >{t('overview.banner.deleteVersion')}</button>
          <button
            onClick={() => ctx?.onCreateVersion(version.id)}
            className="btn btn-primary btn-sm"
          >+ {t('overview.banner.forkConfigNew')}</button>
        </div>
      </BannerShell>
    )
  }

  if (version.status === 'failed') {
    const reason = version.last_failure_reason || latestTask?.error_msg || t('overview.banner.failedReasonDefault')
    return (
      <BannerShell
        tint="err" iconChar="!" iconColor="var(--err)"
        title={`${version.label} · ${t('overview.banner.failedTitle')}`}
        sub={reason}
      >
        <div style={{ ...bannerMetaRow, alignItems: 'center' }}>
          <BannerMeta k={t('overview.banner.metaTime')} v={fmtTime(latestTask?.finished_at)} />
          <span style={{ flex: 1 }} />
          {taskId && <button onClick={goLog} className="btn btn-ghost btn-sm">{t('overview.banner.viewLog')} →</button>}
          <button
            onClick={() => ctx && void ctx.onDeleteVersion(version.id)}
            className="btn btn-secondary btn-sm"
          >{t('overview.banner.deleteVersion')}</button>
          <button
            onClick={() => {
              ctx?.onCreateVersion(version.id)
              toast(t('overview.banner.smallerBatchHint'), 'info')
            }}
            className="btn btn-primary btn-sm"
          >{t('overview.banner.smallerBatchRetry')} ↻</button>
        </div>
      </BannerShell>
    )
  }

  if (version.status === 'training') {
    const startedAt = latestTask?.started_at
    return (
      <BannerShell
        tint="accent" iconChar="●" iconColor="var(--accent)" iconPulse
        title={`${version.label} · ${t('versionStatus.training')}`}
        sub={startedAt ? `${t('overview.banner.startedAt')} ${fmtTime(startedAt)}` : undefined}
      >
        <BannerProgress now={0} total={1} running />
        <div style={bannerMetaRow}>
          <BannerMeta k={t('overview.banner.metaStarted')} v={fmtTime(startedAt)} />
          {latestTask?.is_pausable && (
            <BannerMeta k={t('overview.banner.metaPausable')} v={t('overview.banner.yes')} />
          )}
        </div>
        <div style={bannerActions}>
          {latestTask?.is_pausable && (
            <button
              onClick={() => taskId && api.pauseTask(taskId).catch((e) => toast(String(e), 'error'))}
              className="btn btn-ghost btn-sm"
            >{t('overview.banner.pause')}</button>
          )}
          {taskId && (
            <button
              onClick={() => api.cancelTask(taskId).catch((e) => toast(String(e), 'error'))}
              className="btn btn-secondary btn-sm"
            >{t('overview.banner.cancelTraining')}</button>
          )}
          {taskId && <button onClick={goMonitor} className="btn btn-primary btn-sm">{t('overview.banner.openMonitor')} →</button>}
        </div>
      </BannerShell>
    )
  }

  if (version.status === 'completed') {
    // 测试中加载用完整 path：version 字段优先；空时用 task outputs 第一个 LoRA 文件 兜底
    const loraFromTask = taskOutputs?.files.find((f) => f.is_lora)?.name ?? null
    const loraPathForTest = version.output_lora_path
      || (taskOutputs?.output_dir && loraFromTask ? `${taskOutputs.output_dir}/${loraFromTask}` : null)
    return (
      <BannerShell
        tint="ok" iconChar="✓" iconColor="var(--ok)"
        title={`${version.label} · ${t('versionStatus.completed')}`}
        sub={fmtTime(latestTask?.finished_at)}
      >
        <div style={{ ...bannerMetaRow, alignItems: 'center' }}>
          {taskId && <BannerMeta k={t('overview.banner.metaTaskId')} v={`#${taskId}`} />}
          {taskOutputs && (
            <BannerMeta
              k={t('overview.banner.metaLoraCount')}
              v={taskOutputs.files.filter((f) => f.is_lora).length}
            />
          )}
          <span style={{ flex: 1 }} />
          <button
            onClick={() => ctx?.onCreateVersion(version.id)}
            className="btn btn-ghost btn-sm"
          >{t('overview.banner.copyAsNew')}</button>
          <button
            onClick={() => {
              if (!loraPathForTest) {
                toast(t('overview.banner.noArtifact'), 'error')
                return
              }
              const sp = new URLSearchParams()
              sp.set('lora', loraPathForTest)
              sp.set('projectId', String(projectId))
              sp.set('versionId', String(version.id))
              navigate(`/tools/generate?${sp.toString()}`)
            }}
            className="btn btn-secondary btn-sm"
          >{t('overview.banner.loadInTest')} →</button>
          <button
            onClick={onOpenOutput}
            className="btn btn-primary btn-sm"
          >{t('overview.banner.downloadLora')} ↓</button>
        </div>
      </BannerShell>
    )
  }

  // preparing — 状态叙事由 PipelineStrip（圆形步骤条 + Continue CTA）承担，
  // banner 只负责 task 态（training / failed / completed / canceled）。
  return null
}

/** phase enum → URL step key（StatusBanner 内用，独立于 sidebar 的同名 map）。 */
const PHASE_TO_STEP_LOCAL: Record<VersionPhase, string> = {
  curating:      'curate',
  preprocessing: 'preprocess',
  editing:       'edit',
  regularizing:  'reg',
  ready:         'train',
}

/** cursor 校验：preparing 态下只允许 cursor 及之前的 phase（cursor+1 也禁，
 *  推进必须走 banner 的 "继续 X →" 按钮，那里会调 advance API 校验完成条件）。
 *  非 preparing 态（已训练 / 训练中 / 终态）所有 phase 都允许跳（回看历史）。 */
function canGoVersionPhase(version: Version | null, phase: VersionPhase): boolean {
  if (!version) return false
  if (version.status !== 'preparing') return true
  const cursorIdx = PHASE_ORDER_TIMELINE.findIndex((p) => p.id === version.phase)
  const targetIdx = PHASE_ORDER_TIMELINE.findIndex((p) => p.id === phase)
  return targetIdx <= cursorIdx
}

// ── HeroCard / 详情 card 通用 shell ──────────────────────────────────────

function HeroCard({
  title, count, countSub, action, phase, children,
}: {
  title: string
  count?: number | null
  countSub?: string
  /** disabled 时按钮 opacity 0.4 + cursor not-allowed，点击不触发 onClick（cursor 校验失败用） */
  action?: { label: string; onClick: () => void; disabled?: boolean }
  phase?: string
  children: ReactNode
}) {
  return (
    <div style={{
      padding: 16,
      background: 'var(--bg-surface)',
      border: '1px solid var(--border-subtle)',
      borderRadius: 'var(--r-lg)',
      display: 'flex', flexDirection: 'column', gap: 12,
      height: '100%', minHeight: 0,
    }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
        <h3 style={{ margin: 0, fontSize: 'var(--t-sm)', fontWeight: 600, color: 'var(--fg-primary)' }}>{title}</h3>
        <span style={{ flex: 1 }} />
        {count != null && (
          <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 4, fontFamily: 'var(--font-mono)' }}>
            <span style={{ fontSize: 'var(--t-lg)', fontWeight: 600, color: 'var(--fg-primary)' }}>{count}</span>
            {countSub && <span style={{ fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)' }}>{countSub}</span>}
          </span>
        )}
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10, minHeight: 0, overflow: 'hidden' }}>{children}</div>
      {action && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6, paddingTop: 8,
          borderTop: '1px dashed var(--border-subtle)',
        }}>
          {phase && (
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
              color: 'var(--fg-tertiary)', textTransform: 'uppercase', letterSpacing: '0.06em', flex: 1,
            }}>{phase}</span>
          )}
          <button
            onClick={() => { if (!action.disabled) action.onClick() }}
            disabled={action.disabled}
            style={{
              padding: '4px 10px',
              fontSize: 'var(--t-xs)', color: 'var(--fg-primary)',
              background: 'var(--bg-sunken)', border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--r-sm)',
              cursor: action.disabled ? 'not-allowed' : 'pointer',
              opacity: action.disabled ? 0.4 : 1,
              fontWeight: 500,
            }}
          >{action.label} →</button>
        </div>
      )}
    </div>
  )
}

// ── TrainSetCard (hero) ──────────────────────────────────────────────────
// 文件夹 chips 放 header 右边（跟 Curation 训练集 panel 同款）；body 用
// ImageGrid 渲染当前 folder 的图，点击放大走 ImagePreviewModal。

const EMPTY_SELECTED: Set<string> = new Set()

function TrainSetCard({ project, version }: { project: ProjectDetail; version: Version | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [view, setView] = useState<CurationView | null>(null)
  const [selectedFolder, setSelectedFolder] = useState<string>('all')
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)

  useEffect(() => {
    if (!version) { setView(null); return }
    let cancelled = false
    void api.getCuration(project.id, version.id)
      .then((res) => { if (!cancelled) setView(res) })
      .catch(() => { if (!cancelled) setView(null) })
    return () => { cancelled = true }
  }, [project.id, version])

  const folders = view?.folders ?? []
  const folderCounts: Record<string, number> = useMemo(() => {
    if (!view) return {}
    const out: Record<string, number> = {}
    for (const f of view.folders) out[f] = (view.right[f] ?? []).length
    return out
  }, [view])
  const total = view?.train_total ?? version?.stats?.train_image_count ?? 0

  // 当前选中 folder 的图，转换为 ImageGridItem[]
  const items = useMemo<Array<ImageGridItem & { folder: string; pureName: string }>>(() => {
    if (!view || !version) return []
    const out: Array<ImageGridItem & { folder: string; pureName: string }> = []
    const list = selectedFolder === 'all' ? view.folders : [selectedFolder]
    for (const folder of list) {
      const arr = view.right[folder] ?? []
      for (const it of arr) {
        out.push({
          name: `${folder}/${it.name}`,
          pureName: it.name,
          folder,
          thumbUrl: api.versionThumbUrl(project.id, version.id, 'train', it.name, folder, 256),
        })
      }
    }
    return out
  }, [view, version, project.id, selectedFolder])

  // 预览大图 src（1600 大小）
  const previewItem = previewIdx != null ? items[previewIdx] : null
  const previewSrc = previewItem && version
    ? api.versionThumbUrl(project.id, version.id, 'train', previewItem.pureName, previewItem.folder, 1600)
    : ''

  const actionDisabled = !canGoVersionPhase(version, 'curating')
  const phaseLine = `${t('nav.download')} → ${t('nav.preprocess')} → ${t('nav.curate')}`

  return (
    <div style={{
      padding: 16,
      background: 'var(--bg-surface)',
      border: '1px solid var(--border-subtle)',
      borderRadius: 'var(--r-lg)',
      display: 'flex', flexDirection: 'column', gap: 12,
      height: '100%', minHeight: 0,
    }}>
      {/* Header: title + folder chips on right */}
      <div className="flex items-center gap-3 flex-wrap">
        <h3 className="m-0 text-sm font-semibold" style={{ color: 'var(--fg-primary)' }}>
          {t('overview.detail.folders')}
        </h3>
        <span className="flex-1" />
        {folders.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <FolderChip
              label={t('overview.detail.allFolders')}
              count={total}
              active={selectedFolder === 'all'}
              onClick={() => setSelectedFolder('all')}
            />
            {folders.map((f) => (
              <FolderChip
                key={f}
                label={f}
                count={folderCounts[f] ?? 0}
                active={selectedFolder === f}
                onClick={() => setSelectedFolder(f)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Body: ImageGrid 或 empty */}
      <div className="flex-1 min-h-0">
        {!version || items.length === 0 ? (
          <p className="m-0 text-xs text-fg-tertiary italic">{t('overview.detail.emptyCurate')}</p>
        ) : (
          <ImageGrid
            items={items}
            selected={EMPTY_SELECTED}
            onSelect={() => { /* read-only */ }}
            clickMode="activate"
            onActivate={(name) => setPreviewIdx(items.findIndex((i) => i.name === name))}
            onPreview={(name) => setPreviewIdx(items.findIndex((i) => i.name === name))}
            ariaLabel="overview-train-grid"
          />
        )}
      </div>

      {/* Action row */}
      {version && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6, paddingTop: 8,
          borderTop: '1px dashed var(--border-subtle)',
        }}>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
            color: 'var(--fg-tertiary)', textTransform: 'uppercase', letterSpacing: '0.06em', flex: 1,
          }}>{phaseLine}</span>
          <button
            onClick={() => { if (!actionDisabled) navigate(`/projects/${project.id}/v/${version.id}/curate`) }}
            disabled={actionDisabled}
            style={{
              padding: '4px 10px',
              fontSize: 'var(--t-xs)', color: 'var(--fg-primary)',
              background: 'var(--bg-sunken)', border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--r-sm)',
              cursor: actionDisabled ? 'not-allowed' : 'pointer',
              opacity: actionDisabled ? 0.4 : 1,
              fontWeight: 500,
            }}
          >{t('nav.curate')} · {t('overview.detail.reorganize')} →</button>
        </div>
      )}

      {/* Preview modal */}
      {previewItem && previewIdx != null && (
        <ImagePreviewModal
          src={previewSrc}
          caption={previewItem.name}
          hasPrev={previewIdx > 0}
          hasNext={previewIdx < items.length - 1}
          onClose={() => setPreviewIdx(null)}
          onPrev={() => setPreviewIdx((i) => (i != null && i > 0 ? i - 1 : i))}
          onNext={() => setPreviewIdx((i) => (i != null && i < items.length - 1 ? i + 1 : i))}
        />
      )}
    </div>
  )
}

function FolderChip({
  label, count, active, onClick,
}: { label: string; count: number; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`px-2 py-0.5 rounded-md font-mono transition-colors ${
        active
          ? 'border border-accent bg-accent-soft text-accent'
          : 'border border-dim bg-surface text-fg-secondary hover:bg-overlay'
      }`}
    >
      {label}
      <span className="text-fg-tertiary"> ({count})</span>
    </button>
  )
}

// ── TagDistCard (hero) ───────────────────────────────────────────────────

function TagDistCard({ project, version }: { project: ProjectDetail; version: Version | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const triggerWord = version?.trigger_word ?? ''
  const [tags, setTags] = useState<Array<{ tag: string; n: number }>>([])
  const [uniqueTotal, setUniqueTotal] = useState(0)

  useEffect(() => {
    if (!version) { setTags([]); setUniqueTotal(0); return }
    let cancelled = false
    void api.listCaptionsFull(project.id, version.id)
      .then((res) => {
        if (cancelled) return
        const counter = new Map<string, number>()
        for (const it of res.items) {
          for (const tg of it.tags) counter.set(tg, (counter.get(tg) ?? 0) + 1)
        }
        const arr = Array.from(counter.entries())
          .map(([tag, n]) => ({ tag, n }))
          .sort((a, b) => b.n - a.n || a.tag.localeCompare(b.tag))
        setUniqueTotal(arr.length)
        setTags(arr)
      })
      .catch(() => {
        if (cancelled) return
        setTags([]); setUniqueTotal(0)
      })
    return () => { cancelled = true }
  }, [project.id, version])

  const max = useMemo(() => Math.max(1, ...tags.map((t) => t.n)), [tags])

  return (
    <HeroCard
      title={t('overview.detail.tagDist')}
      count={uniqueTotal}
      countSub={t('overview.detail.tagSuffix')}
      action={version ? {
        label: `${t('nav.tagEdit')}`,
        onClick: () => navigate(`/projects/${project.id}/v/${version.id}/edit`),
        disabled: !canGoVersionPhase(version, 'editing'),
      } : undefined}
      phase={`${t('nav.tag')} → ${t('nav.tagEdit')}`}
    >
      {tags.length === 0 ? (
        <p style={{ margin: 0, fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)', fontStyle: 'italic' }}>
          {t('overview.detail.emptyTag')}
        </p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3, fontFamily: 'var(--font-mono)', overflowY: 'auto', flex: 1, minHeight: 0 }}>
          {tags.map((row) => {
            const pct = row.n / max
            const isTrigger = !!triggerWord && row.tag === triggerWord
            return (
              <div key={row.tag} style={{
                display: 'grid', gridTemplateColumns: '1fr 36px',
                alignItems: 'center', gap: 8,
                padding: '4px 8px',
                minHeight: 22, flexShrink: 0,
                borderRadius: 'var(--r-sm)',
                background: isTrigger ? 'var(--accent-soft)' : 'transparent',
                position: 'relative', overflow: 'hidden',
              }}>
                <div style={{
                  position: 'absolute', left: 0, top: 0, bottom: 0,
                  width: `${pct * 100}%`,
                  background: isTrigger ? 'rgba(237,107,58,0.18)' : 'rgba(237,107,58,0.08)',
                  zIndex: 0,
                }}/>
                <span style={{
                  position: 'relative', zIndex: 1,
                  fontSize: 'var(--t-xs)',
                  color: isTrigger ? 'var(--accent)' : 'var(--fg-primary)',
                  fontWeight: isTrigger ? 700 : 500,
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>{isTrigger ? '★ ' : ''}<TranslatedTag tag={row.tag} /></span>
                <span style={{
                  position: 'relative', zIndex: 1,
                  fontSize: 'var(--t-xs)',
                  color: 'var(--fg-primary)', textAlign: 'right', fontWeight: 600,
                }}>{row.n}</span>
              </div>
            )
          })}
        </div>
      )}
    </HeroCard>
  )
}

// ── HistTile / RegTile (下排 3 tile) ─────────────────────────────────────

function HistTileCard({
  title, bins, action, phase, emptyHint,
}: {
  title: string
  bins: Array<{ key?: string; label: string; n: number }>
  action?: { label: string; onClick: () => void; disabled?: boolean }
  phase?: string
  emptyHint: string
}) {
  return (
    <HeroCard title={title} action={action} phase={phase}>
      {bins.length === 0 ? (
        <p style={{ margin: 0, fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)', fontStyle: 'italic' }}>
          {emptyHint}
        </p>
      ) : (
        <div style={{ overflowY: 'auto' }}>
          <BarHistogram bins={bins} />
        </div>
      )}
    </HeroCard>
  )
}

function RegTileCard({
  regCount, onGoReg, disabled,
}: {
  regCount: number
  onGoReg: () => void
  disabled?: boolean
}) {
  const { t } = useTranslation()
  if (regCount > 0) {
    return (
      <HeroCard
        title={t('overview.detail.regSet')}
        count={regCount}
        countSub={t('overview.detail.imagesSuffix')}
        action={{ label: `${t('nav.reg')}`, onClick: onGoReg, disabled }}
        phase={`${t('nav.reg')}`}
      >
        <p style={{ margin: 0, fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)' }}>
          {t('overview.detail.regCount', { n: regCount })}
        </p>
      </HeroCard>
    )
  }
  return (
    <HeroCard
      title={t('overview.detail.regSet')}
      action={{ label: `${t('nav.reg')}`, onClick: onGoReg, disabled }}
      phase={`${t('nav.reg')} · ${t('overview.banner.skippableHint')}`}
    >
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        gap: 8, padding: '20px 0', color: 'var(--fg-tertiary)',
        background: 'repeating-linear-gradient(45deg, transparent 0 8px, rgba(255,255,255,0.015) 8px 16px)',
        borderRadius: 'var(--r-md)',
      }}>
        <div style={{
          width: 36, height: 36, borderRadius: 'var(--r-md)',
          border: '1px dashed var(--border-default)',
          display: 'grid', placeItems: 'center', color: 'var(--fg-tertiary)', fontSize: 16,
        }}>∅</div>
        <span style={{ fontSize: 'var(--t-xs)', fontStyle: 'italic', textAlign: 'center', maxWidth: 200 }}>
          {t('overview.detail.regEmptyHint')}
        </span>
      </div>
    </HeroCard>
  )
}

// ── DetailGrid (2 hero + 3 tile) ─────────────────────────────────────────

function DetailGrid({ project, version }: { project: ProjectDetail; version: Version | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const vid = version?.id

  // ADR 0010: preprocess 已下沉 version scope；project 概览 hist 用 active
  // version 的 train 数据。没 active version → 空。
  const [preprocessItems, setPreprocessItems] = useState<Array<{ w: number | null; h: number | null }>>([])
  useEffect(() => {
    if (vid == null) { setPreprocessItems([]); return }
    let cancelled = false
    void api.listPreprocessFilesTrain(project.id, vid)
      .then((res) => {
        if (cancelled) return
        setPreprocessItems(res.images.map((i) => ({ w: i.w, h: i.h })))
      })
      .catch(() => { if (!cancelled) setPreprocessItems([]) })
    return () => { cancelled = true }
  }, [project.id, vid])

  // crop workspace - 长宽比 hist 数据源（train scope）
  const [cropItems, setCropItems] = useState<Array<{ w: number; h: number }>>([])
  useEffect(() => {
    if (vid == null) { setCropItems([]); return }
    let cancelled = false
    void api.listCropWorkspaceTrain(project.id, vid)
      .then((res) => {
        if (cancelled) return
        setCropItems(res.images.map((i) => ({ w: i.w, h: i.h })))
      })
      .catch(() => { if (!cancelled) setCropItems([]) })
    return () => { cancelled = true }
  }, [project.id, vid])

  const pixelBins = useMemo(
    () => computePixelHist(preprocessItems).map((b) => ({ key: b.id, label: b.label, n: b.n })),
    [preprocessItems],
  )
  const arBins = useMemo(() => {
    const m = new Map<string, { label: string; n: number; sortKey: number }>()
    for (const im of cropItems) {
      if (im.w <= 0 || im.h <= 0) continue
      const { label, sortKey } = arBucket(im.w / im.h)
      const prev = m.get(label)
      m.set(label, { label, sortKey, n: (prev?.n ?? 0) + 1 })
    }
    return Array.from(m.values())
      .sort((a, b) => b.sortKey - a.sortKey)
      .map((b) => ({ label: b.label, n: b.n }))
  }, [cropItems])

  const regCount = version?.stats?.reg_image_count ?? 0

  return (
    <div style={{ display: 'grid', gridTemplateRows: '1.4fr 1fr', gap: 12, height: '100%', minHeight: 0 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 12, minHeight: 0 }}>
        <TrainSetCard project={project} version={version} />
        <TagDistCard project={project} version={version} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, minHeight: 0 }}>
        <HistTileCard
          title={t('overview.detail.resolutionDist')}
          bins={pixelBins}
          emptyHint={t('overview.detail.emptyResolution')}
          action={version ? { label: `${t('nav.preprocess')}`, onClick: () => navigate(`/projects/${project.id}/v/${version.id}/preprocess?tool=upscale`) } : undefined}
          phase={`${t('nav.preprocess')}`}
        />
        <HistTileCard
          title={t('overview.detail.aspectDist')}
          bins={arBins}
          emptyHint={t('overview.detail.emptyAspect')}
          action={version ? { label: `${t('nav.preprocess')}`, onClick: () => navigate(`/projects/${project.id}/v/${version.id}/preprocess?tool=crop`) } : undefined}
          phase={`${t('nav.preprocess')}`}
        />
        <RegTileCard
          regCount={regCount}
          onGoReg={() => version && navigate(`/projects/${project.id}/v/${version.id}/reg`)}
          disabled={!canGoVersionPhase(version, 'regularizing')}
        />
      </div>
    </div>
  )
}

// ── Tasks / Output 面板（version scope，沿用） ───────────────────────────

const TASK_STATUS_BADGE: Record<string, string> = {
  pending: 'neutral', running: 'accent', paused: 'warn',
  done: 'ok', failed: 'err', canceled: 'neutral',
}

function VersionTasksPanel({ projectId, versionId }: { projectId: number; versionId: number | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [tasks, setTasks] = useState<Task[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    void api.listQueue()
      .then((items) => {
        if (cancelled) return
        const filtered = items
          .filter((tk) => tk.project_id === projectId && (versionId == null || tk.version_id === versionId))
          .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
        setTasks(filtered)
      })
      .catch(() => { if (!cancelled) setTasks([]) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [projectId, versionId])

  if (loading) return <div className="p-6 text-fg-tertiary text-sm">{t('common.loading')}</div>
  if (tasks.length === 0) return <div className="p-6 text-fg-tertiary text-sm italic">{t('overview.tasksEmpty')}</div>

  const fmtTime = (ts: number | null) => ts ? new Date(ts * 1000).toLocaleString() : '—'

  return (
    <div className="p-6">
      <table className="w-full text-sm">
        <thead className="text-fg-tertiary text-xs">
          <tr className="border-b border-subtle">
            <th className="text-left py-2 px-3 font-normal">{t('overview.tasksTable.name')}</th>
            <th className="text-left py-2 px-3 font-normal">{t('overview.tasksTable.status')}</th>
            <th className="text-left py-2 px-3 font-normal">{t('overview.tasksTable.started')}</th>
            <th className="text-left py-2 px-3 font-normal">{t('overview.tasksTable.finished')}</th>
          </tr>
        </thead>
        <tbody>
          {tasks.map((tk) => (
            <tr
              key={tk.id}
              className="border-b border-subtle cursor-pointer hover:bg-overlay"
              onClick={() => navigate(`/queue/${tk.id}`)}
            >
              <td className="py-2 px-3 font-mono">#{tk.id} {tk.name}</td>
              <td className="py-2 px-3"><span className={`badge badge-${TASK_STATUS_BADGE[tk.status] ?? 'neutral'}`}>{tk.status}</span></td>
              <td className="py-2 px-3 text-fg-tertiary text-xs">{fmtTime(tk.started_at ?? null)}</td>
              <td className="py-2 px-3 text-fg-tertiary text-xs">{fmtTime(tk.finished_at ?? null)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function VersionOutputPanel({
  version, latestTask,
}: {
  version: Version | null
  latestTask: Task | null
}) {
  const { t } = useTranslation()
  if (!version) return <div className="p-6 text-fg-tertiary text-sm italic">{t('overview.outputEmpty')}</div>
  if (!latestTask) return <div className="p-6 text-fg-tertiary text-sm italic">{t('overview.outputEmptyVersion')}</div>
  // 复用 QueueDetail OutputsTab：列表 + 排序 + 单文件下载 + 批量打 zip + 打开
  // 文件夹 + 导出 data_exports（跟 task 详情页同款行为）
  return <OutputsTab taskId={latestTask.id} />
}

// ── Main ─────────────────────────────────────────────────────────────────

export default function ProjectOverview() {
  const { t } = useTranslation()
  const { project, activeVersion } = useOutletContext<Ctx>()
  const ctx = useProjectCtx()

  // 初值优先级：URL `?version=N` (从 /queue 项目链接跳来时带) → project.active_version_id → activeVersion
  // 读完 URL 后用 history.replaceState 清掉 query，避免刷新覆盖用户后续在 dropdown 选的版本
  const [selectedVid, setSelectedVid] = useState<number | null>(() => {
    try {
      const sp = new URLSearchParams(window.location.search)
      const v = sp.get('version')
      if (v) {
        const n = Number(v)
        if (Number.isFinite(n)) return n
      }
    } catch { /* ignore */ }
    return project.active_version_id ?? activeVersion?.id ?? null
  })
  useEffect(() => {
    try {
      const url = new URL(window.location.href)
      if (url.searchParams.has('version')) {
        url.searchParams.delete('version')
        window.history.replaceState({}, '', url.toString())
      }
    } catch { /* ignore */ }
  }, [])
  useEffect(() => {
    const stillExists = project.versions.some((v) => v.id === selectedVid)
    if (!stillExists) setSelectedVid(project.active_version_id ?? null)
  }, [project.versions, project.active_version_id, selectedVid])

  const selectedVersion: Version | null =
    project.versions.find((v) => v.id === selectedVid) ?? null

  // 项目全部 task（最近优先）— 右栏 Recent runs + banner / Output 数据源
  const [projTasks, setProjTasks] = useState<Task[]>([])
  useEffect(() => {
    let cancelled = false
    void api.listQueue()
      .then((items) => {
        if (cancelled) return
        const list = items
          .filter((tk) => tk.project_id === project.id)
          .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
        setProjTasks(list)
      })
      .catch(() => { if (!cancelled) setProjTasks([]) })
    return () => { cancelled = true }
  }, [project.id])

  const latestTask = useMemo(
    () => projTasks.find((tk) => tk.version_id === selectedVid) ?? null,
    [projTasks, selectedVid],
  )

  const [activeTab, setActiveTab] = useState<OverviewTab>('details')
  const navigate = useNavigate()

  const tabBtnCls = (tab: OverviewTab) => [
    'py-2 px-[18px] text-sm border-0 bg-transparent -mb-px cursor-pointer transition-colors border-b-2',
    activeTab === tab
      ? 'font-semibold text-accent border-accent'
      : 'font-normal text-fg-tertiary hover:text-fg-primary border-transparent',
  ].join(' ')

  return (
    <div className="fade-in flex flex-col h-full min-h-0">
      <PageHeader
        eyebrow="Project"
        accentEyebrow
        title={project.title}
        subtitle={project.note ?? undefined}
        actions={
          <>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => ctx?.onExportTrain()}
              disabled={!selectedVersion || (ctx?.exporting ?? false)}
            >
              {ctx?.exporting ? t('sidebar.exporting') : t('overview.actions.exportBundle')}
            </button>
            <button className="btn btn-primary btn-sm" onClick={() => ctx?.onCreateVersion()}>
              + {t('overview.versionSelector.newVersion')}
            </button>
          </>
        }
      />

      <div className="flex-1 min-h-0 overflow-y-auto px-7 pt-5 pb-7">
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 320px', gap: 20, alignItems: 'start' }}>
          {/* ── 左列：pipeline → (task banner) → versions → details ──── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20, minWidth: 0 }}>
            <PipelineStrip project={project} version={selectedVersion} />

            {selectedVersion && (
              <StatusBanner
                projectId={project.id}
                version={selectedVersion}
                latestTask={latestTask}
                onOpenOutput={() => setActiveTab('output')}
              />
            )}

            <div>
              <div className="caption" style={{ marginBottom: 10 }}>
                {t('overview.versionsCaption')} · {project.versions.length}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {project.versions.map((v) => (
                  <VersionRowCard
                    key={v.id}
                    project={project}
                    v={v}
                    selected={v.id === selectedVid}
                    onView={(vid) => setSelectedVid(vid)}
                  />
                ))}
              </div>
            </div>

            <div>
              <div className="flex gap-0.5 border-b border-subtle mb-4">
                <button className={tabBtnCls('details')} onClick={() => setActiveTab('details')}>
                  {t('overview.tabDetails')}
                </button>
                <button className={tabBtnCls('tasks')} onClick={() => setActiveTab('tasks')}>
                  {t('overview.tabTasks')}
                </button>
                <button className={tabBtnCls('output')} onClick={() => setActiveTab('output')}>
                  {t('overview.tabOutput')}
                </button>
              </div>
              {activeTab === 'details' && (
                <div style={{ height: 'clamp(520px, 64vh, 760px)' }}>
                  <DetailGrid project={project} version={selectedVersion} />
                </div>
              )}
              {activeTab === 'tasks' && (
                <VersionTasksPanel projectId={project.id} versionId={selectedVid} />
              )}
              {activeTab === 'output' && (
                <VersionOutputPanel version={selectedVersion} latestTask={latestTask} />
              )}
            </div>
          </div>

          {/* ── 右栏：dataset 统计 + recent runs ──── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20, position: 'sticky', top: 0 }}>
            <div className="card" style={{ padding: 20 }}>
              <div className="caption" style={{ marginBottom: 12 }}>{t('overview.datasetCaption')}</div>
              {([
                [t('overview.dataset.pool'), project.download_image_count ?? 0],
                [t('overview.dataset.train'), selectedVersion?.stats?.train_image_count ?? 0],
                [t('overview.dataset.reg'), selectedVersion?.stats?.reg_image_count ?? 0],
                [t('overview.dataset.tagged'), selectedVersion?.stats?.tagged_image_count ?? 0],
              ] as Array<[string, number]>).map(([k, v], i, arr) => (
                <div
                  key={k}
                  style={{
                    display: 'flex', justifyContent: 'space-between', padding: '9px 0',
                    borderBottom: i < arr.length - 1 ? '1px solid var(--border-subtle)' : 'none',
                  }}
                >
                  <span style={{ fontSize: 'var(--t-sm)', color: 'var(--fg-secondary)' }}>{k}</span>
                  <span className="mono" style={{ fontSize: 'var(--t-sm)', fontWeight: 600 }}>{v}</span>
                </div>
              ))}
            </div>

            <div className="card" style={{ padding: 20 }}>
              <div className="caption" style={{ marginBottom: 12 }}>{t('overview.recentRunsCaption')}</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {projTasks.length === 0 && (
                  <span style={{ fontSize: 'var(--t-sm)', color: 'var(--fg-tertiary)' }}>{t('overview.noRuns')}</span>
                )}
                {projTasks.slice(0, 6).map((tk) => (
                  <button
                    key={tk.id}
                    onClick={() => navigate(`/queue/${tk.id}`)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 10, background: 'none',
                      border: 'none', padding: '4px 0', cursor: 'pointer', textAlign: 'left',
                    }}
                  >
                    <span className="mono" style={{ fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)', width: 34, flex: 'none' }}>#{tk.id}</span>
                    <span style={{ flex: 1, fontSize: 'var(--t-sm)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {tk.config_name || tk.name}
                    </span>
                    <span className={`badge badge-${TASK_STATUS_BADGE[tk.status] ?? 'neutral'}`}>{tk.status}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
