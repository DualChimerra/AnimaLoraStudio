import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { api, type CaptionEntry, type PresetSummary, type ProjectSummary } from '../api/client'
import { useProjectCtx } from '../context/ProjectContext'
import { useSettingsDrawer } from '../lib/SettingsDrawer'

type IconKey = 'folder' | 'queue' | 'preset' | 'monitor' | 'cog' | 'image' | 'step' | 'tag'

interface Item {
  id: string
  label: string
  sub?: string
  group: string
  icon: IconKey
  /** 路由跳转。跟 action 二选一。 */
  path?: string
  /** 自定义动作（如打开抽屉）。优先于 path。 */
  action?: () => void
}

const SEARCH_ICON = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" />
  </svg>
)

const CHEVRON_ICON = (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 6l6 6-6 6" />
  </svg>
)

/** 每个结果左侧的小图标（prototype CommandPalette：icon + label + sub + chevron）。 */
const ITEM_ICONS: Record<IconKey, React.ReactNode> = {
  folder:  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>,
  queue:   <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M4 6h16M4 12h10M4 18h16"/><circle cx="18" cy="12" r="2" fill="currentColor"/></svg>,
  preset:  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><line x1="6" y1="4" x2="6" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/><line x1="18" y1="4" x2="18" y2="20"/><circle cx="6" cy="9" r="2" fill="var(--bg-surface)"/><circle cx="12" cy="15" r="2" fill="var(--bg-surface)"/><circle cx="18" cy="7" r="2" fill="var(--bg-surface)"/></svg>,
  monitor: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l4-6 4 3 5-9 5 7"/></svg>,
  cog:     <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>,
  image:   <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="1.6" fill="currentColor"/><path d="m21 15-5-5L5 21"/></svg>,
  step:    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9 12h6"/></svg>,
  tag:     <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2H2v10l9.29 9.29a1 1 0 0 0 1.41 0l7.3-7.3a1 1 0 0 0 0-1.41L12 2z"/><path d="M7 7h.01"/></svg>,
}

interface Props {
  open: boolean
  onClose: () => void
}

export default function CommandPalette({ open, onClose }: Props) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const ctx = useProjectCtx()
  const settingsDrawer = useSettingsDrawer()
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const [query, setQuery] = useState('')
  const [activeIdx, setActiveIdx] = useState(0)

  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [projectsLoaded, setProjectsLoaded] = useState(false)

  const [presets, setPresets] = useState<PresetSummary[]>([])
  const [presetsLoaded, setPresetsLoaded] = useState(false)

  const [captions, setCaptions] = useState<CaptionEntry[]>([])
  const [captionsCacheKey, setCaptionsCacheKey] = useState<string | null>(null)
  const [captionsLoading, setCaptionsLoading] = useState(false)

  useEffect(() => {
    if (open) {
      setQuery('')
      setActiveIdx(0)
      setTimeout(() => inputRef.current?.focus(), 40)
    } else {
      setProjectsLoaded(false)
      setPresetsLoaded(false)
    }
  }, [open])

  useEffect(() => {
    if (!open || projectsLoaded) return
    let cancelled = false
    api.listProjects().then((items) => {
      if (!cancelled) { setProjects(items); setProjectsLoaded(true) }
    }).catch(() => {
      if (!cancelled) setProjectsLoaded(true)
    })
    return () => { cancelled = true }
  }, [open, projectsLoaded])

  useEffect(() => {
    if (!open || presetsLoaded) return
    let cancelled = false
    api.listPresets().then((items) => {
      if (!cancelled) { setPresets(items); setPresetsLoaded(true) }
    }).catch(() => {
      if (!cancelled) setPresetsLoaded(true)
    })
    return () => { cancelled = true }
  }, [open, presetsLoaded])

  const pid = ctx?.project?.id
  const vid = ctx?.activeVersion?.id
  const queryEnoughForTags = query.length >= 2

  useEffect(() => {
    if (!open || !queryEnoughForTags || !pid || !vid) return
    const key = `${pid}:${vid}`
    if (captionsCacheKey === key) return

    let cancelled = false
    setCaptionsLoading(true)
    api.listCaptionsFull(pid, vid).then((result) => {
      if (!cancelled) {
        setCaptions(result.items)
        setCaptionsCacheKey(key)
        setCaptionsLoading(false)
      }
    }).catch(() => {
      if (!cancelled) setCaptionsLoading(false)
    })
    return () => { cancelled = true }
  }, [open, queryEnoughForTags, pid, vid, captionsCacheKey])

  const allItems = useMemo<Item[]>(() => {
    const items: Item[] = []

    items.push({ id: 'home',     label: t('commandPalette.home'),     sub: t('commandPalette.homeSub'),     group: t('commandPalette.pages'), icon: 'folder',  path: '/' })
    items.push({ id: 'queue',    label: t('nav.queue'),               sub: t('commandPalette.queueSub'),    group: t('commandPalette.pages'), icon: 'queue',   path: '/queue' })
    items.push({ id: 'generate', label: t('nav.generate'),            sub: t('commandPalette.generateSub'), group: t('commandPalette.pages'), icon: 'image',   path: '/tools/generate' })
    items.push({ id: 'presets',  label: t('nav.presets'),             sub: t('commandPalette.presetsSub'), group: t('commandPalette.pages'), icon: 'preset',  path: '/tools/presets' })
    items.push({ id: 'monitor',  label: t('nav.monitor'),             sub: t('commandPalette.monitorSub'), group: t('commandPalette.pages'), icon: 'monitor', path: '/tools/monitor' })
    items.push({ id: 'settings', label: t('nav.settings'),            sub: t('commandPalette.settingsSub'), group: t('commandPalette.pages'), icon: 'cog',     action: () => settingsDrawer.open() })

    for (const p of presets) {
      items.push({
        id: `preset:${p.name}`,
        label: p.name,
        sub: t('commandPalette.presetItem'),
        group: t('commandPalette.presets'),
        icon: 'preset',
        path: '/tools/presets',
      })
    }

    for (const p of projects) {
      items.push({
        id: `project:${p.id}`,
        label: p.title || `#${p.id}`,
        sub: p.slug ? `/${p.slug}` : t('commandPalette.projectItem', { id: p.id }),
        group: t('commandPalette.projects'),
        icon: 'folder',
        path: `/projects/${p.id}`,
      })
    }

    if (ctx) {
      const cpid = ctx.project.id
      const cvid = ctx.activeVersion?.id
      const group = t('commandPalette.currentProject')
      items.push({ id: `overview:${cpid}`, label: t('nav.overview'), sub: ctx.project.title, group, icon: 'folder', path: `/projects/${cpid}` })
      items.push({ id: `download:${cpid}`, label: t('nav.download'), sub: ctx.project.title, group, icon: 'step', path: `/projects/${cpid}/download` })
      if (cvid) {
        const base = `/projects/${cpid}/v/${cvid}`
        items.push({ id: `curate:${cpid}`, label: t('nav.curate'),   sub: ctx.project.title, group, icon: 'step', path: `${base}/curate` })
        items.push({ id: `edit:${cpid}`,   label: t('nav.tagEdit'),  sub: ctx.project.title, group, icon: 'step', path: `${base}/edit` })
        items.push({ id: `reg:${cpid}`,    label: t('nav.reg'),      sub: ctx.project.title, group, icon: 'step', path: `${base}/reg` })
        items.push({ id: `train:${cpid}`,  label: t('nav.train'),    sub: ctx.project.title, group, icon: 'step', path: `${base}/train` })
      }
    }

    return items
  }, [projects, presets, ctx, t, settingsDrawer])

  const filteredNav = useMemo(() => {
    if (!query.trim()) return allItems
    const q = query.toLowerCase()
    return allItems.filter(
      (item) =>
        item.label.toLowerCase().includes(q) ||
        (item.sub ?? '').toLowerCase().includes(q) ||
        item.group.toLowerCase().includes(q),
    )
  }, [allItems, query])

  const tagItems = useMemo<Item[]>(() => {
    if (!queryEnoughForTags || !ctx?.activeVersion || captions.length === 0) return []
    const cpid = ctx.project.id
    const cvid = ctx.activeVersion.id
    const q = query.toLowerCase()
    const tagCounts = new Map<string, number>()

    for (const c of captions) {
      for (const tag of c.tags) {
        if (tag.toLowerCase().includes(q)) {
          tagCounts.set(tag, (tagCounts.get(tag) ?? 0) + 1)
        }
      }
    }

    return Array.from(tagCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12)
      .map(([tag, count]): Item => ({
        id: `tag:${tag}`,
        label: tag,
        sub: t('commandPalette.imageCount', { n: count }),
        group: t('commandPalette.tags'),
        icon: 'tag',
        path: `/projects/${cpid}/v/${cvid}/edit`,
      }))
  }, [captions, queryEnoughForTags, query, ctx, t])

  const filtered = useMemo(() => [...filteredNav, ...tagItems], [filteredNav, tagItems])

  const grouped = useMemo(() => {
    const map = new Map<string, Item[]>()
    for (const item of filtered) {
      if (!map.has(item.group)) map.set(item.group, [])
      map.get(item.group)!.push(item)
    }
    return map
  }, [filtered])

  const flatItems = useMemo(() => {
    const out: Item[] = []
    for (const [, items] of grouped) out.push(...items)
    return out
  }, [grouped])

  const select = useCallback(
    (item: Item) => {
      if (item.action) item.action()
      else if (item.path) navigate(item.path)
      onClose()
    },
    [navigate, onClose],
  )

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIdx((i) => Math.min(i + 1, flatItems.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIdx((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (flatItems[activeIdx]) select(flatItems[activeIdx])
    } else if (e.key === 'Escape') {
      onClose()
    }
  }

  useEffect(() => {
    const el = listRef.current
    if (!el) return
    const active = el.querySelector(`[data-palette-idx="${activeIdx}"]`) as HTMLElement | null
    if (active) active.scrollIntoView({ block: 'nearest' })
  }, [activeIdx])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-[80] flex justify-center px-4"
      style={{ paddingTop: '12vh', background: 'rgba(23,24,26,0.42)', backdropFilter: 'blur(2px)' }}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div
        className="card w-full max-w-[560px] h-fit overflow-hidden flex flex-col"
        style={{ boxShadow: 'var(--sh-xl)', maxHeight: 'min(70vh, 560px)' }}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2.5 px-[18px] py-3.5 border-b border-subtle">
          <span className="text-fg-tertiary">{SEARCH_ICON}</span>
          <input
            ref={inputRef}
            type="text"
            className="flex-1 bg-transparent border-none outline-none text-md text-fg-primary placeholder:text-fg-disabled"
            placeholder={t('commandPalette.placeholder')}
            value={query}
            onChange={(e) => { setQuery(e.target.value); setActiveIdx(0) }}
            onKeyDown={handleKeyDown}
          />
          {captionsLoading && (
            <span className="text-2xs text-fg-tertiary animate-pulse">{t('commandPalette.searchingTags')}</span>
          )}
          <kbd className="kbd">esc</kbd>
        </div>

        <div ref={listRef} className="flex-1 overflow-y-auto p-2">
          {filtered.length === 0 ? (
            <div className="text-sm text-fg-tertiary text-center py-6">{t('commandPalette.noResults')}</div>
          ) : (
            [...grouped.entries()].map(([group, items]) => (
              <div key={group} className="mb-1">
                <div className="px-3 py-1.5">
                  <span className="text-2xs text-fg-tertiary font-semibold uppercase tracking-wider">
                    {group}
                  </span>
                </div>
                {items.map((item) => {
                  const idx = flatItems.indexOf(item)
                  const isActive = idx === activeIdx
                  return (
                    <button
                      key={item.id}
                      data-palette-idx={idx}
                      onClick={() => select(item)}
                      onMouseEnter={() => setActiveIdx(idx)}
                      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-left border-none cursor-pointer transition-colors ${
                        isActive ? 'bg-overlay' : 'bg-transparent hover:bg-overlay'
                      }`}
                    >
                      <span className="text-fg-tertiary shrink-0 grid place-items-center w-4">{ITEM_ICONS[item.icon]}</span>
                      <span className="flex-1 min-w-0 flex items-baseline gap-2">
                        <span className="text-sm font-semibold text-fg-primary overflow-hidden text-ellipsis whitespace-nowrap shrink-0 max-w-[55%]">
                          {item.label}
                        </span>
                        {item.sub && (
                          <span className="text-xs text-fg-tertiary overflow-hidden text-ellipsis whitespace-nowrap">
                            {item.sub}
                          </span>
                        )}
                      </span>
                      <span className="text-fg-tertiary shrink-0">{CHEVRON_ICON}</span>
                    </button>
                  )
                })}
              </div>
            ))
          )}
        </div>

        <div className="flex items-center gap-4 px-[18px] py-2 border-t border-subtle text-2xs text-fg-tertiary">
          <span className="flex items-center gap-1"><kbd className="kbd">↑↓</kbd> {t('commandPalette.navigate')}</span>
          <span className="flex items-center gap-1"><kbd className="kbd">enter</kbd> {t('commandPalette.select')}</span>
          <span className="flex items-center gap-1"><kbd className="kbd">esc</kbd> {t('commandPalette.close')}</span>
          {queryEnoughForTags && ctx?.activeVersion && (
            <span className="ml-auto opacity-70">{t('commandPalette.searchTagsHint', { n: captions.length })}</span>
          )}
        </div>
      </div>
    </div>
  )
}
