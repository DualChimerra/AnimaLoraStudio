import { useEffect, useState } from 'react'
import { api, type SystemStats as SystemStatsData } from '../api/client'
import { useEventStream } from '../lib/useEventStream'

function barColor(pct: number): string {
  if (pct > 90) return 'var(--err)'
  if (pct > 70) return 'var(--warn)'
  return 'var(--ok)'
}

/** redesign 原型 SystemStats：一个 sunken pill，CPU/GPU/MEM 各为
 *  caption 标签 + 36px 迷你进度条 + 百分比；VRAM 用文本 used/totalG。
 *  (prototype shell.jsx → SystemStats) */
function MeterItem({ label, pct, tooltip }: { label: string; pct: number; tooltip: string }) {
  const clamped = Math.min(100, Math.max(0, pct))
  return (
    <div className="flex items-center gap-1.5" title={tooltip}>
      <span className="caption text-2xs">{label}</span>
      <div className="w-9 h-[5px] rounded-full bg-overlay overflow-hidden shrink-0">
        <div
          className="h-full rounded-full transition-[width] duration-500 ease-out"
          style={{ width: `${clamped}%`, background: barColor(clamped) }}
        />
      </div>
      <span className="font-mono text-2xs text-fg-secondary tabular-nums w-[26px]">{Math.round(clamped)}%</span>
    </div>
  )
}

export default function SystemStats() {
  const [stats, setStats] = useState<SystemStatsData | null>(null)

  // mount 时拉一次冷启动 (避免空白等 2.5s 首个 SSE 事件)，之后纯靠后端
  // sampler 通过 SSE 推送。SSE 重连时 onOpen 也补一次冷启动，防漏。
  useEffect(() => {
    let cancelled = false
    api.systemStats().then((s) => {
      if (!cancelled) setStats(s)
    }).catch(() => {/* 首次失败：等 SSE 第一帧就行 */})
    return () => { cancelled = true }
  }, [])

  useEventStream(
    (evt) => {
      if (evt.type !== 'system_stats_updated') return
      const payload = evt.payload as SystemStatsData | undefined
      if (payload) setStats(payload)
    },
    {
      onOpen: () => {
        // SSE 重连：补一次冷启动；服务端 sampler 仍在跑，下次 tick 会自然推
        // 上来，但这一次显式 GET 让 UI 立刻刷新
        api.systemStats().then((s) => setStats(s)).catch(() => {})
      },
    },
  )

  if (!stats) return null

  const gpu0 = stats.gpu && stats.gpu.length > 0 ? stats.gpu[0] : null
  const ramPct = stats.ram_total_gb > 0 ? (stats.ram_used_gb / stats.ram_total_gb) * 100 : 0
  const vramPct = gpu0 && gpu0.vram_total_gb > 0 ? (gpu0.vram_used_gb / gpu0.vram_total_gb) * 100 : 0

  const gpuExtra = stats.gpu && stats.gpu.length > 1
    ? ` (+${stats.gpu.length - 1} more)`
    : ''
  const gpuTempText = gpu0?.temp_c != null ? ` · ${gpu0.temp_c}°C` : ''
  const gpuLabel = gpu0 ? `${gpu0.name}${gpuTempText}${gpuExtra}` : ''

  return (
    <div className="hidden md:flex items-center gap-3.5 shrink-0 px-3 py-[5px] rounded-md bg-sunken border border-subtle">
      <MeterItem
        label="CPU"
        pct={stats.cpu_pct}
        tooltip={`CPU usage ${stats.cpu_pct.toFixed(1)}%`}
      />
      {gpu0 && (
        <MeterItem
          label="GPU"
          pct={gpu0.util_pct}
          tooltip={`GPU utilization · ${gpuLabel}`}
        />
      )}
      <MeterItem
        label="MEM"
        pct={ramPct}
        tooltip={`RAM ${stats.ram_used_gb.toFixed(1)} / ${stats.ram_total_gb.toFixed(1)} GB (${ramPct.toFixed(0)}%)`}
      />
      {gpu0 && (
        <div
          className="flex items-center gap-1.5"
          title={`VRAM ${gpu0.vram_used_gb.toFixed(1)} / ${gpu0.vram_total_gb.toFixed(1)} GB (${vramPct.toFixed(0)}%) · ${gpuLabel}`}
        >
          <span className="caption text-2xs">VRAM</span>
          <span className="font-mono text-2xs text-fg-secondary tabular-nums">
            {gpu0.vram_used_gb.toFixed(1)}/{Math.round(gpu0.vram_total_gb)}G
          </span>
        </div>
      )}
    </div>
  )
}
