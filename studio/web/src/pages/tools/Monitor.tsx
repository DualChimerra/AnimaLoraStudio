import { useEffect, useMemo, useState } from 'react'
import { api, type HealthResponse, type Task } from '../../api/client'
import MonitorDashboard from '../../components/MonitorDashboard'
import PageHeader from '../../components/PageHeader'

export default function MonitorPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tasks, setTasks] = useState<Task[]>([])
  // `?task=N` 深链：直接把监控页锁定到指定 task（书签 / 外部链接用）。
  const initialTaskId = useMemo<number | null>(() => {
    if (typeof window === 'undefined') return null
    const raw = new URLSearchParams(window.location.search).get('task')
    const n = raw === null ? NaN : Number(raw)
    return Number.isFinite(n) && n > 0 ? n : null
  }, [])
  const [taskId, setTaskId] = useState<number | null>(initialTaskId)

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(String(e)))
    api.listQueue().then(setTasks).catch(() => setTasks([]))
  }, [])

  const defaultTaskId = useMemo<number | null>(() => {
    const running = tasks.find((t) => t.status === 'running')
    if (running) return running.id
    const ended = [...tasks]
      .filter((t) => t.finished_at)
      .sort((a, b) => (b.finished_at ?? 0) - (a.finished_at ?? 0))[0]
    return ended?.id ?? null
  }, [tasks])

  useEffect(() => {
    if (taskId === null && defaultTaskId !== null) setTaskId(defaultTaskId)
  }, [defaultTaskId, taskId])

  const ok = !error && health?.status === 'ok'
  const selectedTask = tasks.find((t) => t.id === taskId)

  return (
    <div className="flex flex-col h-full min-h-0 overflow-hidden">
      <PageHeader
        title="Monitor"
        eyebrow="Live training"
        subtitle="Real-time loss / LR curves and sample images for a training task."
        sticky
      />
      <div className="flex-1 min-h-0 flex flex-col p-6 overflow-hidden">
      {/* 顶部状态栏 */}
      <section className="card text-xs flex items-center gap-3 shrink-0 flex-wrap"
        style={{ padding: '10px 16px', margin: '0 0 12px 0' }}>
        {/* 健康指示 */}
        <span className={`inline-block w-2 h-2 rounded-full ${ok ? 'bg-ok' : 'bg-err'}`}
          style={{ boxShadow: ok ? '0 0 6px var(--ok)' : '0 0 6px var(--err)' }} />
        <span className={`font-semibold font-mono ${ok ? 'text-ok' : 'text-err'}`}>
          {error ? 'offline' : health?.status ?? '...'}
        </span>
        {health && (
          <span className="text-fg-tertiary font-mono">
            v{health.version}
          </span>
        )}

        <span className="text-fg-tertiary">|</span>

        {/* 任务选择 */}
        <span className="text-fg-tertiary">Task</span>
        <select
          value={taskId ?? ''}
          onChange={(e) => setTaskId(e.target.value === '' ? null : Number(e.target.value))}
          className="rounded-sm bg-sunken border border-subtle text-xs text-fg-primary"
          style={{ padding: '4px 10px', outline: 'none' }}
        >
          <option value="">(latest running, empty if none)</option>
          {tasks.map((t) => (
            <option key={t.id} value={t.id}>
              #{t.id} · {t.name} · {t.status}
            </option>
          ))}
        </select>

        {selectedTask && (
          <>
            <span className="text-fg-tertiary">|</span>
            <span className={statusBadge(selectedTask.status)}>
              {statusLabel(selectedTask.status)}
            </span>
          </>
        )}

        <span style={{ flex: 1 }} />
      </section>

      {/* 监控主体 */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {taskId !== null ? (
          <MonitorDashboard taskId={taskId} />
        ) : (
          <div className="flex items-center justify-center h-full text-fg-tertiary text-sm flex-col gap-2">
            <span className="text-xl">📊</span>
            <span>No training tasks</span>
            <span className="text-xs">Monitoring data appears automatically once training starts</span>
          </div>
        )}
      </div>
      </div>
    </div>
  )
}

function statusBadge(status: string): string {
  switch (status) {
    case 'running': return 'badge badge-accent'
    case 'pending': return 'badge badge-neutral'
    case 'scheduled': return 'badge badge-neutral'
    case 'done': return 'badge badge-ok'
    case 'failed': return 'badge badge-err'
    case 'canceled': return 'badge badge-neutral'
    default: return 'badge badge-neutral'
  }
}

function statusLabel(status: string): string {
  switch (status) {
    case 'running': return 'Running'
    case 'pending': return 'Pending'
    case 'scheduled': return 'Scheduled'
    case 'done': return 'Done'
    case 'failed': return 'Failed'
    case 'canceled': return 'Canceled'
    default: return status
  }
}
