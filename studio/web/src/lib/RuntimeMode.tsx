// RuntimeMode.tsx —— Colab / Local 运行模式的全局数据层（本 fork）。
//
// 后端 `/api/runtime` 是唯一真相（见 studio/infrastructure/runtime_mode.py）。
// 这层做三件事：
//   1. 首屏拉一次状态，`mode === ''`（用户没选过）且未被 env 钉死时把 `needsPick`
//      打开 —— `<RuntimeModeGate>` 据此弹选择框；
//   2. 暴露 `setMode` 给选择框和设置区共用；
//   3. 把 `effective` 模式广播给需要按模式改文案 / 行为的组件（Topbar 徽标、
//      设置区的本地路径提示等）。
//
// 拉取失败**不**阻断应用：后端刚起来时 SPA 可能先于 router 就绪，模式只影响
// 提示与默认值，硬卡首屏得不偿失。失败时 `info` 保持 null、`needsPick` 为
// false，用户照常用，下次刷新再问。
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

import { api, type RuntimeInfo, type RuntimeMode } from '../api/client'

interface RuntimeModeValue {
  info: RuntimeInfo | null
  /** 生效模式；还没拿到状态时按 'local' 兜底（本地是绝大多数场景）。 */
  mode: RuntimeMode
  /** true = 需要弹选择框（用户从没选过，且没被 ALS_RUNTIME_MODE 钉死）。 */
  needsPick: boolean
  loading: boolean
  error: string | null
  /** 持久化选择；成功后 needsPick 落下。抛错交给调用方展示。 */
  setMode: (mode: RuntimeMode) => Promise<RuntimeInfo>
  reload: () => Promise<void>
}

const Ctx = createContext<RuntimeModeValue | null>(null)

export function RuntimeModeProvider({ children }: { children: ReactNode }) {
  const [info, setInfo] = useState<RuntimeInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const reload = useCallback(async () => {
    try {
      const next = await api.getRuntime()
      setInfo(next)
      setError(null)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void reload() }, [reload])

  const setMode = useCallback(async (mode: RuntimeMode) => {
    const next = await api.setRuntimeMode(mode)
    setInfo(next)
    setError(null)
    return next
  }, [])

  const value = useMemo<RuntimeModeValue>(() => ({
    info,
    mode: info?.effective ?? 'local',
    // locked（env 注入）时永远不问 —— 环境已经替用户答过了。
    needsPick: !!info && !info.locked && info.mode === '',
    loading,
    error,
    setMode,
    reload,
  }), [info, loading, error, setMode, reload])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useRuntimeMode(): RuntimeModeValue {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useRuntimeMode must be used inside <RuntimeModeProvider>')
  return ctx
}

/** 同上，但没有 Provider 时返回 null 而不是抛错。
 *
 *  给「可能在 Provider 之外被单独渲染」的挂件用 —— Topbar 的模式徽标、Settings
 *  里的模式区都属此类：它们在真实应用里必然在 Provider 内（main.tsx 挂在根
 *  上），但组件测试常只 render 单页/单组件。让一个装饰性徽标把整页 render 打挂
 *  是不划算的取舍；这些调用点在 `info` 为 null 时本来就要走"还没拿到状态"的
 *  分支，多一个 null 来源不增加复杂度。 */
export function useRuntimeModeOptional(): RuntimeModeValue | null {
  return useContext(Ctx)
}
