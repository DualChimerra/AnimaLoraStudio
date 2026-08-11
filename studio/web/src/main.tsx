import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { DialogProvider } from './components/Dialog'
import { ErrorBoundary } from './components/ErrorBoundary'
import RuntimeModeGate from './components/RuntimeModeGate'
import { ToastProvider } from './components/Toast'
import { installGlobalErrorHandlers } from './lib/errors/setup'
import { RuntimeModeProvider } from './lib/RuntimeMode'
import { SettingsDataProvider } from './lib/SettingsData'
import { SettingsDrawerProvider } from './lib/SettingsDrawer'
import { initTheme } from './lib/theme'
import './i18n'
import './index.css'

// ADR-0009 PR-3 C2: window.onerror + unhandledrejection 三路捕获 → /api/client-errors
installGlobalErrorHandlers()

initTheme()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <ToastProvider>
        <DialogProvider>
          <SettingsDataProvider>
            <SettingsDrawerProvider>
              {/* RuntimeModeProvider 包在 App 外：Topbar 徽标 / Settings 区都要读
                  模式，而 Gate 需要在整个应用之上盖一层（首次进来必答一次）。 */}
              <RuntimeModeProvider>
                <App />
                <RuntimeModeGate />
              </RuntimeModeProvider>
            </SettingsDrawerProvider>
          </SettingsDataProvider>
        </DialogProvider>
      </ToastProvider>
    </ErrorBoundary>
  </React.StrictMode>,
)
