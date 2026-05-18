import { useState, useEffect, useCallback } from 'react';
import { useChat } from './hooks/useChat';
import { useAuth } from './hooks/useAuth';
import { useTheme } from './hooks/useTheme';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import InputBar from './components/InputBar';
import LoginScreen from './components/LoginScreen';
import SharedView from './components/SharedView';
import type { HealthStatus } from './types';

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';

export default function App() {
  // ── Shared conversation route (/?share=<token>) ───────────────────────────
  const shareToken = new URLSearchParams(window.location.search).get('share');
  if (shareToken) {
    return (
      <div className="h-full" style={{ background: 'var(--bg-base)' }}>
        <SharedView shareToken={shareToken} />
      </div>
    );
  }

  const { user, login, register, logout, loginWithGoogle } = useAuth();
  const { theme, toggleTheme }            = useTheme();

  // ── Chat (scoped to the authenticated user) ───────────────────────────────
  // Pass both userId and the JWT token so that tokenRef is updated
  // synchronously during render — before any userId-change effects fire.
  // This prevents two bugs:
  //   1. Race condition: /conversations was fetched without a token on login.
  //   2. Cross-user leak: stale token from the previous user caused the server
  //      to return the wrong conversation list for the newly logged-in user.
  // Pass `logout` as the auth-error callback so that any 401 from a
  // conversation endpoint (expired or missing JWT) immediately shows the
  // login screen instead of silently failing.
  const chat = useChat(user?.id ?? '__guest__', user?.token ?? '', logout);

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [health, setHealth]           = useState<HealthStatus>('unreachable');
  const [prefillText, setPrefillText] = useState('');

  // ── Health check (poll every 30 s) ────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_URL}/health`);
        if (!res.ok) { setHealth('unreachable'); return; }
        const data = (await res.json()) as { status: string };
        setHealth(data.status === 'ok' ? 'ok' : 'degraded');
      } catch {
        setHealth('unreachable');
      }
    };
    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, []);

  const handleExampleClick = useCallback((q: string) => {
    setPrefillText(q);
  }, []);

  const handlePrefillUsed = useCallback(() => {
    setPrefillText('');
  }, []);

  // ── Auth gate ─────────────────────────────────────────────────────────────
  if (!user) {
    return (
      <div className="h-full" style={{ background: 'var(--bg-base)' }}>
        <LoginScreen onLogin={login} onRegister={register} onGoogleLogin={loginWithGoogle} />
      </div>
    );
  }

  return (
    <div className="flex h-full" style={{ background: 'var(--bg-base)' }}>
      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        sessions={chat.sessions}
        currentSessionId={chat.currentSessionId}
        onSelectSession={id => {
          chat.setCurrentSessionId(id);
          chat.setMessages([]);
          chat.loadSessionMessages(id);
          setSidebarOpen(false);
        }}
        onNewChat={chat.startNewChat}
        onDeleteSession={chat.deleteSession}
        onRenameSession={chat.renameSession}
        onShareSession={chat.shareSession}
        settings={chat.settings}
        onSettingsChange={chat.setSettings}
        health={health}
        user={user}
        onLogout={logout}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      {/* ── Mobile backdrop ──────────────────────────────────────────────── */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-10 md:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* ── Main area ────────────────────────────────────────────────────── */}
      <div
        className="flex flex-col flex-1 min-w-0"
        style={{ background: 'var(--bg-chat)' }}
      >
        {/* Mobile top bar */}
        <header
          className="flex items-center gap-3 px-4 py-3 border-b md:hidden flex-shrink-0"
          style={{ borderColor: 'var(--border)' }}
        >
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-1.5 rounded-lg transition-colors"
            aria-label="Open menu"
            onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg-ai-bubble)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
          >
            <svg
              width="18" height="18" viewBox="0 0 24 24"
              fill="none" stroke="currentColor" strokeWidth="2"
              style={{ color: 'var(--text-primary)' }}
            >
              <line x1="3" y1="6"  x2="21" y2="6"  />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>
          <span className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>
            DocAssist
          </span>
          {/* User initial on mobile */}
          <div
            className="ml-auto flex items-center justify-center w-6 h-6 rounded-full text-xs font-semibold uppercase"
            style={{ background: 'var(--accent)', color: 'white' }}
            title={user.username}
          >
            {user.username.charAt(0)}
          </div>
        </header>

        {/* Chat messages */}
        <ChatArea
          messages={chat.messages}
          isStreaming={chat.isStreaming}
          onExampleClick={handleExampleClick}
          username={user.username}
          onFeedback={chat.submitFeedback}
        />

        {/* Input */}
        <InputBar
          onSend={chat.sendMessage}
          onStop={chat.stopStreaming}
          disabled={chat.isStreaming}
          prefillText={prefillText || undefined}
          onPrefillUsed={handlePrefillUsed}
          scope={chat.scope}
          onScopeChange={chat.setScope}
          indexedDocs={chat.indexedDocs}
          onDocIndexed={chat.refreshIndexedDocs}
          userId={user.id}
        />
      </div>
    </div>
  );
}
