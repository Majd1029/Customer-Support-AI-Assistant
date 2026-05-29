import { useState, useEffect, useRef } from 'react';
import {
  BookOpen, Plus, MessageSquare, Settings2, X, LogOut, Trash2,
  Sun, Moon, Bell, Info, FolderOpen, Link2, Check, Pencil,
} from 'lucide-react';
import type { Session, Settings, HealthStatus, User } from '../types';
import StatusDot from './StatusDot';
import CrawlersPanel from './CrawlersPanel';

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';

interface SidebarProps {
  open:               boolean;
  onClose:            () => void;
  sessions:           Session[];
  currentSessionId:   string | null;
  onSelectSession:    (id: string) => void;
  onNewChat:          () => void;
  onDeleteSession:    (id: string) => void;
  onRenameSession?:   (id: string, label: string) => void;
  onShareSession:     (id: string) => Promise<string | null>;
  settings:           Settings;
  onSettingsChange:   (s: Settings) => void;
  health:             HealthStatus;
  user:               User;
  onLogout:           () => void;
  theme:              'light' | 'dark';
  onToggleTheme:      () => void;
}

export default function Sidebar({
  open, onClose, sessions, currentSessionId,
  onSelectSession, onNewChat, onDeleteSession, onRenameSession, onShareSession,
  settings, onSettingsChange, health, user, onLogout,
  theme, onToggleTheme,
}: SidebarProps) {
  const [hoveredSession, setHoveredSession]         = useState<string | null>(null);
  const [settingsModalOpen, setSettingsModalOpen]   = useState(false);
  const [crawlersPanelOpen, setCrawlersPanelOpen]   = useState(false);
  const [copiedSession, setCopiedSession]           = useState<string | null>(null);
  // Inline rename state
  const [editingSession, setEditingSession]         = useState<string | null>(null);
  const [editingLabel, setEditingLabel]             = useState('');
  const renameInputRef                              = useRef<HTMLInputElement>(null);

  const commitRename = (id: string) => {
    const trimmed = editingLabel.trim();
    if (trimmed && onRenameSession) onRenameSession(id, trimmed);
    setEditingSession(null);
  };

  return (
    <>
      <aside
        className={[
          'flex flex-col h-full w-[260px] flex-shrink-0',
          'fixed top-0 left-0 z-20 md:static md:translate-x-0',
          'transition-transform duration-300 ease-in-out',
          open ? 'translate-x-0' : '-translate-x-full',
        ].join(' ')}
        style={{ background: 'var(--bg-sidebar)' }}
      >
        {/* ── Logo row ─────────────────────────────────────────────────────── */}
        <div className="flex items-center gap-2.5 px-5 pt-5 pb-4">
          <div
            className="flex items-center justify-center w-8 h-8 rounded-lg flex-shrink-0"
            style={{ background: 'var(--accent)' }}
          >
            <BookOpen size={15} color="white" />
          </div>
          <span className="font-semibold text-sm" style={{ color: 'var(--text-on-dark)' }}>
            CustomerAssist
          </span>
          <div className="ml-auto flex items-center gap-2">
            <StatusDot status={health} />
            <button
              className="md:hidden p-1 rounded hover:bg-white/5 transition-colors"
              onClick={onClose}
              aria-label="Close sidebar"
            >
              <X size={16} style={{ color: 'var(--text-muted)' }} />
            </button>
          </div>
        </div>

        {/* ── New Chat button ───────────────────────────────────────────────── */}
        <div className="px-4 pb-3">
          <button
            onClick={() => { onNewChat(); onClose(); }}
            className="flex items-center justify-center gap-2 w-full px-3 py-2.5 rounded-lg text-sm font-medium transition-colors"
            style={{ background: 'var(--accent)', color: 'white' }}
            onMouseEnter={e => (e.currentTarget.style.background = 'var(--accent-hover)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'var(--accent)')}
          >
            <Plus size={15} />
            New Chat
          </button>
        </div>

        {/* ── Session history ───────────────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto px-3 pb-2">
          {sessions.length > 0 && (
            <>
              <p
                className="text-xs font-medium mb-1.5 px-2 pt-1"
                style={{ color: 'var(--text-muted)' }}
              >
                History
              </p>
              <div className="space-y-0.5">
                {sessions.map(s => {
                  const active  = s.id === currentSessionId;
                  const hovered = s.id === hoveredSession;
                  return (
                    <div
                      key={s.id}
                      className="group relative flex items-center rounded-lg transition-colors"
                      style={{
                        background: active
                          ? 'rgba(99,102,241,0.15)'
                          : hovered ? 'rgba(255,255,255,0.05)' : 'transparent',
                        borderLeft: active ? '2px solid var(--accent)' : '2px solid transparent',
                        paddingLeft: active ? '0' : '0',
                      }}
                      onMouseEnter={() => setHoveredSession(s.id)}
                      onMouseLeave={() => setHoveredSession(null)}
                    >
                      {editingSession === s.id ? (
                        /* ── Inline rename input ────────────────────────── */
                        <input
                          ref={renameInputRef}
                          className="flex-1 min-w-0 mx-2 my-1 px-2 py-1 rounded text-sm outline-none"
                          style={{
                            background:  'rgba(255,255,255,0.08)',
                            border:      '1px solid var(--accent)',
                            color:       'var(--text-on-dark)',
                          }}
                          value={editingLabel}
                          onChange={e => setEditingLabel(e.target.value)}
                          onBlur={() => commitRename(s.id)}
                          onKeyDown={e => {
                            if (e.key === 'Enter')  { commitRename(s.id); }
                            if (e.key === 'Escape') { setEditingSession(null); }
                          }}
                          autoFocus
                          maxLength={80}
                          onClick={e => e.stopPropagation()}
                        />
                      ) : (
                        /* ── Normal session button ──────────────────────── */
                        <button
                          onClick={() => { onSelectSession(s.id); onClose(); }}
                          onDoubleClick={e => {
                            e.stopPropagation();
                            setEditingSession(s.id);
                            setEditingLabel(s.label);
                            // Focus handled by autoFocus on the input
                          }}
                          className="flex items-center gap-2 flex-1 min-w-0 px-3 py-2 text-sm text-left"
                          style={{ color: active ? 'var(--text-on-dark)' : 'var(--text-muted)' }}
                          title="Double-click to rename"
                        >
                          <MessageSquare size={13} className="flex-shrink-0 opacity-60" />
                          <span className="truncate">{s.label}</span>
                        </button>
                      )}

                      {hovered && editingSession !== s.id && (
                        <div className="flex items-center flex-shrink-0 mr-1">
                          {/* Rename button */}
                          {onRenameSession && (
                            <button
                              onClick={e => {
                                e.stopPropagation();
                                setEditingSession(s.id);
                                setEditingLabel(s.label);
                              }}
                              className="p-1.5 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                              style={{ color: 'var(--text-muted)' }}
                              title="Rename chat"
                              aria-label="Rename chat"
                            >
                              <Pencil size={12} />
                            </button>
                          )}
                          {/* Share button */}
                          <button
                            onClick={async e => {
                              e.stopPropagation();
                              const url = await onShareSession(s.id);
                              if (url) {
                                await navigator.clipboard.writeText(url).catch(() => {});
                                setCopiedSession(s.id);
                                setTimeout(() => setCopiedSession(null), 2000);
                              }
                            }}
                            className="p-1.5 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                            style={{ color: copiedSession === s.id ? 'var(--accent)' : 'var(--text-muted)' }}
                            title={copiedSession === s.id ? 'Link copied!' : 'Share conversation'}
                            aria-label="Share conversation"
                          >
                            {copiedSession === s.id ? <Check size={12} /> : <Link2 size={12} />}
                          </button>
                          {/* Delete button */}
                          <button
                            onClick={e => {
                              e.stopPropagation();
                              onDeleteSession(s.id);
                            }}
                            className="p-1.5 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                            style={{ color: 'var(--text-muted)' }}
                            title="Delete chat"
                            aria-label="Delete chat"
                          >
                            <Trash2 size={12} />
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>

        {/* ── Bottom nav buttons ────────────────────────────────────────────── */}
        <div
          className="px-3 pt-2 border-t"
          style={{ borderColor: 'var(--border-dark)' }}
        >
          {/* Data Sources — admin only */}
          {user.role === 'admin' && (
            <button
              className="flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm transition-colors"
              style={{ color: 'var(--text-muted)' }}
              onClick={() => setCrawlersPanelOpen(true)}
              onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.04)')}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
            >
              <FolderOpen size={14} />
              <span>Data Sources</span>
            </button>
          )}
          <button
            className="flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm transition-colors"
            style={{ color: 'var(--text-muted)' }}
            onClick={() => setSettingsModalOpen(true)}
            onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.04)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
          >
            <Settings2 size={14} />
            <span>Settings</span>
          </button>
        </div>

        {/* ── User info + logout ────────────────────────────────────────────── */}
        <div
          className="px-3 pb-5 pt-2 border-t"
          style={{ borderColor: 'var(--border-dark)' }}
        >
          <div className="flex items-center gap-2.5 px-2 py-2">
            <div
              className="flex items-center justify-center w-7 h-7 rounded-full flex-shrink-0 text-xs font-semibold uppercase"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              {user.username.charAt(0)}
            </div>
            <div className="flex-1 min-w-0 flex items-center gap-1.5">
              <span
                className="text-sm truncate font-medium"
                style={{ color: 'var(--text-on-dark)' }}
              >
                {user.username}
              </span>
              {user.role === 'admin' && (
                <span
                  className="flex-shrink-0 text-xs font-semibold px-1.5 py-0.5 rounded"
                  style={{
                    background: 'rgba(99,102,241,0.18)',
                    color: 'var(--accent)',
                    fontSize: '10px',
                    letterSpacing: '0.04em',
                  }}
                >
                  ADMIN
                </span>
              )}
            </div>
            <button
              onClick={onLogout}
              className="p-1.5 rounded-lg transition-colors flex-shrink-0"
              style={{ color: 'var(--text-muted)' }}
              title="Sign out"
              aria-label="Sign out"
              onMouseEnter={e => {
                (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.07)';
                (e.currentTarget as HTMLButtonElement).style.color = 'var(--danger)';
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
                (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-muted)';
              }}
            >
              <LogOut size={15} />
            </button>
          </div>
        </div>
      </aside>

      {/* ── Settings Modal ─────────────────────────────────────────────────── */}
      {settingsModalOpen && (
        <SettingsModal
          settings={settings}
          onSettingsChange={onSettingsChange}
          theme={theme}
          onToggleTheme={onToggleTheme}
          user={user}
          onClose={() => setSettingsModalOpen(false)}
        />
      )}

      {/* ── Crawlers Panel — admin only ────────────────────────────────────── */}
      {crawlersPanelOpen && user.role === 'admin' && (
        <CrawlersPanel onClose={() => setCrawlersPanelOpen(false)} user={user} />
      )}
    </>
  );
}

// ── Settings Modal ────────────────────────────────────────────────────────────
function SettingsModal({
  settings,
  onSettingsChange,
  theme,
  onToggleTheme,
  user,
  onClose,
}: {
  settings:         Settings;
  onSettingsChange: (s: Settings) => void;
  theme:            'light' | 'dark';
  onToggleTheme:    () => void;
  user:             User;
  onClose:          () => void;
}) {
  const [notifications, setNotifications] = useState(() => localStorage.getItem('customerassist_notif') !== 'false');
  const [docsIndexed, setDocsIndexed]     = useState<number | null>(null);

  // Fetch document count when modal opens
  useEffect(() => {
    fetch(`${API_URL}/index/documents?limit=1`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d?.total != null) setDocsIndexed(d.total); })
      .catch(() => {});
  }, []);

  // Ensure memory, judge, and multi-hop are always on
  useEffect(() => {
    const needsUpdate = !settings.memory || !settings.judge || !settings.multiHop;
    if (needsUpdate) {
      onSettingsChange({ ...settings, memory: true, judge: true, multiHop: true });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleNotifToggle = (v: boolean) => {
    setNotifications(v);
    localStorage.setItem('customerassist_notif', String(v));
  };

  return (
    /* Backdrop */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(2px)' }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      {/* Modal card */}
      <div
        className="relative w-[420px] max-h-[90vh] flex flex-col rounded-2xl shadow-2xl"
        style={{ background: 'var(--bg-sidebar)', border: '1px solid var(--border-dark)' }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-6 py-4 border-b flex-shrink-0"
          style={{ borderColor: 'var(--border-dark)' }}
        >
          <span className="font-semibold text-base" style={{ color: 'var(--text-on-dark)' }}>
            Settings
          </span>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg transition-colors"
            style={{ color: 'var(--text-muted)' }}
            onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.07)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
            aria-label="Close settings"
          >
            <X size={16} />
          </button>
        </div>

        <div className="px-6 py-5 space-y-6 overflow-y-auto flex-1">

          {/* ── APPEARANCE ─────────────────────────────────────────────────── */}
          <Section icon={<Sun size={14} />} title="APPEARANCE">
            <div className="flex items-center justify-between">
              <span className="text-sm" style={{ color: 'var(--text-on-dark)' }}>Theme</span>
              {/* Segmented control */}
              <div
                className="flex rounded-lg p-0.5 gap-0.5"
                style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid var(--border-dark)' }}
              >
                {(['light', 'dark'] as const).map(t => (
                  <button
                    key={t}
                    onClick={() => { if (theme !== t) onToggleTheme(); }}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-150"
                    style={{
                      background: theme === t ? 'var(--accent)' : 'transparent',
                      color:      theme === t ? 'white' : 'var(--text-muted)',
                    }}
                  >
                    {t === 'light' ? <Sun size={11} /> : <Moon size={11} />}
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </button>
                ))}
              </div>
            </div>
          </Section>

          {/* ── NOTIFICATIONS ──────────────────────────────────────────────── */}
          <Section icon={<Bell size={14} />} title="NOTIFICATIONS">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm" style={{ color: 'var(--text-on-dark)' }}>Enable notifications</p>
                <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                  Alerts when indexing completes
                </p>
              </div>
              <Toggle checked={notifications} onChange={handleNotifToggle} />
            </div>
          </Section>

          {/* ── ABOUT ──────────────────────────────────────────────────────── */}
          <Section icon={<Info size={14} />} title="ABOUT">
            <div className="space-y-2.5">
              <AboutRow label="Version"          value="1.0.0" />
              <AboutRow label="Active user"      value={user.username} />
              <AboutRow
                label="Documents indexed"
                value={docsIndexed !== null ? String(docsIndexed) : '—'}
              />
            </div>
          </Section>

        </div>

        {/* ── Done button footer ─────────────────────────────────────────────── */}
        <div
          className="px-6 py-4 border-t flex-shrink-0"
          style={{ borderColor: 'var(--border-dark)' }}
        >
          <button
            onClick={onClose}
            className="w-full py-2.5 rounded-xl text-sm font-semibold transition-colors"
            style={{ background: 'var(--accent)', color: 'white' }}
            onMouseEnter={e => (e.currentTarget.style.background = 'var(--accent-hover)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'var(--accent)')}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Section wrapper ───────────────────────────────────────────────────────────
function Section({
  icon, title, badge, children,
}: {
  icon:      React.ReactNode;
  title:     string;
  badge?:    string;
  children:  React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <span style={{ color: 'var(--text-muted)' }}>{icon}</span>
        <span
          className="text-xs font-semibold tracking-widest"
          style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}
        >
          {title}
        </span>
        {badge && (
          <span
            className="text-[10px] font-bold px-1.5 py-0.5 rounded"
            style={{ background: 'rgba(99,102,241,0.25)', color: 'var(--accent)' }}
          >
            {badge}
          </span>
        )}
      </div>
      <div
        className="rounded-xl p-4 space-y-3"
        style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border-dark)' }}
      >
        {children}
      </div>
    </div>
  );
}

// ── About row ─────────────────────────────────────────────────────────────────
function AboutRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm" style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span className="text-sm font-medium" style={{ color: 'var(--text-on-dark)' }}>{value}</span>
    </div>
  );
}

// ── Toggle switch ─────────────────────────────────────────────────────────────
function Toggle({
  checked,
  onChange,
}: {
  checked:  boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="relative flex-shrink-0 rounded-full transition-colors duration-200"
      style={{
        width:    '44px',
        height:   '24px',
        // overflow hidden ensures the thumb never bleeds past the pill edge
        overflow: 'hidden',
        background: checked ? 'var(--accent)' : 'rgba(255,255,255,0.18)',
        boxShadow: checked
          ? 'inset 0 1px 3px rgba(0,0,0,0.25)'
          : 'inset 0 1px 3px rgba(0,0,0,0.30)',
        outline: 'none',
      }}
    >
      {/* Thumb — positioned via left to avoid transform-origin issues */}
      <span
        className="absolute rounded-full bg-white transition-all duration-200"
        style={{
          width:  '18px',
          height: '18px',
          top:    '3px',
          left:   checked ? '23px' : '3px',
          boxShadow: '0 1px 4px rgba(0,0,0,0.40)',
        }}
      />
    </button>
  );
}

// ── Always-on indicator row ───────────────────────────────────────────────────
function AlwaysOnRow({ label, description }: { label: string; description: string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="min-w-0">
        <p className="text-sm" style={{ color: 'var(--text-on-dark)' }}>{label}</p>
        <p className="text-xs mt-0.5 truncate" style={{ color: 'var(--text-muted)' }}>{description}</p>
      </div>
      <span
        className="flex-shrink-0 text-[10px] font-semibold px-2 py-1 rounded-full cursor-default"
        title="Always enabled — cannot be turned off in this build"
        style={{
          background: 'rgba(99,102,241,0.18)',
          color: 'var(--accent)',
          letterSpacing: '0.04em',
        }}
      >
        ON
      </span>
    </div>
  );
}
