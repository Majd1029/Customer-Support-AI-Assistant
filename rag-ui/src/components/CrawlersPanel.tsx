import { useState, useEffect, useRef } from 'react';
import {
  X, Mail, HardDrive, Play, RefreshCw, ChevronDown,
  CheckCircle, XCircle, Loader, Clock, FolderOpen, User as UserIcon,
  Globe, Users, Lock,
} from 'lucide-react';
import type { User } from '../types';
import { friendlyFetchError, friendlyError } from '../utils/errors';

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';
const POLL_INTERVAL_MS = 2000;

// localStorage key for persisting the user's Google email across sessions
const googleEmailKey = (userId: string) => `docassist_google_email_${userId}`;

// ── Types ─────────────────────────────────────────────────────────────────────

interface CrawlJob {
  job_id:      string;
  source:      'gmail' | 'drive';
  status:      'running' | 'done' | 'failed';
  total:       number;
  indexed:     number;
  skipped:     number;
  errors:      number;
  current:     string;
  started_at:  string;
  finished_at: string | null;
  error:       string | null;
  config:      Record<string, unknown>;
}

interface DriveFile {
  file_id:       string;
  name:          string;
  mime_type:     string;
  owner_email:   string;
  allowed_users: string[];
  is_public:     boolean;
  crawled_at:    string | null;
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function CrawlersPanel({
  onClose,
  user,
}: {
  onClose: () => void;
  user:    User;
}) {
  const [tab, setTab]         = useState<'gmail' | 'drive'>('gmail');
  const [activeJob, setActiveJob] = useState<CrawlJob | null>(null);
  const [history, setHistory] = useState<CrawlJob[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [driveFiles, setDriveFiles]         = useState<DriveFile[]>([]);
  const [driveFilesLoading, setDriveFilesLoading] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const pollRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const stoppedRef = useRef(false);

  // Google email — persisted in localStorage per user so it survives refresh
  const [googleEmail, setGoogleEmail] = useState<string>(() => {
    try { return localStorage.getItem(googleEmailKey(user.id)) ?? ''; }
    catch { return ''; }
  });

  const saveGoogleEmail = (v: string) => {
    setGoogleEmail(v);
    try { localStorage.setItem(googleEmailKey(user.id), v); } catch { /* ignore */ }
    if (v) fetchDriveFiles(v);
  };

  // Load history + drive files on mount
  useEffect(() => {
    fetchHistory();
    if (googleEmail) fetchDriveFiles(googleEmail);
  }, []);

  // Poll active job
  useEffect(() => {
    if (activeJob?.status !== 'running') return;

    stoppedRef.current = false;

    // One-shot stop helper — idempotent; safe to call from both inside the
    // async callback and from the effect cleanup.
    const stop = () => {
      stoppedRef.current = true;
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };

    pollRef.current = setInterval(async () => {
      // Guard against ticks that fire after we've already stopped
      if (stoppedRef.current) return;

      try {
        const r = await fetch(`${API_URL}/crawl/status/${activeJob.job_id}`, {
          headers: user.token ? { Authorization: `Bearer ${user.token}` } : {},
        });

        // Guard again — we may have been stopped while the fetch was in-flight
        if (stoppedRef.current) return;

        if (r.status === 404) {
          // Job left the server's active-job registry → treat as done
          stop();
          setActiveJob(prev => {
            if (prev?.source === 'drive') fetchDriveFiles(googleEmail);
            return prev ? { ...prev, status: 'done' } : null;
          });
          fetchHistory();
        } else if (r.ok) {
          const data: CrawlJob = await r.json();
          setActiveJob(data);
          if (data.status !== 'running') {
            stop();
            if (data.source === 'drive') fetchDriveFiles(googleEmail);
            fetchHistory();
          }
        }
      } catch {
        // ignore transient network errors
      }
    }, POLL_INTERVAL_MS);

    return stop;
  }, [activeJob?.job_id, activeJob?.status]);

  const authHeaders = (): Record<string, string> =>
    user.token ? { Authorization: `Bearer ${user.token}` } : {};

  async function fetchHistory() {
    setHistoryLoading(true);
    try {
      const r = await fetch(`${API_URL}/crawl/history?limit=20`, {
        headers: authHeaders(),
      });
      if (r.ok) {
        const d = await r.json();
        setHistory(d.runs ?? []);
      }
    } catch { /* ignore */ }
    setHistoryLoading(false);
  }

  async function fetchDriveFiles(email: string) {
    if (!email) return;
    setDriveFilesLoading(true);
    try {
      const r = await fetch(
        `${API_URL}/drive/files?owner_email=${encodeURIComponent(email)}&limit=200`,
        { headers: authHeaders() },
      );
      if (r.ok) {
        const d = await r.json();
        setDriveFiles(d.files ?? []);
      }
    } catch { /* ignore */ }
    setDriveFilesLoading(false);
  }

  async function startCrawl(source: 'gmail' | 'drive', body: Record<string, unknown>) {
    setStartError(null);
    try {
      const r = await fetch(`${API_URL}/crawl/${source}`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body:    JSON.stringify(body),
      });
      if (!r.ok) {
        const msg = await friendlyFetchError(r);
        setStartError(msg);
        return;
      }
      const data = await r.json();
      setActiveJob({
        job_id:      data.job_id,
        source,
        status:      'running',
        total:       0, indexed: 0, skipped: 0, errors: 0,
        current:     '',
        started_at:  new Date().toISOString(),
        finished_at: null,
        error:       null,
        config:      body,
      });
    } catch (e) {
      setStartError(friendlyError(0, (e as Error).message || '') || '⚠️ Could not reach the server. Check your connection.');
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(2px)' }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="relative w-[520px] max-h-[90vh] overflow-y-auto rounded-2xl shadow-2xl flex flex-col"
        style={{ background: 'var(--bg-sidebar)', border: '1px solid var(--border-dark)' }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-6 py-4 border-b flex-shrink-0"
          style={{ borderColor: 'var(--border-dark)' }}
        >
          <div className="flex items-center gap-2.5">
            <FolderOpen size={16} style={{ color: 'var(--accent)' }} />
            <span className="font-semibold text-base" style={{ color: 'var(--text-on-dark)' }}>
              Data Sources
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg transition-colors text-[var(--text-muted)] hover:bg-[rgba(255,255,255,0.07)]"
            title="Close data sources panel"
            aria-label="Close data sources panel"
          >
            <X size={16} />
          </button>
        </div>

        <div className="px-6 py-5 space-y-5 flex-1 overflow-y-auto">

          {/* ── Owner identity card ────────────────────────────────────────── */}
          <OwnerCard
            userId={user.id}
            username={user.username}
            googleEmail={googleEmail}
            onEmailChange={saveGoogleEmail}
          />

          {/* ── Active job status ─────────────────────────────────────────── */}
          {activeJob && <ActiveJobCard job={activeJob} />}

          {/* ── Source tabs ───────────────────────────────────────────────── */}
          <div>
            <div
              className="flex rounded-xl p-1 mb-4"
              style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-dark)' }}
            >
              {(['gmail', 'drive'] as const).map(t => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className="flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-all duration-150"
                  style={{
                    background: tab === t ? 'var(--accent)' : 'transparent',
                    color:      tab === t ? 'white' : 'var(--text-muted)',
                  }}
                >
                  {t === 'gmail' ? <Mail size={13} /> : <HardDrive size={13} />}
                  {t === 'gmail' ? 'Gmail' : 'Google Drive'}
                </button>
              ))}
            </div>

            {tab === 'gmail'
              ? (
                <GmailForm
                  onStart={body => { setStartError(null); startCrawl('gmail', {
                    ...body,
                    user_id:    user.id,
                    user_email: googleEmail,
                  }); }}
                  disabled={activeJob?.status === 'running'}
                />
              ) : (
                <DriveForm
                  onStart={body => { setStartError(null); startCrawl('drive', {
                    ...body,
                    user_id:    user.id,
                    user_email: googleEmail,
                  }); }}
                  disabled={activeJob?.status === 'running'}
                />
              )
            }
            {startError && (
              <p style={{ color: 'var(--danger, #ef4444)', fontSize: '0.85rem', marginTop: '0.5rem', padding: '0.5rem 0.75rem', background: 'rgba(239,68,68,0.08)', borderRadius: '6px' }}>
                {startError}
              </p>
            )}
          </div>

          {/* ── History ───────────────────────────────────────────────────── */}
          <HistorySection
            runs={history}
            loading={historyLoading}
            onRefresh={fetchHistory}
          />

          {/* ── Drive indexed files with permission badges ─────────────────── */}
          {googleEmail && (
            <DriveFilesSection
              files={driveFiles}
              loading={driveFilesLoading}
              onRefresh={() => fetchDriveFiles(googleEmail)}
            />
          )}

        </div>
      </div>
    </div>
  );
}

// ── Owner identity card ───────────────────────────────────────────────────────

function OwnerCard({
  userId,
  username,
  googleEmail,
  onEmailChange,
}: {
  userId:       string;
  username:     string;
  googleEmail:  string;
  onEmailChange:(v: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft,   setDraft]   = useState(googleEmail);

  const handleSave = () => {
    onEmailChange(draft.trim());
    setEditing(false);
  };

  return (
    <div
      className="rounded-xl p-4"
      style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border-dark)' }}
    >
      {/* Title row */}
      <div className="flex items-center gap-2 mb-3">
        <UserIcon size={13} style={{ color: 'var(--accent)' }} />
        <span className="text-xs font-semibold tracking-widest" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
          CRAWL OWNER
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-3">
        {/* Account (read-only — app username) */}
        <div>
          <p className="text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>App account</p>
          <div
            className="px-3 py-2 rounded-lg text-sm"
            style={{ background: 'rgba(255,255,255,0.06)', color: 'var(--text-muted)' }}
          >
            {username}
            <span className="ml-1 text-[10px] opacity-60">({userId})</span>
          </div>
        </div>

        {/* Gmail address (editable, saved to localStorage) */}
        <div>
          <p className="text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>Gmail address</p>
          {editing ? (
            <div className="flex gap-1">
              <input
                autoFocus
                type="email"
                value={draft}
                onChange={e => setDraft(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') handleSave(); if (e.key === 'Escape') setEditing(false); }}
                placeholder="you@gmail.com"
                className="flex-1 min-w-0 px-2 py-1.5 rounded-lg text-sm border outline-none"
                style={{ background: 'rgba(255,255,255,0.08)', color: 'var(--text-on-dark)', borderColor: 'var(--accent)' }}
              />
              <button
                onClick={handleSave}
                className="px-2 rounded-lg text-xs font-medium"
                style={{ background: 'var(--accent)', color: 'white' }}
              >
                Save
              </button>
            </div>
          ) : (
            <button
              onClick={() => { setDraft(googleEmail); setEditing(true); }}
              className="w-full text-left px-3 py-2 rounded-lg text-sm border transition-colors"
              style={{
                background:  'rgba(255,255,255,0.06)',
                color:       googleEmail ? 'var(--text-on-dark)' : 'var(--text-muted)',
                borderColor: 'var(--border-dark)',
              }}
              onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--accent)')}
              onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border-dark)')}
            >
              {googleEmail || <span className="opacity-50">Click to set…</span>}
            </button>
          )}
        </div>
      </div>

      {/* Info note */}
      <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-muted)' }}>
        Crawled content will be tagged with your account ID in Qdrant so it can be
        filtered to your data at query time. Set your Gmail address to enable
        per-user access control.
      </p>
    </div>
  );
}

// ── Active job card ───────────────────────────────────────────────────────────

function ActiveJobCard({ job }: { job: CrawlJob }) {
  const pct = job.total > 0
    ? Math.round((job.indexed + job.errors) / job.total * 100)
    : null;

  const statusColor =
    job.status === 'done'   ? 'var(--success, #22c55e)' :
    job.status === 'failed' ? 'var(--danger,  #ef4444)' :
    'var(--accent)';

  return (
    <div
      className="rounded-xl p-4"
      style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border-dark)' }}
    >
      <div className="flex items-center gap-2 mb-3">
        {job.status === 'running' && <Loader size={13} className="animate-spin" style={{ color: 'var(--accent)' }} />}
        {job.status === 'done'    && <CheckCircle size={13} style={{ color: statusColor }} />}
        {job.status === 'failed'  && <XCircle     size={13} style={{ color: statusColor }} />}
        <span className="text-sm font-medium" style={{ color: 'var(--text-on-dark)' }}>
          {job.source === 'gmail' ? 'Gmail' : 'Google Drive'} crawl
          {job.status === 'running' ? ' in progress…' : job.status === 'done' ? ' complete' : ' failed'}
        </span>
        {pct !== null && (
          <span className="ml-auto text-xs" style={{ color: 'var(--text-muted)' }}>{pct}%</span>
        )}
      </div>

      {/* Progress bar */}
      {job.total > 0 && (
        <div
          className="w-full h-1.5 rounded-full mb-3 overflow-hidden"
          style={{ background: 'rgba(255,255,255,0.1)' }}
        >
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${pct}%`, background: statusColor }}
          />
        </div>
      )}

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-2 text-center mb-2">
        {[
          { label: 'Indexed',  value: job.indexed, color: 'var(--success, #22c55e)' },
          { label: 'Skipped',  value: job.skipped, color: 'var(--text-muted)' },
          { label: 'Errors',   value: job.errors,  color: job.errors > 0 ? 'var(--danger, #ef4444)' : 'var(--text-muted)' },
        ].map(s => (
          <div key={s.label}
            className="rounded-lg py-2"
            style={{ background: 'rgba(255,255,255,0.03)' }}
          >
            <p className="text-base font-semibold" style={{ color: s.color }}>{s.value}</p>
            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{s.label}</p>
          </div>
        ))}
      </div>

      {/* Current file */}
      {job.current && (
        <p className="text-xs truncate" style={{ color: 'var(--text-muted)' }}>
          Processing: {job.current}
        </p>
      )}
      {job.error && (
        <p className="text-xs mt-1" style={{ color: 'var(--danger, #ef4444)' }}>
          Error: {job.error}
        </p>
      )}
    </div>
  );
}

// ── Gmail form ────────────────────────────────────────────────────────────────

function GmailForm({
  onStart,
  disabled,
}: {
  onStart:  (body: Record<string, unknown>) => void;
  disabled: boolean;
}) {
  const [label,      setLabel]      = useState('INBOX');
  const [query,      setQuery]      = useState('');
  const [after,      setAfter]      = useState('');
  const [before,     setBefore]     = useState('');
  const [maxResults, setMaxResults] = useState(100);
  const [advanced,   setAdvanced]   = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onStart({
      label:       label || null,
      query:       query  || null,
      after:       after  || null,
      before:      before || null,
      max_results: maxResults,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <Field label="Label">
        <input
          type="text"
          value={label}
          onChange={e => setLabel(e.target.value)}
          placeholder="INBOX"
          className="w-full px-3 py-2 rounded-lg text-sm border outline-none"
          style={fieldStyle}
        />
      </Field>

      {/* Advanced toggle */}
      <button
        type="button"
        onClick={() => setAdvanced(v => !v)}
        className="flex items-center gap-1.5 text-xs transition-colors"
        style={{ color: 'var(--text-muted)' }}
      >
        <ChevronDown
          size={12}
          style={{ transform: advanced ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}
        />
        {advanced ? 'Hide' : 'Show'} advanced filters
      </button>

      {advanced && (
        <div className="space-y-3 pt-1">
          <Field label="Search query">
            <input
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder='e.g. from:boss@example.com has:attachment'
              className="w-full px-3 py-2 rounded-lg text-sm border outline-none"
              style={fieldStyle}
            />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="After date">
              <input type="date" value={after} onChange={e => setAfter(e.target.value)}
                className="w-full px-3 py-2 rounded-lg text-sm border outline-none" style={fieldStyle} />
            </Field>
            <Field label="Before date">
              <input type="date" value={before} onChange={e => setBefore(e.target.value)}
                className="w-full px-3 py-2 rounded-lg text-sm border outline-none" style={fieldStyle} />
            </Field>
          </div>
          <Field label="Max emails">
            <input
              type="number" min={1} max={1000}
              value={maxResults} onChange={e => setMaxResults(Number(e.target.value))}
              className="w-full px-3 py-2 rounded-lg text-sm border outline-none" style={fieldStyle}
            />
          </Field>
        </div>
      )}

      <StartButton disabled={disabled} />
    </form>
  );
}

// ── Drive form ────────────────────────────────────────────────────────────────

function DriveForm({
  onStart,
  disabled,
}: {
  onStart:  (body: Record<string, unknown>) => void;
  disabled: boolean;
}) {
  const [folderId,      setFolderId]      = useState('');
  const [recursive,     setRecursive]     = useState(false);
  const [incremental,   setIncremental]   = useState(true);   // skip unchanged files by default
  const [modifiedAfter, setModifiedAfter] = useState('');
  const [types,         setTypes]         = useState('');
  const [maxResults,    setMaxResults]    = useState(200);
  const [advanced,      setAdvanced]      = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onStart({
      folder_id:      folderId || null,
      recursive,
      incremental,
      modified_after: modifiedAfter || null,
      types:          types ? types.split(',').map(t => t.trim()).filter(Boolean) : null,
      max_results:    maxResults,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <Field label="Folder ID" hint="Leave blank for My Drive root">
        <input
          type="text"
          value={folderId}
          onChange={e => setFolderId(e.target.value)}
          placeholder="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"
          className="w-full px-3 py-2 rounded-lg text-sm border outline-none"
          style={fieldStyle}
        />
      </Field>

      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm" style={{ color: 'var(--text-on-dark)' }}>Recursive</p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Include sub-folders</p>
        </div>
        <RecursiveToggle checked={recursive} onChange={setRecursive} />
      </div>

      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm" style={{ color: 'var(--text-on-dark)' }}>Incremental</p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Skip unchanged files</p>
        </div>
        <RecursiveToggle checked={incremental} onChange={setIncremental} />
      </div>

      {/* Advanced toggle */}
      <button
        type="button"
        onClick={() => setAdvanced(v => !v)}
        className="flex items-center gap-1.5 text-xs transition-colors"
        style={{ color: 'var(--text-muted)' }}
      >
        <ChevronDown
          size={12}
          style={{ transform: advanced ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}
        />
        {advanced ? 'Hide' : 'Show'} advanced filters
      </button>

      {advanced && (
        <div className="space-y-3 pt-1">
          <Field label="File types" hint="Comma-separated: pdf, docx, pptx, xlsx, txt">
            <input
              type="text"
              value={types}
              onChange={e => setTypes(e.target.value)}
              placeholder="pdf, docx, pptx"
              className="w-full px-3 py-2 rounded-lg text-sm border outline-none"
              style={fieldStyle}
            />
          </Field>
          <Field label="Modified after">
            <input type="date" value={modifiedAfter} onChange={e => setModifiedAfter(e.target.value)}
              className="w-full px-3 py-2 rounded-lg text-sm border outline-none" style={fieldStyle} />
          </Field>
          <Field label="Max files">
            <input
              type="number" min={1} max={1000}
              value={maxResults} onChange={e => setMaxResults(Number(e.target.value))}
              className="w-full px-3 py-2 rounded-lg text-sm border outline-none" style={fieldStyle}
            />
          </Field>
        </div>
      )}

      <StartButton disabled={disabled} />
    </form>
  );
}

// ── History section ───────────────────────────────────────────────────────────

function HistorySection({
  runs,
  loading,
  onRefresh,
}: {
  runs:      CrawlJob[];
  loading:   boolean;
  onRefresh: () => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold tracking-widest" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
          CRAWL HISTORY
        </span>
        <button
          onClick={onRefresh}
          className="p-1 rounded transition-colors"
          title="Refresh history"
          style={{ color: 'var(--text-muted)' }}
          onMouseEnter={e => (e.currentTarget.style.color = 'var(--accent)')}
          onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-muted)')}
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {loading ? (
        <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>Loading…</p>
      ) : runs.length === 0 ? (
        <div
          className="rounded-xl p-6 text-center"
          style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-dark)' }}
        >
          <Clock size={20} className="mx-auto mb-2 opacity-30" style={{ color: 'var(--text-muted)' }} />
          <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-on-dark)' }}>No crawls yet</p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Start a Gmail or Drive crawl above to index your emails and files.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {runs.map((run, i) => (
            <HistoryRow key={run.job_id ?? i} run={run} />
          ))}
        </div>
      )}
    </div>
  );
}

function HistoryRow({ run }: { run: CrawlJob }) {
  const icon  = run.source === 'gmail' ? <Mail size={12} /> : <HardDrive size={12} />;
  const label = run.source === 'gmail' ? 'Gmail' : 'Drive';
  const date  = run.finished_at
    ? new Date(run.finished_at).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })
    : new Date(run.started_at).toLocaleString(undefined,  { dateStyle: 'short', timeStyle: 'short' });

  // Show owner from the job config if present
  const owner = (run.config?.user_email as string | undefined) || (run.config?.user_id as string | undefined);

  const statusIcon =
    run.status === 'done'   ? <CheckCircle size={12} style={{ color: 'var(--success, #22c55e)' }} /> :
    run.status === 'failed' ? <XCircle     size={12} style={{ color: 'var(--danger,  #ef4444)' }} /> :
    <Loader size={12} className="animate-spin" style={{ color: 'var(--accent)' }} />;

  return (
    <div
      className="flex items-center gap-3 px-3 py-2.5 rounded-lg"
      style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-dark)' }}
    >
      <span style={{ color: 'var(--text-muted)' }}>{icon}</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium" style={{ color: 'var(--text-on-dark)' }}>{label}</span>
          {statusIcon}
        </div>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {run.indexed} indexed
          {run.skipped > 0 && <span className="ml-1">· {run.skipped} skipped</span>}
          {run.errors  > 0 && <span className="ml-1" style={{ color: 'var(--danger, #ef4444)' }}>· {run.errors} errors</span>}
          <span className="ml-1">· {date}</span>
          {owner && <span className="ml-1 opacity-60">· {owner}</span>}
        </p>
        {run.error && (
          <p className="text-xs truncate mt-0.5" style={{ color: 'var(--danger, #ef4444)' }}>
            {run.error}
          </p>
        )}
      </div>
      <div className="text-right flex-shrink-0">
        <p className="text-sm font-semibold" style={{ color: 'var(--text-on-dark)' }}>
          {run.total}
        </p>
        <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>total</p>
      </div>
    </div>
  );
}

// ── Drive indexed files section ───────────────────────────────────────────────

function DriveFilesSection({
  files,
  loading,
  onRefresh,
}: {
  files:     DriveFile[];
  loading:   boolean;
  onRefresh: () => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <HardDrive size={12} style={{ color: 'var(--accent)' }} />
          <span className="text-xs font-semibold tracking-widest" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
            INDEXED DRIVE FILES
          </span>
          {files.length > 0 && (
            <span
              className="px-1.5 py-0.5 rounded-full text-[10px] font-semibold"
              style={{ background: 'rgba(99,102,241,0.2)', color: 'var(--accent)' }}
            >
              {files.length}
            </span>
          )}
        </div>
        <button
          onClick={onRefresh}
          className="p-1 rounded transition-colors"
          title="Refresh"
          style={{ color: 'var(--text-muted)' }}
          onMouseEnter={e => (e.currentTarget.style.color = 'var(--accent)')}
          onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-muted)')}
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {loading ? (
        <p className="text-xs text-center py-3" style={{ color: 'var(--text-muted)' }}>Loading…</p>
      ) : files.length === 0 ? (
        <div
          className="rounded-xl p-4 text-center"
          style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-dark)' }}
        >
          <HardDrive size={18} className="mx-auto mb-2 opacity-25" style={{ color: 'var(--text-muted)' }} />
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            No Drive files indexed yet. Run a Drive crawl to see files here.
          </p>
        </div>
      ) : (
        <div className="space-y-1.5 max-h-64 overflow-y-auto pr-0.5">
          {files.map(f => (
            <DriveFileRow key={f.file_id} file={f} />
          ))}
        </div>
      )}
    </div>
  );
}

function DriveFileRow({ file }: { file: DriveFile }) {
  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  const extLabel = ext ? `.${ext}` : '';

  return (
    <div
      className="flex items-center gap-2.5 px-3 py-2 rounded-lg"
      style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-dark)' }}
    >
      {/* File name + extension */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="text-xs truncate" style={{ color: 'var(--text-on-dark)' }} title={file.name}>
            {file.name}
          </span>
          {extLabel && (
            <span
              className="flex-shrink-0 px-1 py-0 rounded text-[9px] font-mono uppercase"
              style={{ background: 'rgba(255,255,255,0.08)', color: 'var(--text-muted)' }}
            >
              {extLabel}
            </span>
          )}
        </div>
        {file.crawled_at && (
          <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
            {new Date(file.crawled_at).toLocaleDateString(undefined, { dateStyle: 'short' })}
          </p>
        )}
      </div>

      {/* Permission badge */}
      <PermissionBadge isPublic={file.is_public} allowedUsers={file.allowed_users} />
    </div>
  );
}

function PermissionBadge({
  isPublic,
  allowedUsers,
}: {
  isPublic:     boolean;
  allowedUsers: string[];
}) {
  if (isPublic) {
    return (
      <div
        className="flex items-center gap-1 px-2 py-1 rounded-full flex-shrink-0"
        style={{ background: 'rgba(34,197,94,0.12)', color: '#22c55e' }}
        title="Anyone with the link can access this file"
      >
        <Globe size={10} />
        <span className="text-[10px] font-medium">Public</span>
      </div>
    );
  }

  // More than 1 means it's shared (owner + others)
  if (allowedUsers.length > 1) {
    return (
      <div
        className="flex items-center gap-1 px-2 py-1 rounded-full flex-shrink-0"
        style={{ background: 'rgba(251,191,36,0.12)', color: '#fbbf24' }}
        title={`Shared with: ${allowedUsers.join(', ')}`}
      >
        <Users size={10} />
        <span className="text-[10px] font-medium">Shared {allowedUsers.length}</span>
      </div>
    );
  }

  return (
    <div
      className="flex items-center gap-1 px-2 py-1 rounded-full flex-shrink-0"
      style={{ background: 'rgba(148,163,184,0.12)', color: 'var(--text-muted)' }}
      title="Only you can access this file"
    >
      <Lock size={10} />
      <span className="text-[10px] font-medium">Private</span>
    </div>
  );
}

// ── Shared sub-components ─────────────────────────────────────────────────────

const fieldStyle: React.CSSProperties = {
  background:  'rgba(255,255,255,0.05)',
  color:       'var(--text-on-dark)',
  borderColor: 'var(--border-dark)',
};

function Field({
  label,
  hint,
  children,
}: {
  label:    string;
  hint?:    string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-baseline gap-2 mb-1">
        <label className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>{label}</label>
        {hint && <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{hint}</span>}
      </div>
      {children}
    </div>
  );
}

function RecursiveToggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      style={{
        position:     'relative',
        width:        '44px',
        height:       '24px',
        borderRadius: '9999px',
        border:       'none',
        padding:      0,
        flexShrink:   0,
        cursor:       'pointer',
        transition:   'background 0.2s',
        background:   checked ? 'var(--accent)' : 'rgba(255,255,255,0.15)',
      }}
    >
      <span
        style={{
          position:     'absolute',
          top:          '4px',
          left:         '4px',
          width:        '16px',
          height:       '16px',
          borderRadius: '50%',
          background:   'white',
          boxShadow:    '0 1px 3px rgba(0,0,0,0.3)',
          transition:   'transform 0.2s',
          transform:    checked ? 'translateX(20px)' : 'translateX(0)',
        }}
      />
    </button>
  );
}

function StartButton({ disabled }: { disabled: boolean }) {
  return (
    <button
      type="submit"
      disabled={disabled}
      className="flex items-center justify-center gap-2 w-full py-2.5 rounded-lg text-sm font-medium transition-colors mt-1"
      style={{
        background: disabled ? 'rgba(99,102,241,0.35)' : 'var(--accent)',
        color:      disabled ? 'rgba(255,255,255,0.45)' : 'white',
        cursor:     disabled ? 'not-allowed' : 'pointer',
      }}
    >
      <Play size={13} />
      {disabled ? 'Crawl running…' : 'Start Crawl'}
    </button>
  );
}
