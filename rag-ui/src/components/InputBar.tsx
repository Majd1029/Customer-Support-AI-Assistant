import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Send, Square, Paperclip, X, CheckCircle, AlertCircle, Loader,
  BookOpen, FileText, Database, ChevronDown, Globe, Users, Lock,
} from 'lucide-react';
import type { ScopeState } from '../types';
import { friendlyFetchError, friendlyError } from '../utils/errors';

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';

// localStorage key used by CrawlersPanel to persist the user's Google email
const googleEmailKey = (userId: string) => `customerassist_google_email_${userId}`;

interface DrivePermInfo {
  is_public:     boolean;
  allowed_users: string[];
}

type UploadStatus =
  | { state: 'idle' }
  | { state: 'uploading';  filename: string }
  | { state: 'processing'; filename: string }   // OCR / extraction running server-side
  | { state: 'indexing';   filename: string }
  | { state: 'done';       filename: string }
  | { state: 'error';      message: string };

interface InputBarProps {
  onSend:           (text: string) => void;
  onStop?:          () => void;
  disabled:         boolean;
  prefillText?:     string;
  onPrefillUsed?:   () => void;
  scope:            ScopeState;
  onScopeChange:    (s: ScopeState) => void;
  indexedDocs:      string[];
  onDocIndexed:     () => void;
  /** The logged-in user's id — stamped as owner_id on every uploaded file. */
  userId?:          string;
}

export default function InputBar({
  onSend,
  onStop,
  disabled,
  prefillText,
  onPrefillUsed,
  scope,
  onScopeChange,
  indexedDocs,
  onDocIndexed,
  userId,
}: InputBarProps) {
  const [value, setValue]           = useState('');
  const [upload, setUpload]         = useState<UploadStatus>({ state: 'idle' });
  const [docPickerOpen, setDocPickerOpen] = useState(false);
  // Map of doc filename → Drive permission info (loaded lazily on picker open)
  const [drivePerms, setDrivePerms] = useState<Record<string, DrivePermInfo>>({});
  const textareaRef                 = useRef<HTMLTextAreaElement>(null);
  const fileInputRef                = useRef<HTMLInputElement>(null);
  const pickerRef                   = useRef<HTMLDivElement>(null);
  const pollIntervalRef             = useRef<ReturnType<typeof setInterval> | null>(null);
  const MAX_CHARS                   = 2000;

  // Cancel any in-flight poll timer when the component unmounts
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current !== null) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  // Consume prefill text from parent
  useEffect(() => {
    if (prefillText) {
      setValue(prefillText);
      onPrefillUsed?.();
      textareaRef.current?.focus();
    }
  }, [prefillText, onPrefillUsed]);

  // Auto-resize textarea (1–6 rows)
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 6 * 24 + 16)}px`;
  }, [value]);

  // Auto-dismiss success/error status after 4 s
  useEffect(() => {
    if (upload.state === 'done' || upload.state === 'error') {
      const id = setTimeout(() => setUpload({ state: 'idle' }), 4000);
      return () => clearTimeout(id);
    }
  }, [upload.state]);

  // Close doc picker when clicking outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) {
        setDocPickerOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Fetch Drive permission info for all indexed docs (lazy, on picker open)
  const fetchDrivePerms = useCallback(async () => {
    if (!userId || userId === '__guest__') return;
    try {
      const email = localStorage.getItem(googleEmailKey(userId)) ?? '';
      if (!email) return;
      const r = await fetch(`${API_URL}/drive/files?owner_email=${encodeURIComponent(email)}&limit=500`);
      if (!r.ok) return;
      const data = await r.json() as { files?: Array<{ name: string; is_public: boolean; allowed_users: string[] }> };
      const map: Record<string, DrivePermInfo> = {};
      for (const f of data.files ?? []) {
        map[f.name] = { is_public: f.is_public, allowed_users: f.allowed_users };
      }
      setDrivePerms(map);
    } catch { /* ignore — permissions are UI-only enhancement */ }
  }, [userId]);

  const handleSend = useCallback(() => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  }, [value, disabled, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!fileInputRef.current) return;
    fileInputRef.current.value = '';
    if (!file) return;

    // Cancel any previous poll that may still be running
    if (pollIntervalRef.current !== null) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }

    setUpload({ state: 'uploading', filename: file.name });

    try {
      // ── 1. Send file with async_mode=true ─────────────────────────────────
      // The server returns a job_id immediately (HTTP 202) without waiting for
      // OCR/extraction to finish, so we never hit the gateway 504 timeout.
      const formData = new FormData();
      formData.append('file', file);
      formData.append('async_mode', 'true');
      // Tag the upload with the current user so documents are per-user isolated.
      if (userId && userId !== '__guest__') {
        formData.append('owner_id', userId);
      }

      const uploadRes = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body:   formData,
      });
      if (!uploadRes.ok) {
        const msg = await friendlyFetchError(uploadRes);
        setUpload({ state: 'error', message: msg });
        return;
      }

      const uploadData = (await uploadRes.json()) as {
        job_id?:      string;
        result_file?: string;   // present if server somehow responded synchronously
        error?:       string;
      };

      // ── 2. Poll GET /tasks/{job_id} until extraction is complete ───────────
      let resultFile: string;

      if (uploadData.job_id) {
        setUpload({ state: 'processing', filename: file.name });

        resultFile = await new Promise<string>((resolve, reject) => {
          const poll = async () => {
            try {
              const pollRes = await fetch(`${API_URL}/tasks/${uploadData.job_id}`);

              if (pollRes.status === 404) {
                clearInterval(pollIntervalRef.current!);
                pollIntervalRef.current = null;
                reject(new Error('⚠️ The processing job expired. Please try uploading again.'));
                return;
              }
              if (!pollRes.ok) return; // transient network hiccup, keep polling

              const job = (await pollRes.json()) as {
                status:       string;
                result_file?: string;
                error?:       string;
              };

              if (job.status === 'done') {
                // Resolve with result_file (may be empty string if worker
                // already auto-indexed — that's fine, we skip POST /index below)
                clearInterval(pollIntervalRef.current!);
                pollIntervalRef.current = null;
                resolve(job.result_file ?? '');
              } else if (job.status === 'failed') {
                clearInterval(pollIntervalRef.current!);
                pollIntervalRef.current = null;
                reject(new Error(friendlyError(500, job.error ?? '') || '⚠️ File processing failed. Please try again.'));
              }
              // 'pending' / 'processing' → keep waiting
            } catch {
              // network error during poll — keep trying
            }
          };

          pollIntervalRef.current = setInterval(poll, 2000);
          // Run immediately once so we don't wait 2s on fast (non-PDF) files
          void poll();
        });

      } else if (uploadData.result_file) {
        // Server responded synchronously (shouldn't happen with async_mode=true,
        // but handle gracefully so we don't break if the API changes).
        resultFile = uploadData.result_file;
      } else {
        setUpload({ state: 'error', message: friendlyError(500, uploadData.error ?? '') || '⚠️ No result from server. Please try again.' });
        return;
      }

      // ── 3. Embed + index the extracted chunks into Qdrant ──────────────────
      // When using Celery (async path), the worker already auto-indexed during
      // extraction.  result_file is empty in that case — skip the redundant call.
      if (resultFile) {
        setUpload({ state: 'indexing', filename: file.name });
        const indexRes = await fetch(`${API_URL}/index`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ result_file: resultFile }),
        });
        if (!indexRes.ok) {
          const msg = await friendlyFetchError(indexRes);
          setUpload({ state: 'error', message: msg });
          return;
        }
      }

      setUpload({ state: 'done', filename: file.name });
      await onDocIndexed();
      onScopeChange({ mode: 'specific', doc: file.name });

    } catch (err) {
      if (pollIntervalRef.current !== null) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      const raw = (err as Error).message || '';
      setUpload({ state: 'error', message: friendlyError(0, raw) || '⚠️ Could not reach the server. Check your connection.' });
    }
  }, [onDocIndexed, onScopeChange, userId]);

  const showCounter = value.length > 200;
  const overLimit   = value.length > MAX_CHARS;
  const busy        = upload.state === 'uploading' || upload.state === 'processing' || upload.state === 'indexing';

  // ── Scope label helper ─────────────────────────────────────────────────────
  const scopeLabel =
    scope.mode === 'specific' && scope.doc
      ? scope.doc.length > 22 ? scope.doc.slice(0, 20) + '…' : scope.doc
      : scope.mode === 'specific'
      ? 'Pick a document'
      : 'All documents';

  return (
    <div
      className="sticky bottom-0 px-4 pb-4 pt-2 md:px-8"
      style={{
        background:    'var(--bg-chat)',
        borderTop:     '1px solid var(--border)',
        paddingBottom: 'max(1rem, env(safe-area-inset-bottom))',
      }}
    >
      {/* ── Upload status banner ────────────────────────────────────────────── */}
      {upload.state !== 'idle' && (
        <div className="max-w-3xl mx-auto mb-2">
          <UploadBanner status={upload} onDismiss={() => setUpload({ state: 'idle' })} />
        </div>
      )}

      {/* ── Scope selector row ─────────────────────────────────────────────── */}
      <div className="max-w-3xl mx-auto mb-2 flex items-center gap-1.5 flex-wrap">
        {/* All documents pill */}
        <ScopePill
          active={scope.mode === 'all'}
          icon={<BookOpen size={11} />}
          label="All documents"
          onClick={() => onScopeChange({ mode: 'all', doc: null })}
        />

        {/* Specific document pill + picker */}
        <div className="relative" ref={pickerRef}>
          <button
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-all border"
            style={{
              background: scope.mode === 'specific'
                ? 'var(--accent)'
                : 'transparent',
              color: scope.mode === 'specific'
                ? 'white'
                : 'var(--text-muted)',
              borderColor: scope.mode === 'specific'
                ? 'var(--accent)'
                : 'var(--border)',
            }}
            onClick={() => {
              if (scope.mode !== 'specific') {
                onScopeChange({ mode: 'specific', doc: indexedDocs[0] ?? null });
              }
              const opening = docPickerOpen === false;
              setDocPickerOpen(v => !v);
              if (opening) void fetchDrivePerms();
            }}
          >
            <FileText size={11} />
            <span>{scopeLabel}</span>
            <ChevronDown
              size={10}
              style={{ transform: docPickerOpen ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.15s' }}
            />
          </button>

          {/* Document dropdown */}
          {docPickerOpen && (
            <div
              className="absolute bottom-full mb-1.5 left-0 z-30 rounded-xl shadow-lg border overflow-hidden"
              style={{
                background:  'var(--bg-sidebar)',
                borderColor: 'var(--border-dark)',
                minWidth:    '220px',
                maxHeight:   '220px',
                overflowY:   'auto',
              }}
            >
              {indexedDocs.length === 0 ? (
                <p className="px-4 py-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                  No documents indexed yet.
                  <br />Upload a file with 📎 to get started.
                </p>
              ) : (
                indexedDocs.map(doc => {
                  const perm = drivePerms[doc];
                  return (
                    <button
                      key={doc}
                      onClick={() => {
                        onScopeChange({ mode: 'specific', doc });
                        setDocPickerOpen(false);
                      }}
                      className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs transition-colors"
                      style={{
                        background: scope.doc === doc
                          ? 'rgba(99,102,241,0.15)'
                          : 'transparent',
                        color: 'var(--text-on-dark)',
                      }}
                      onMouseEnter={e => {
                        if (scope.doc !== doc)
                          (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.05)';
                      }}
                      onMouseLeave={e => {
                        if (scope.doc !== doc)
                          (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
                      }}
                    >
                      <FileText size={11} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                      <span className="truncate flex-1" title={doc}>{doc}</span>
                      {/* Drive permission icon (only for files in the drive_store) */}
                      {perm && (
                        perm.is_public
                          ? <span title="Public (anyone with link)" style={{ flexShrink: 0, display: 'flex' }}><Globe size={10} style={{ color: '#22c55e' }} /></span>
                          : perm.allowed_users.length > 1
                            ? <span title={`Shared with ${perm.allowed_users.length} users`} style={{ flexShrink: 0, display: 'flex' }}><Users size={10} style={{ color: '#fbbf24' }} /></span>
                            : <span title="Private (only you)" style={{ flexShrink: 0, display: 'flex' }}><Lock  size={10} style={{ color: 'var(--text-muted)' }} /></span>
                      )}
                      {scope.doc === doc && (
                        <span className="text-[10px] flex-shrink-0" style={{ color: 'var(--accent)' }}>✓</span>
                      )}
                    </button>
                  );
                })
              )}
            </div>
          )}
        </div>

        {/* Knowledge base pill */}
        <ScopePill
          active={false}
          icon={<Database size={11} />}
          label="Knowledge base"
          onClick={() => onScopeChange({ mode: 'all', doc: null })}
          tooltip="Same as all documents in this system"
        />

        {scope.mode === 'specific' && scope.doc && (
          <span className="text-xs ml-auto" style={{ color: 'var(--text-muted)' }}>
            Searching in: <strong style={{ color: 'var(--accent)' }}>{scope.doc}</strong>
          </span>
        )}
      </div>

      {/* ── Main input row ─────────────────────────────────────────────────── */}
      <div
        className="max-w-3xl mx-auto flex items-end gap-2 rounded-xl border px-3 py-2 shadow-sm transition-all duration-150"
        style={{
          borderColor: 'var(--border)',
          background:  'var(--bg-chat)',
        }}
        onFocusCapture={e => (e.currentTarget.style.borderColor = 'var(--accent)')}
        onBlurCapture={e  => (e.currentTarget.style.borderColor = 'var(--border)')}
      >
        {/* Attachment button */}
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept=".pdf,.docx,.pptx,.xlsx,.txt,.md,.eml,.png,.jpg,.jpeg,.gif,.csv"
          onChange={handleFileChange}
          aria-label="Attach file"
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={busy}
          className="flex items-center justify-center w-8 h-8 rounded-lg flex-shrink-0 transition-colors mb-0.5"
          style={{
            color:      busy ? 'var(--accent)' : 'var(--text-muted)',
            background: 'transparent',
          }}
          onMouseEnter={e => {
            if (!busy) (e.currentTarget as HTMLButtonElement).style.background = 'rgba(99,102,241,0.08)';
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
          }}
          title="Attach a document"
          aria-label="Attach file"
        >
          {busy
            ? <span className="w-3.5 h-3.5 rounded-full border-2 border-t-transparent animate-spin"
                    style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
            : <Paperclip size={16} />
          }
        </button>

        {/* Textarea */}
        <textarea
          id="main-input"
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={e => setValue(e.target.value.slice(0, MAX_CHARS))}
          onKeyDown={handleKeyDown}
          placeholder={
            scope.mode === 'specific' && scope.doc
              ? `Ask about ${scope.doc}…`
              : 'Ask anything about your documents…'
          }
          disabled={disabled}
          className="flex-1 resize-none bg-transparent outline-none text-sm leading-6 py-0.5 placeholder:text-gray-400 disabled:opacity-60"
          style={{
            color:      'var(--text-primary)',
            fontFamily: 'DM Sans, sans-serif',
            minHeight:  '28px',
            maxHeight:  '144px',
          }}
        />

        {/* Right-side controls */}
        <div className="flex items-center gap-1.5 pb-0.5">
          {showCounter && (
            <span
              className="text-xs font-mono"
              style={{ color: overLimit ? 'var(--danger)' : 'var(--text-muted)' }}
            >
              {value.length}/{MAX_CHARS}
            </span>
          )}

          {disabled && onStop ? (
            <button
              onClick={onStop}
              className="flex items-center justify-center w-8 h-8 rounded-lg transition-colors"
              style={{ background: 'var(--danger)', color: 'white' }}
              title="Stop generation"
            >
              <Square size={14} fill="white" />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={disabled || !value.trim() || overLimit}
              className="flex items-center justify-center w-8 h-8 rounded-lg transition-all duration-150"
              style={{
                background: disabled || !value.trim() || overLimit
                  ? 'var(--border)'
                  : 'var(--accent)',
                color: disabled || !value.trim() || overLimit
                  ? 'var(--text-muted)'
                  : 'white',
                transform: disabled ? 'scale(0.92)' : 'scale(1)',
                opacity:   disabled ? 0.6 : 1,
              }}
              title="Send (Enter)"
            >
              {disabled ? (
                <span
                  className="w-3.5 h-3.5 rounded-full border-2 border-t-transparent animate-spin"
                  style={{ borderColor: 'var(--text-muted)', borderTopColor: 'transparent' }}
                />
              ) : (
                <Send size={14} />
              )}
            </button>
          )}
        </div>
      </div>

      <p className="text-center text-xs mt-1.5" style={{ color: 'var(--text-muted)' }}>
        Enter to send · Shift+Enter for newline · 📎 attach a document
      </p>
    </div>
  );
}

// ── Scope pill button ─────────────────────────────────────────────────────────
function ScopePill({
  active, icon, label, onClick, tooltip,
}: {
  active:    boolean;
  icon:      React.ReactNode;
  label:     string;
  onClick:   () => void;
  tooltip?:  string;
}) {
  return (
    <button
      className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-all border"
      style={{
        background:  active ? 'var(--accent)' : 'transparent',
        color:       active ? 'white' : 'var(--text-muted)',
        borderColor: active ? 'var(--accent)' : 'var(--border)',
      }}
      onClick={onClick}
      title={tooltip}
      onMouseEnter={e => {
        if (!active) {
          (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--accent)';
          (e.currentTarget as HTMLButtonElement).style.color = 'var(--accent)';
        }
      }}
      onMouseLeave={e => {
        if (!active) {
          (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--border)';
          (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-muted)';
        }
      }}
    >
      {icon}
      {label}
    </button>
  );
}

// ── Upload status banner ──────────────────────────────────────────────────────
function UploadBanner({
  status,
  onDismiss,
}: {
  status:    UploadStatus;
  onDismiss: () => void;
}) {
  if (status.state === 'idle') return null;

  const { icon, text, bg, fg } = (() => {
    switch (status.state) {
      case 'uploading':
        return {
          icon: <Loader size={13} className="animate-spin" />,
          text: `Uploading ${status.filename}…`,
          bg:   'rgba(99,102,241,0.08)',
          fg:   'var(--accent)',
        };
      case 'processing':
        return {
          icon: <Loader size={13} className="animate-spin" />,
          text: `Extracting ${status.filename}… (may take a minute for scanned PDFs)`,
          bg:   'rgba(99,102,241,0.08)',
          fg:   'var(--accent)',
        };
      case 'indexing':
        return {
          icon: <Loader size={13} className="animate-spin" />,
          text: `Indexing ${status.filename}…`,
          bg:   'rgba(99,102,241,0.08)',
          fg:   'var(--accent)',
        };
      case 'done':
        return {
          icon: <CheckCircle size={13} />,
          text: `${status.filename} indexed — ready to query!`,
          bg:   'rgba(34,197,94,0.08)',
          fg:   'var(--success)',
        };
      case 'error':
        return {
          icon: <AlertCircle size={13} />,
          text: status.message,
          bg:   'rgba(239,68,68,0.08)',
          fg:   'var(--danger)',
        };
    }
  })();

  return (
    <div
      className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs"
      style={{ background: bg, color: fg }}
    >
      {icon}
      <span className="flex-1 truncate">{text}</span>
      {(status.state === 'done' || status.state === 'error') && (
        <button
          onClick={onDismiss}
          className="flex-shrink-0 opacity-60 hover:opacity-100 transition-opacity"
          aria-label="Dismiss"
        >
          <X size={12} />
        </button>
      )}
    </div>
  );
}
