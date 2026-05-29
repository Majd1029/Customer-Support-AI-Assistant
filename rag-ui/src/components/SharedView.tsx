import { useState, useEffect } from 'react';
import { BookOpen, Lock, AlertCircle, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Source } from '../types';

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';

interface SharedMessage {
  role:     'user' | 'assistant';
  content:  string;
  metadata: Record<string, unknown> | null;
}

interface SharedConversation {
  label:    string;
  session_id: string;
  messages: SharedMessage[];
}

interface SharedViewProps {
  shareToken: string;
}

export default function SharedView({ shareToken }: SharedViewProps) {
  const [conv, setConv]       = useState<SharedConversation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_URL}/shared/${encodeURIComponent(shareToken)}`);
        if (res.status === 404) { setError('not_found'); return; }
        if (!res.ok) { setError('server'); return; }
        const data = await res.json() as SharedConversation;
        setConv(data);
      } catch {
        setError('network');
      } finally {
        setLoading(false);
      }
    })();
  }, [shareToken]);

  return (
    <div className="flex flex-col h-full" style={{ background: 'var(--bg-base)' }}>
      {/* ── Top banner ────────────────────────────────────────────────────────── */}
      <header
        className="flex items-center justify-between px-5 py-3 border-b flex-shrink-0"
        style={{
          background:   'var(--bg-sidebar)',
          borderColor:  'var(--border-dark)',
        }}
      >
        <div className="flex items-center gap-2.5">
          <div
            className="flex items-center justify-center w-7 h-7 rounded-lg flex-shrink-0"
            style={{ background: 'var(--accent)' }}
          >
            <BookOpen size={13} color="white" />
          </div>
          <span className="font-semibold text-sm" style={{ color: 'var(--text-on-dark)' }}>
            CustomerAssist
          </span>
          {conv && (
            <>
              <span className="text-sm" style={{ color: 'var(--text-muted)' }}>/</span>
              <span
                className="text-sm font-medium truncate max-w-[300px]"
                style={{ color: 'var(--text-on-dark)' }}
              >
                {conv.label}
              </span>
            </>
          )}
        </div>

        {/* Read-only badge */}
        <div
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium"
          style={{ background: 'rgba(255,255,255,0.07)', color: 'var(--text-muted)' }}
        >
          <Lock size={11} />
          Read-only
        </div>
      </header>

      {/* ── Body ──────────────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="flex flex-col items-center justify-center h-full gap-3">
            <Loader2 size={22} className="animate-spin" style={{ color: 'var(--accent)' }} />
            <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Loading conversation…
            </span>
          </div>
        )}

        {!loading && error && <ErrorState type={error} />}

        {!loading && conv && (
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
            {conv.messages.map((msg, i) => (
              <SharedMessage key={i} msg={msg} />
            ))}
          </div>
        )}
      </div>

      {/* ── Footer note ───────────────────────────────────────────────────────── */}
      {!loading && conv && (
        <div
          className="flex items-center justify-center py-3 text-xs border-t flex-shrink-0"
          style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}
        >
          <Lock size={10} className="mr-1.5" />
          This is a read-only shared view.&nbsp;
          <a
            href="/"
            className="underline hover:opacity-80 transition-opacity"
            style={{ color: 'var(--accent)' }}
          >
            Sign in to start your own conversation.
          </a>
        </div>
      )}
    </div>
  );
}

// ── Single message bubble ─────────────────────────────────────────────────────
function SharedMessage({ msg }: { msg: SharedMessage }) {
  const isUser = msg.role === 'user';
  const sources = (msg.metadata?.sources ?? []) as Source[];

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] ${isUser ? 'rounded-2xl rounded-br-sm px-4 py-2.5' : 'rounded-2xl rounded-bl-sm px-4 py-3'}`}
        style={{
          background: isUser ? 'var(--accent)' : 'var(--bg-ai-bubble)',
          color:      isUser ? 'white'          : 'var(--text-primary)',
        }}
      >
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
        ) : (
          <div className="prose prose-sm max-w-none text-sm" style={{ color: 'var(--text-primary)' }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {msg.content}
            </ReactMarkdown>
          </div>
        )}

        {/* Source chips */}
        {!isUser && sources.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
            {sources.map((src, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs"
                style={{ background: 'var(--bg-base)', color: 'var(--text-muted)' }}
              >
                {src.filename}
                {src.page != null && `, p.${src.page}`}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Error state ───────────────────────────────────────────────────────────────
function ErrorState({ type }: { type: string }) {
  const messages: Record<string, { title: string; body: string }> = {
    not_found: {
      title: 'Conversation not found',
      body:  'This share link may be invalid or the conversation may have been deleted.',
    },
    network: {
      title: 'Connection error',
      body:  'Could not reach the server. Please check your connection and try again.',
    },
    server: {
      title: 'Server error',
      body:  'Something went wrong on our end. Please try again later.',
    },
  };
  const { title, body } = messages[type] ?? messages['server'];

  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 px-6 text-center">
      <div
        className="flex items-center justify-center w-12 h-12 rounded-full"
        style={{ background: 'rgba(239,68,68,0.12)' }}
      >
        <AlertCircle size={22} style={{ color: 'var(--danger)' }} />
      </div>
      <div>
        <p className="font-semibold text-base mb-1" style={{ color: 'var(--text-primary)' }}>
          {title}
        </p>
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          {body}
        </p>
      </div>
      <a
        href="/"
        className="text-sm underline hover:opacity-80 transition-opacity"
        style={{ color: 'var(--accent)' }}
      >
        Go to CustomerAssist
      </a>
    </div>
  );
}
