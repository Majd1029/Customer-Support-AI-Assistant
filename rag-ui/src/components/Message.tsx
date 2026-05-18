import React, { useState, useMemo } from 'react';
import type { ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { FileText, X, ThumbsUp, ThumbsDown, AlertTriangle } from 'lucide-react';
import type { Message, Source } from '../types';
import EvalPanel from './EvalPanel';
import SourceCard from './SourceCard';

interface MessageProps {
  message:    Message;
  username?:  string;
  onFeedback?: (rating: 1 | -1) => void;
}

// ── Popover showing the chunk excerpt ─────────────────────────────────────────
function SourcePopover({
  source,
  onClose,
}: {
  source: Source;
  onClose: () => void;
}) {
  return (
    <div
      className="absolute z-40 rounded-xl shadow-xl border text-xs"
      style={{
        background:  'var(--bg-sidebar)',
        borderColor: 'var(--border-dark)',
        width:       '280px',
        top:         '100%',
        left:        0,
        marginTop:   '4px',
      }}
      // prevent click from bubbling and immediately re-toggling
      onClick={e => e.stopPropagation()}
    >
      {/* Header */}
      <div
        className="flex items-start justify-between gap-2 px-3 pt-3 pb-2 border-b"
        style={{ borderColor: 'var(--border-dark)' }}
      >
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <FileText size={11} style={{ color: 'var(--accent)', flexShrink: 0 }} />
            <span
              className="font-medium truncate"
              style={{ color: 'var(--text-on-dark)' }}
              title={source.filename}
            >
              {source.filename}
            </span>
          </div>
          <div className="flex items-center gap-1.5 mt-0.5">
            {source.page != null && (
              <span
                className="px-1 rounded font-mono text-[10px]"
                style={{ background: 'rgba(99,102,241,0.15)', color: 'var(--accent)' }}
              >
                p.{source.page}
              </span>
            )}
            {source.section && (
              <span
                className="text-[10px] truncate max-w-[150px]"
                style={{ color: 'var(--text-muted)' }}
                title={source.section}
              >
                § {source.section}
              </span>
            )}
            {source.score != null && (
              <span className="text-[10px] font-mono ml-auto" style={{ color: 'var(--text-muted)' }}>
                {source.score.toFixed(2)}
              </span>
            )}
          </div>
        </div>
        <button
          onClick={onClose}
          className="flex-shrink-0 p-0.5 rounded transition-colors"
          style={{ color: 'var(--text-muted)' }}
          onMouseEnter={e => (e.currentTarget.style.color = 'var(--text-on-dark)')}
          onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-muted)')}
          aria-label="Close"
        >
          <X size={12} />
        </button>
      </div>

      {/* Content excerpt */}
      {source.content?.trim() ? (
        <p
          className="px-3 py-2.5 text-[11px] leading-relaxed whitespace-pre-wrap break-words"
          style={{
            color:        'var(--text-muted)',
            fontFamily:   'DM Mono, monospace',
            maxHeight:    '140px',
            overflowY:    'auto',
          }}
        >
          {source.content}
          {source.content.length >= 400 && (
            <span style={{ opacity: 0.5 }}> …</span>
          )}
        </p>
      ) : (
        <p className="px-3 py-2.5 text-[11px]" style={{ color: 'var(--text-muted)' }}>
          No excerpt available.
        </p>
      )}
    </div>
  );
}

// ── Citation chip (inline in text) ────────────────────────────────────────────
function CitationChip({
  filename,
  page,
  sources,
}: {
  filename: string;
  page:     string;
  sources:  Source[];
}) {
  const [open, setOpen] = useState(false);

  // Match the source by filename (exact or stem-based) and page
  const matched = useMemo(() => {
    const pg = parseInt(page, 10);
    return (
      sources.find(
        s =>
          (s.filename === filename || s.filename.replace(/\.[^.]+$/, '') === filename) &&
          (s.page === pg || s.page == null)
      ) ??
      sources.find(
        s => s.filename === filename || s.filename.replace(/\.[^.]+$/, '') === filename
      ) ??
      null
    );
  }, [sources, filename, page]);

  return (
    <span className="relative inline-block align-middle">
      <button
        className="inline-flex items-center gap-1 mx-0.5 px-1.5 py-0.5 rounded text-xs font-medium transition-colors"
        style={{
          background: open ? 'var(--accent)' : 'rgba(99,102,241,0.1)',
          color:      open ? 'white' : 'var(--accent)',
          border:     `1px solid ${open ? 'var(--accent)' : 'rgba(99,102,241,0.25)'}`,
        }}
        onClick={() => setOpen(v => !v)}
        title={matched ? 'Click to view excerpt' : `${filename}, p.${page}`}
      >
        <FileText size={10} />
        {filename}, p.{page}
      </button>

      {open && matched && (
        <>
          {/* Backdrop to close */}
          <span
            className="fixed inset-0 z-30"
            onClick={() => setOpen(false)}
          />
          <SourcePopover source={matched} onClose={() => setOpen(false)} />
        </>
      )}
    </span>
  );
}

// ── Replace [Source: filename, page N] markers with chips ─────────────────────
function processTextWithCitations(text: string, sources: Source[]): ReactNode[] {
  const re = /\[Source:\s*([^,\]]+?),\s*page\s*(\d+)\]/g;
  const nodes: ReactNode[] = [];
  let lastIdx = 0;
  let match: RegExpExecArray | null;

  while ((match = re.exec(text)) !== null) {
    if (match.index > lastIdx) {
      nodes.push(text.slice(lastIdx, match.index));
    }
    nodes.push(
      <CitationChip
        key={`${match.index}-${match[1]}`}
        filename={match[1].trim()}
        page={match[2]}
        sources={sources}
      />
    );
    lastIdx = match.index + match[0].length;
  }

  if (lastIdx < text.length) nodes.push(text.slice(lastIdx));
  return nodes;
}

// react-markdown v9 passes an extra `node` prop (the hast AST node) to every
// custom component.  Destructure it out so it isn't spread onto the DOM <p>.
type ParagraphProps = React.ComponentPropsWithoutRef<'p'> & { node?: unknown };

function CitationParagraph({
  children,
  node: _node,
  sources,
  ...rest
}: ParagraphProps & { sources: Source[] }) {
  const processNode = (n: ReactNode): ReactNode => {
    if (typeof n === 'string') {
      const parts = processTextWithCitations(n, sources);
      return parts.length === 1 && typeof parts[0] === 'string'
        ? parts[0]
        : <>{parts}</>;
    }
    return n;
  };

  return (
    <p {...rest}>
      {Array.isArray(children)
        ? children.map((c, i) => <span key={i}>{processNode(c as ReactNode)}</span>)
        : processNode(children)}
    </p>
  );
}

// ── Metadata badge strip ───────────────────────────────────────────────────────
function MetadataStrip({ message }: { message: Message }) {
  const meta = message.metadata;
  if (!meta) return null;

  const items: { label: string; value: string; warn?: boolean }[] = [];

  if (meta.confidence != null) {
    items.push({
      label: 'confidence',
      value: `${Math.round(meta.confidence * 100)}%`,
      warn:  meta.confidence < 0.5,
    });
  }
  if (meta.hops > 1) items.push({ label: 'hops', value: String(meta.hops) });
  if (meta.citationCount != null) {
    items.push({ label: 'citations', value: String(meta.citationCount) });
  }
  if (meta.retrievalMs != null) {
    items.push({ label: 'retrieval', value: `${Math.round(meta.retrievalMs)}ms` });
  }
  if (meta.generationMs != null) {
    items.push({ label: 'generation', value: `${Math.round(meta.generationMs)}ms` });
  }
  if (meta.rewriteTier && meta.rewriteTier !== 'skip' && meta.rewrittenQuery) {
    items.push({ label: 'rewrite', value: meta.rewriteTier });
  }
  if (meta.noAnswer) {
    items.push({ label: 'status', value: 'no answer', warn: true });
  }

  if (items.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-1.5 mt-1.5">
      {items.map(it => (
        <span
          key={it.label}
          className="inline-flex items-center gap-1 text-xs px-1.5 py-0.5 rounded font-mono"
          style={{
            background: it.warn ? 'var(--badge-warn-bg)' : 'var(--badge-bg)',
            color:      it.warn ? 'var(--danger)' : 'var(--text-muted)',
          }}
        >
          <span style={{ opacity: 0.6 }}>{it.label}</span>
          <span style={{ color: it.warn ? 'var(--danger)' : 'var(--text-primary)' }}>
            {it.value}
          </span>
        </span>
      ))}
    </div>
  );
}

// ── Avatar ────────────────────────────────────────────────────────────────────
function Avatar({ role, username }: { role: 'user' | 'assistant'; username?: string }) {
  if (role === 'user') {
    const initial = username ? username.charAt(0).toUpperCase() : 'U';
    return (
      <div
        className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-semibold"
        style={{ background: 'var(--accent)', color: 'white' }}
        title={username}
      >
        {initial}
      </div>
    );
  }
  return (
    <div
      className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0"
      style={{ background: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.2)' }}
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2.5">
        <circle cx="12" cy="12" r="3"/>
        <path d="M12 2v3m0 14v3M2 12h3m14 0h3m-4.22-7.78-2.12 2.12M6.34 17.66l-2.12 2.12m14.14 0-2.12-2.12M6.34 6.34 4.22 4.22"/>
      </svg>
    </div>
  );
}

// ── Escalation notice ─────────────────────────────────────────────────────────
function EscalationBanner({ message: msg }: { message: string }) {
  return (
    <div
      className="flex items-start gap-2 mt-2 px-3 py-2.5 rounded-xl text-xs leading-relaxed"
      style={{
        background:  'rgba(251,191,36,0.10)',
        border:      '1px solid rgba(251,191,36,0.30)',
        color:       '#fbbf24',
      }}
    >
      <AlertTriangle size={13} className="flex-shrink-0 mt-0.5" />
      <span>{msg}</span>
    </div>
  );
}

// ── Feedback thumbs ───────────────────────────────────────────────────────────
function FeedbackRow({
  rating,
  onFeedback,
}: {
  rating:     0 | 1 | -1;
  onFeedback: (r: 1 | -1) => void;
}) {
  return (
    <div className="flex items-center gap-1 mt-1.5">
      <button
        onClick={() => onFeedback(1)}
        title="Helpful"
        aria-label="Mark as helpful"
        className="p-1 rounded transition-colors"
        style={{
          color:      rating === 1  ? '#22c55e' : 'var(--text-muted)',
          background: rating === 1  ? 'rgba(34,197,94,0.12)' : 'transparent',
        }}
        onMouseEnter={e => { if (rating !== 1)  (e.currentTarget as HTMLButtonElement).style.color = '#22c55e'; }}
        onMouseLeave={e => { if (rating !== 1)  (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-muted)'; }}
      >
        <ThumbsUp size={13} />
      </button>
      <button
        onClick={() => onFeedback(-1)}
        title="Not helpful"
        aria-label="Mark as not helpful"
        className="p-1 rounded transition-colors"
        style={{
          color:      rating === -1 ? 'var(--danger)' : 'var(--text-muted)',
          background: rating === -1 ? 'rgba(239,68,68,0.10)' : 'transparent',
        }}
        onMouseEnter={e => { if (rating !== -1) (e.currentTarget as HTMLButtonElement).style.color = 'var(--danger)'; }}
        onMouseLeave={e => { if (rating !== -1) (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-muted)'; }}
      >
        <ThumbsDown size={13} />
      </button>
    </div>
  );
}

// ── Main Message component ────────────────────────────────────────────────────
export default function MessageComponent({ message, username, onFeedback }: MessageProps) {
  const isUser = message.role === 'user';

  // Build the markdown components map once per render, capturing message.sources
  const markdownComponents = useMemo(() => ({
    p: (props: ParagraphProps) => (
      <CitationParagraph {...props} sources={message.sources} />
    ),
  }), [message.sources]);

  return (
    <div
      className={`flex gap-2.5 animate-slide-up ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
    >
      <Avatar role={message.role} username={username} />

      <div className={`flex flex-col max-w-[80%] ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Bubble */}
        <div
          className="rounded-2xl px-4 py-3 text-sm leading-relaxed"
          style={
            isUser
              ? {
                  background: 'var(--bg-user-bubble)',
                  color:      'var(--text-on-dark)',
                  borderBottomRightRadius: '4px',
                  boxShadow: '0 1px 4px rgba(99,102,241,0.18)',
                }
              : {
                  background:  'var(--bg-ai-bubble)',
                  color:       'var(--text-primary)',
                  borderBottomLeftRadius: '4px',
                  minWidth:    '120px',
                  border:      '1px solid var(--border)',
                  boxShadow:   '0 1px 3px rgba(0,0,0,0.04)',
                }
          }
        >
          {isUser ? (
            <p className="whitespace-pre-wrap break-words">{message.content}</p>
          ) : (
            <>
              {message.error ? (
                <p className="text-red-500 text-sm">{message.error}</p>
              ) : (
                <div className="prose">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={markdownComponents}
                  >
                    {message.content + (message.streaming ? ' ▋' : '')}
                  </ReactMarkdown>
                </div>
              )}
            </>
          )}
        </div>

        {/* Below-bubble chrome — only for assistant */}
        {!isUser && (
          <>
            {/* Escalation notice — shown when the backend routed to a human agent */}
            {message.metadata?.escalationMessage && (
              <EscalationBanner message={message.metadata.escalationMessage} />
            )}
            <MetadataStrip message={message} />
            <SourceCard sources={message.sources} />
            {message.evalState && !message.evalState.loading && (
              <EvalPanel evalState={message.evalState} />
            )}
            {message.evalState?.loading && message.evalState.alignmentScore == null && (
              <EvalPanel evalState={message.evalState} />
            )}
            {/* Feedback thumbs — show once streaming is done */}
            {!message.streaming && !message.error && onFeedback && (
              <FeedbackRow
                rating={message.metadata?.rating ?? 0}
                onFeedback={onFeedback}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}
