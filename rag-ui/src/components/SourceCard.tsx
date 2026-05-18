import { useState } from 'react';
import { FileText, ChevronDown, ExternalLink } from 'lucide-react';
import type { Source } from '../types';

interface SourceCardProps {
  sources: Source[];
}

export default function SourceCard({ sources }: SourceCardProps) {
  const [open, setOpen] = useState(false);

  if (sources.length === 0) return null;

  return (
    <div className="mt-2">
      {/* Toggle row */}
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-1.5 text-xs transition-colors rounded px-1 py-0.5"
        style={{ color: 'var(--text-muted)' }}
        onMouseEnter={e => (e.currentTarget.style.color = 'var(--accent)')}
        onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-muted)')}
      >
        <FileText size={12} />
        <span>{sources.length} source{sources.length > 1 ? 's' : ''}</span>
        <ChevronDown
          size={11}
          className="transition-transform duration-200"
          style={{ transform: open ? 'rotate(180deg)' : 'rotate(0deg)' }}
        />
      </button>

      {/* Cards */}
      {open && (
        <div className="mt-1.5 grid gap-2 sm:grid-cols-2">
          {sources.map((src, idx) => (
            <SourceChip key={src.chunkId ?? idx} source={src} index={idx + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

function SourceChip({ source, index }: { source: Source; index: number }) {
  const [expanded, setExpanded] = useState(false);

  const scoreColor =
    source.score == null
      ? 'var(--text-muted)'
      : source.score >= 0.7
      ? 'var(--success)'
      : source.score >= 0.4
      ? 'var(--warning)'
      : 'var(--danger)';

  const hasContent = !!source.content?.trim();

  return (
    <div
      className="rounded-lg border text-xs overflow-hidden"
      style={{ borderColor: 'var(--border)', background: '#fafafa' }}
    >
      {/* ── Header row (always visible) ──────────────────────────────────── */}
      <button
        className="w-full flex items-start gap-2 p-2.5 text-left transition-colors"
        style={{ background: 'transparent' }}
        onClick={() => hasContent && setExpanded(v => !v)}
        title={hasContent ? (expanded ? 'Hide excerpt' : 'Show text excerpt') : undefined}
        aria-expanded={hasContent ? expanded : undefined}
      >
        {/* Citation index badge */}
        <div
          className="flex items-center justify-center w-4 h-4 rounded text-[10px] font-bold flex-shrink-0 mt-0.5"
          style={{ background: 'rgba(99,102,241,0.12)', color: 'var(--accent)' }}
        >
          {index}
        </div>

        <div className="min-w-0 flex-1 space-y-0.5">
          <div className="flex items-center gap-1.5">
            <FileText size={11} className="flex-shrink-0" style={{ color: 'var(--accent)' }} />
            <p
              className="font-medium truncate"
              style={{ color: 'var(--text-primary)' }}
              title={source.filename}
            >
              {source.filename}
            </p>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            {source.page != null && (
              <span
                className="px-1 rounded font-mono"
                style={{ background: 'rgba(99,102,241,0.08)', color: 'var(--accent)' }}
              >
                p.{source.page}
              </span>
            )}
            {source.section && (
              <span
                className="truncate max-w-[120px]"
                style={{ color: 'var(--text-muted)' }}
                title={source.section}
              >
                § {source.section}
              </span>
            )}
          </div>
        </div>

        {/* Score + expand indicator */}
        <div className="flex flex-col items-end gap-1 flex-shrink-0">
          {source.score != null && (
            <>
              <span className="font-mono text-xs" style={{ color: scoreColor }}>
                {source.score.toFixed(2)}
              </span>
              <div
                className="w-12 h-1 rounded-full overflow-hidden"
                style={{ background: 'var(--border)' }}
              >
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${Math.min(100, Math.round(source.score * 100))}%`,
                    background: scoreColor,
                  }}
                />
              </div>
            </>
          )}
          {hasContent && (
            <ChevronDown
              size={10}
              style={{
                color: 'var(--text-muted)',
                transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.15s',
                marginTop: 2,
              }}
            />
          )}
        </div>
      </button>

      {/* ── Expandable content preview ────────────────────────────────────── */}
      {expanded && hasContent && (
        <div
          className="px-3 pb-3 pt-1 border-t"
          style={{ borderColor: 'var(--border)' }}
        >
          <div
            className="flex items-center gap-1 mb-1.5"
            style={{ color: 'var(--text-muted)' }}
          >
            <ExternalLink size={10} />
            <span className="text-[10px] font-medium uppercase tracking-wide">Excerpt</span>
          </div>
          <p
            className="text-xs leading-relaxed whitespace-pre-wrap break-words"
            style={{
              color:           'var(--text-primary)',
              fontFamily:      'DM Mono, monospace',
              fontSize:        '11px',
              opacity:         0.85,
              maxHeight:       '120px',
              overflowY:       'auto',
              background:      'rgba(99,102,241,0.04)',
              borderRadius:    '6px',
              padding:         '8px',
              borderLeft:      '2px solid var(--accent)',
            }}
          >
            {source.content}
            {source.content && source.content.length >= 400 && (
              <span style={{ color: 'var(--text-muted)' }}> …</span>
            )}
          </p>
        </div>
      )}
    </div>
  );
}
