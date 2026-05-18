import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import type { EvalState } from '../types';

interface EvalPanelProps {
  evalState: EvalState;
}

type Verdict = 'pass' | 'fail' | 'low_confidence' | 'off_topic';

const VERDICT_CONFIG: Record<string, { dot: string; bg: string; text: string; label: string }> = {
  pass:           { dot: 'var(--success)',  bg: 'var(--verdict-pass-bg)', text: 'var(--verdict-pass-fg)', label: 'Pass' },
  fail:           { dot: 'var(--danger)',   bg: 'var(--verdict-fail-bg)', text: 'var(--verdict-fail-fg)', label: 'Fail' },
  low_confidence: { dot: 'var(--warning)',  bg: 'var(--verdict-warn-bg)', text: 'var(--verdict-warn-fg)', label: 'Low confidence' },
  off_topic:      { dot: '#f97316',         bg: 'var(--verdict-info-bg)', text: 'var(--verdict-info-fg)', label: 'Off topic' },
};

function getVerdictConfig(verdict: string | null) {
  if (!verdict) return null;
  return VERDICT_CONFIG[verdict as Verdict] ?? VERDICT_CONFIG.fail;
}

// ── Score bar ─────────────────────────────────────────────────────────────────
function ScoreBar({
  label,
  score,
  delayMs = 0,
}: {
  label:    string;
  score:    number | null;
  delayMs?: number;
}) {
  const pct = score != null ? Math.round(score * 100) : 0;
  const barColor = score == null
    ? 'var(--border)'
    : score >= 0.7
    ? 'var(--accent)'
    : score >= 0.5
    ? 'var(--warning)'
    : 'var(--danger)';

  return (
    <div className="flex items-center gap-3 py-0.5">
      <span
        className="text-xs flex-shrink-0 w-40"
        style={{ color: 'var(--text-muted)' }}
      >
        {label}
      </span>
      <div
        className="flex-1 h-1.5 rounded-full overflow-hidden"
        style={{ background: 'var(--border)' }}
      >
        {score != null && (
          <div
            className="h-full rounded-full"
            style={{
              width: `${pct}%`,
              background: barColor,
              animation: `bar-grow 300ms ease-out ${delayMs}ms both`,
            }}
          />
        )}
      </div>
      <span
        className="text-xs font-mono w-9 text-right flex-shrink-0"
        style={{ color: 'var(--text-primary)' }}
      >
        {score != null ? score.toFixed(2) : '—'}
      </span>
    </div>
  );
}

export default function EvalPanel({ evalState }: EvalPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const vc = getVerdictConfig(evalState.verdict);

  // If still loading and no scores yet, show a minimal spinner row
  if (evalState.loading && evalState.alignmentScore == null) {
    return (
      <div className="flex items-center gap-2 mt-2">
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Evaluating…
        </span>
        <span
          className="w-3 h-3 rounded-full border-2 border-t-transparent animate-spin"
          style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }}
        />
      </div>
    );
  }

  const hasJudge = !!(evalState.judge?.dimensions && Object.keys(evalState.judge.dimensions).length > 0);

  return (
    <div className="mt-2 text-sm">
      {/* ── Compact row ──────────────────────────────────────────────────── */}
      <button
        onClick={() => setExpanded(v => !v)}
        className="flex items-center gap-2 text-xs rounded-lg px-2 py-1 transition-colors w-full text-left"
        style={{ color: 'var(--text-muted)' }}
        onMouseEnter={e => (e.currentTarget.style.background = 'var(--eval-hover)')}
        onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
      >
        {/* Verdict dot + badge */}
        {vc && (
          <span
            className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
            style={{ background: vc.bg, color: vc.text }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: vc.dot }} />
            {vc.label}
          </span>
        )}

        {/* Quick scores */}
        {evalState.alignmentScore != null && (
          <span>Alignment {evalState.alignmentScore.toFixed(2)}</span>
        )}
        {evalState.confidenceScore != null && (
          <span>Grounding {evalState.confidenceScore.toFixed(2)}</span>
        )}

        <ChevronDown
          size={12}
          className="ml-auto transition-transform duration-200"
          style={{ transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)' }}
        />
      </button>

      {/* ── Expanded panel ───────────────────────────────────────────────── */}
      {expanded && (
        <div
          className="mt-1.5 rounded-xl border p-4 space-y-1"
          style={{ borderColor: 'var(--border)', background: 'var(--eval-bg)' }}
        >
          <p className="text-xs font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
            Evaluation Results
          </p>

          <ScoreBar
            label="Semantic Alignment"
            score={evalState.alignmentScore}
            delayMs={0}
          />
          <ScoreBar
            label="Grounding Score"
            score={evalState.confidenceScore}
            delayMs={60}
          />

          {/* LLM Judge section */}
          {hasJudge && evalState.judge && (
            <>
              <div
                className="my-2 border-t text-xs font-medium pt-2"
                style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}
              >
                LLM Judge
              </div>

              {Object.entries(evalState.judge.dimensions).map(([key, dim], idx) => (
                <ScoreBar
                  key={key}
                  label={key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                  score={dim.score}
                  delayMs={idx * 40}
                />
              ))}

              <div className="my-2 border-t" style={{ borderColor: 'var(--border)' }} />

              <ScoreBar
                label="Overall"
                score={evalState.judge.overall}
                delayMs={Object.keys(evalState.judge.dimensions).length * 40}
              />
            </>
          )}

          {/* Feedback text */}
          {evalState.feedback && (
            <p
              className="text-xs mt-2 italic"
              style={{ color: 'var(--text-muted)' }}
            >
              &ldquo;{evalState.feedback}&rdquo;
            </p>
          )}
        </div>
      )}
    </div>
  );
}
