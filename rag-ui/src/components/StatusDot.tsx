import type { HealthStatus } from '../types';

interface StatusDotProps {
  status: HealthStatus;
}

const STATUS_CONFIG: Record<HealthStatus, { color: string; label: string }> = {
  ok:          { color: 'var(--success)', label: 'Connected' },
  degraded:    { color: 'var(--warning)', label: 'Degraded' },
  unreachable: { color: 'var(--danger)',  label: 'Offline' },
};

export default function StatusDot({ status }: StatusDotProps) {
  const { color, label } = STATUS_CONFIG[status];
  return (
    <div className="flex items-center gap-1.5" title={label}>
      <span
        className="w-2 h-2 rounded-full flex-shrink-0"
        style={{ background: color }}
      />
      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{label}</span>
    </div>
  );
}
