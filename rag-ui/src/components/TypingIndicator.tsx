export default function TypingIndicator() {
  return (
    <div
      className="inline-flex items-center gap-1 px-4 py-3 rounded-2xl"
      style={{ background: 'var(--bg-ai-bubble)' }}
    >
      {[0, 1, 2].map(i => (
        <span
          key={i}
          className="w-2 h-2 rounded-full"
          style={{
            background: 'var(--text-muted)',
            animation: `bounce-dot 1.4s ease-in-out ${i * 0.2}s infinite`,
          }}
        />
      ))}
    </div>
  );
}
