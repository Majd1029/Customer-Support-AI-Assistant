import { useEffect, useRef } from 'react';
import type { Message } from '../types';
import MessageComponent from './Message';
import TypingIndicator from './TypingIndicator';

interface ChatAreaProps {
  messages:       Message[];
  isStreaming:    boolean;
  onExampleClick: (q: string) => void;
  username?:      string;
  onFeedback?:    (messageId: string, rating: 1 | -1) => void;
}

export default function ChatArea({ messages, isStreaming, onExampleClick, username, onFeedback }: ChatAreaProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto">
        <EmptyState onExampleClick={onExampleClick} />
        <div ref={bottomRef} />
      </div>
    );
  }

  // Show typing indicator only when streaming and the last assistant message is still empty
  const lastMsg = messages[messages.length - 1];
  const showTyping =
    isStreaming &&
    lastMsg.role === 'assistant' &&
    lastMsg.streaming &&
    lastMsg.content === '' &&
    !lastMsg.error;

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8">
      <div className="max-w-3xl mx-auto space-y-6">
        {messages.map(msg => (
          <MessageComponent
            key={msg.id}
            message={msg}
            username={username}
            onFeedback={
              msg.role === 'assistant' && onFeedback
                ? (rating: 1 | -1) => onFeedback(msg.id, rating)
                : undefined
            }
          />
        ))}
        {showTyping && (
          <div className="flex gap-2.5">
            {/* Spacer to align with assistant avatar */}
            <div className="w-7 flex-shrink-0" />
            <TypingIndicator />
          </div>
        )}
      </div>
      <div ref={bottomRef} />
    </div>
  );
}

// ── Empty state ───────────────────────────────────────────────────────────────
function EmptyState({ onExampleClick }: { onExampleClick: (q: string) => void }) {
  return (
    <div
      className="flex flex-col items-center justify-center min-h-full px-6 py-16 text-center"
      style={{ minHeight: 'calc(100vh - 180px)' }}
    >
      {/* Inline SVG illustration */}
      <svg
        width="88"
        height="88"
        viewBox="0 0 88 88"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >
        <rect x="10" y="6" width="50" height="64" rx="7" fill="rgba(99,102,241,0.08)" stroke="#6366f1" strokeWidth="2" />
        <rect x="20" y="20" width="30" height="3.5" rx="1.75" fill="#6366f1" opacity="0.5" />
        <rect x="20" y="30" width="22" height="3" rx="1.5" fill="#6366f1" opacity="0.35" />
        <rect x="20" y="39" width="26" height="3" rx="1.5" fill="#6366f1" opacity="0.3" />
        <rect x="20" y="48" width="18" height="3" rx="1.5" fill="#6366f1" opacity="0.2" />
        {/* Magnifier */}
        <circle cx="64" cy="60" r="18" fill="#6366f1" />
        <circle cx="64" cy="60" r="8" stroke="white" strokeWidth="2.5" fill="none" />
        <line x1="70" y1="66" x2="78" y2="74" stroke="white" strokeWidth="3" strokeLinecap="round" />
      </svg>

      <h2
        className="mt-6 text-xl font-semibold"
        style={{ color: 'var(--text-primary)' }}
      >
        Ask anything about your documents
      </h2>
    </div>
  );
}
