import { useState, useCallback, useRef, useEffect } from 'react';
import type { Message, Session, Settings, Source, EvalState, JudgeResult, ScopeState, MessageMetadata } from '../types';

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';
const JSON_HEADERS = { 'Content-Type': 'application/json' };

/**
 * Lightweight client-side language hint detector.
 *
 * Returns an ISO 639-1 code that is passed to the backend as `language_hint`
 * so the answer is generated in the same language as the query, without
 * requiring the user to manually pick a language in Settings.
 *
 * Priority:
 *   1. Arabic Unicode block  → 'ar'
 *   2. French diacritics / common French function words → 'fr'
 *   3. Fallback → '' (backend auto-detects via langdetect)
 */
function detectQueryLanguage(text: string): string {
  // Arabic: any character in the Arabic Unicode block
  if (/[؀-ۿݐ-ݿࢠ-ࣿ]/.test(text)) return 'ar';
  // French: accented characters typical to French OR short French function words
  if (
    /[àâæçéèêëîïôùûüœÀÂÆÇÉÈÊËÎÏÔÙÛÜŒ]/.test(text) ||
    /\b(comment|pourquoi|qu[''e]|est-ce|bonjour|merci|je\s|tu\s|nous\s|vous\s|c'est|qu'est|quoi)\b/i.test(text)
  ) return 'fr';
  return '';
}

function makeEmptyEvalNodes(): EvalState['nodes'] {
  return {
    semantic_alignment: { score: null, done: false },
    grounding_score:    { score: null, done: false },
    llm_judge:          { result: null, done: false },
  };
}

function parseSource(s: Record<string, unknown>): Source {
  return {
    filename: ((s.source ?? s.filename ?? '') as string),
    page:     (s.page_start ?? s.page ?? null) as number | null,
    section:  (s.section ?? null) as string | null,
    score:    (s.score ?? null) as number | null,
    chunkId:  (s.chunk_id ?? null) as string | null,
    content:  (s.content ?? null) as string | null,
  };
}

function sessionsKey(userId: string): string {
  return `docassist_sessions_${userId}`;
}

function loadSessions(userId: string): Session[] {
  try {
    return JSON.parse(localStorage.getItem(sessionsKey(userId)) ?? '[]') as Session[];
  } catch {
    return [];
  }
}

export function useChat(userId: string, userToken: string = '', onAuthError?: () => void) {
  const [messages, setMessages] = useState<Message[]>([]);
  // Live reference — lets callbacks read current messages without stale closure.
  // IMPORTANT: updated synchronously during render (not in a useEffect) so that
  // setTimeout(0) callbacks in the sendMessage finally block always see the
  // latest messages — including the assistant message with streaming:false set
  // by the 'done' SSE handler.  If we used a useEffect here, the effect might
  // not have fired yet when setTimeout(0) runs, causing the filter
  // `!m.streaming` to exclude the assistant message and save only the user turn.
  const messagesRef = useRef<Message[]>(messages);
  messagesRef.current = messages;  // synchronous, no re-render triggered

  const [sessions, setSessions] = useState<Session[]>(() => loadSessions(userId));
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  // Live reference for session id (avoids stale closure in submitFeedback).
  // Updated synchronously during render for the same reason as messagesRef above.
  const currentSessionIdRef = useRef<string | null>(currentSessionId);
  currentSessionIdRef.current = currentSessionId;
  const [settings, setSettings] = useState<Settings>({
    memory:   true,
    judge:    false,
    multiHop: false,
    limit:    5,
  });
  const [scope, setScope] = useState<ScopeState>({ mode: 'all', doc: null });
  const [indexedDocs, setIndexedDocs] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // ── Token header helper ────────────────────────────────────────────────────
  // IMPORTANT: set tokenRef.current synchronously during render (not in an
  // effect) so that effects firing in the same render cycle (e.g. the userId
  // change effect that fetches /conversations) always see the current token.
  // This avoids two bugs:
  //   1. Race condition on login — userId effect fires before setToken effect.
  //   2. Cross-user history leak — stale token from previous user is used when
  //      a new user logs in, causing the server to return the wrong conversations.
  const tokenRef = useRef<string>(userToken);
  tokenRef.current = userToken;   // synchronous, no re-render triggered

  const authHeaders = useCallback((): Record<string, string> => {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    if (tokenRef.current) h['Authorization'] = `Bearer ${tokenRef.current}`;
    return h;
  }, []);

  // ── Fetch list of indexed documents (scoped to this user via JWT) ─────────
  const refreshIndexedDocs = useCallback(async () => {
    try {
      // The backend applies RBAC via JWT: users see their own docs + global KB,
      // admins see everything. No owner_id param needed from the client.
      const res = await fetch(`${API_URL}/index/documents?limit=200`, {
        headers: authHeaders(),
      });
      if (!res.ok) return;
      const data = (await res.json()) as { documents?: Array<{ source: string }> };
      const names = (data.documents ?? []).map(d => d.source).filter(Boolean);
      setIndexedDocs(names);
    } catch {
      // Silently ignore — Qdrant may not be running
    }
  }, [userId]);

  // Fetch on mount and whenever userId changes
  useEffect(() => { refreshIndexedDocs(); }, [userId, refreshIndexedDocs]);

  // ── Persist sessions to localStorage whenever they change ────────────────────
  // IMPORTANT: we must NOT persist when userId just changed but sessions haven't
  // been reset yet. React fires effects in declaration order, and the persist
  // effect appears BEFORE the reset effect. On a userId change both effects
  // fire in the same flush: persist runs first with (userId=newUser, sessions=oldUser's
  // data) and would write the old user's sessions into the new user's localStorage
  // key — permanently contaminating their history.
  //
  // Guard: sessionsOwnerRef tracks which user the current `sessions` state belongs
  // to. The reset effect updates this ref FIRST, before calling setSessions.
  // So when the persist effect fires on a userId change, sessionsOwnerRef.current
  // is still the OLD userId — the mismatch causes it to skip the write safely.
  const sessionsOwnerRef = useRef<string>(userId);

  useEffect(() => {
    if (sessionsOwnerRef.current !== userId) return;  // sessions not yet reset for this user
    localStorage.setItem(sessionsKey(userId), JSON.stringify(sessions));
  }, [sessions, userId]);

  // ── Reset chat when userId changes (user switches account) ─────────────────
  useEffect(() => {
    // Update the owner ref BEFORE setSessions so the persist effect above
    // sees the correct owner on the very next render.
    sessionsOwnerRef.current = userId;
    abortRef.current?.abort();
    setIsStreaming(false);
    setMessages([]);
    setCurrentSessionId(null);
    setSessions(loadSessions(userId));
    setScope({ mode: 'all', doc: null });

    // Merge server sessions with localStorage.
    // The server's list_sessions() returns camelCase {id, label, createdAt, shareToken}
    // which already matches the frontend Session interface — no remapping needed.
    if (userId !== '__guest__') {
      fetch(`${API_URL}/conversations`, { headers: authHeaders() })
        .then(r => {
          if (r.status === 401) { onAuthError?.(); return null; }
          return r.ok ? r.json() : null;
        })
        .then((data: {
          conversations?: Array<{
            id:          string;
            label:       string;
            createdAt:   number;
            shareToken?: string | null;
          }>;
        } | null) => {
          if (data?.conversations?.length) {
            const serverSessions: Session[] = data.conversations.map(c => ({
              id:         c.id,
              label:      c.label ?? 'Chat',
              createdAt:  c.createdAt,
              shareToken: c.shareToken ?? null,
            }));
            // Server is the authoritative source of user-scoped sessions.
            // Replace localStorage entirely with the server's verified list so
            // any previously cross-contaminated sessions are wiped out.
            localStorage.setItem(sessionsKey(userId), JSON.stringify(serverSessions));
            setSessions(serverSessions);
          }
        })
        .catch(() => {});
    }
  }, [userId]);

  // ── Load messages for a session from server ────────────────────────────────
  const loadSessionMessages = useCallback(async (sessionId: string) => {
    if (!sessionId || userId === '__guest__') return;
    try {
      const res = await fetch(
        `${API_URL}/conversations/${sessionId}/messages`,
        { headers: authHeaders() },
      );
      if (res.status === 401) { onAuthError?.(); return; }
      if (!res.ok) return;
      const data = await res.json() as { messages?: Array<{ role: string; content: string; metadata: unknown }> };
      const loaded: Message[] = (data.messages ?? []).map(m => ({
        id:        crypto.randomUUID(),
        role:      m.role as 'user' | 'assistant',
        content:   m.content,
        streaming: false,
        sources:   (m.metadata as Record<string, unknown> | null)?.['sources'] as Source[] ?? [],
        evalState: (m.metadata as Record<string, unknown> | null)?.['evalState'] as EvalState | null ?? null,
        metadata:  (m.metadata as Record<string, unknown> | null)?.['messageMetadata'] as Message['metadata'] ?? null,
        error:     null,
      }));
      setMessages(loaded);
    } catch { /* silently ignore */ }
  }, [userId, authHeaders]);

  // ── Helpers ────────────────────────────────────────────────────────────────
  const updateMessage = useCallback(
    (id: string, updater: (msg: Message) => Message) =>
      setMessages(prev => prev.map(m => (m.id === id ? updater(m) : m))),
    []
  );

  const createSession = useCallback(async (): Promise<string> => {
    try {
      const res = await fetch(`${API_URL}/sessions`, {
        method:  'POST',
        headers: JSON_HEADERS,
      });
      if (!res.ok) throw new Error('Non-OK');
      const data = (await res.json()) as { session_id: string };
      const sid = data.session_id;
      // Register in conversation store (user_id comes from JWT, not body)
      if (userId !== '__guest__') {
        fetch(`${API_URL}/conversations/${sid}/messages`, {
          method: 'POST',
          headers: authHeaders(),
          body: JSON.stringify({ messages: [], label: 'New chat' }),
        }).catch(() => {});
      }
      return sid;
    } catch {
      return crypto.randomUUID();
    }
  }, [userId, authHeaders]);

  // ── Public actions ─────────────────────────────────────────────────────────
  const startNewChat = useCallback(async () => {
    abortRef.current?.abort();
    setIsStreaming(false);
    const sid = await createSession();
    setCurrentSessionId(sid);
    setMessages([]);
    setSessions(prev => [
      { id: sid, label: 'New chat', createdAt: Date.now() },
      ...prev,
    ]);
  }, [createSession]);

  const sendMessage = useCallback(
    async (input: string) => {
      if (!input.trim() || isStreaming) return;

      let sid = currentSessionId;
      if (!sid) {
        sid = await createSession();
        setCurrentSessionId(sid);
        setSessions(prev => [
          { id: sid!, label: input.slice(0, 50), createdAt: Date.now() },
          ...prev,
        ]);
      } else {
        setSessions(prev =>
          prev.map(s =>
            s.id === sid && s.label === 'New chat'
              ? { ...s, label: input.slice(0, 50) }
              : s
          )
        );
      }

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: 'user',
        content: input,
        streaming: false,
        sources: [],
        evalState: null,
        metadata: null,
        error: null,
      };

      const assistantId = crypto.randomUUID();
      const assistantMsg: Message = {
        id: assistantId,
        role: 'assistant',
        content: '',
        streaming: true,
        sources: [],
        evalState: null,
        metadata: null,
        error: null,
      };

      setMessages(prev => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      // Build source_filter based on current scope
      const sourceFilter =
        scope.mode === 'specific' && scope.doc
          ? { sources: [scope.doc] }
          : undefined;

      // Capture sid for use in finally block
      const activeSid = sid;

      try {
          // Build request headers — include JWT so the backend can apply RBAC
        // (users see own docs + global KB; admins see all). No owner_id needed
        // from the client — the server resolves it from the token.
        const askHeaders: Record<string, string> = { 'Content-Type': 'application/json' };
        if (tokenRef.current) askHeaders['Authorization'] = `Bearer ${tokenRef.current}`;

        const response = await fetch(`${API_URL}/ask/stream`, {
          method:  'POST',
          headers: askHeaders,
          signal:  controller.signal,
          body: JSON.stringify({
            question:       input,
            session_id:     sid,
            memory_enabled: settings.memory,
            judge:          settings.judge,
            multi_hop:      settings.multiHop,
            limit:          settings.limit,
            // Detected from the query text so the answer comes back in the
            // same language automatically — no manual language setting needed.
            language_hint:  detectQueryLanguage(input),
            ...(sourceFilter ? { source_filter: sourceFilter } : {}),
          }),
        });

        if (!response.ok) {
          const errText = await response.text().catch(() => `HTTP ${response.status}`);
          updateMessage(assistantId, m => ({
            ...m,
            streaming: false,
            error: `Server error ${response.status}: ${errText}`,
          }));
          return;
        }

        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        const processEvent = (jsonStr: string) => {
          let evt: Record<string, unknown>;
          try { evt = JSON.parse(jsonStr) as Record<string, unknown>; }
          catch { return; }

          switch (evt.type as string) {
            case 'sources': {
              const raw = (evt.sources ?? []) as Array<Record<string, unknown>>;
              updateMessage(assistantId, m => ({ ...m, sources: raw.map(parseSource) }));
              break;
            }
            case 'token': {
              updateMessage(assistantId, m => ({
                ...m,
                content: m.content + ((evt.content ?? '') as string),
              }));
              break;
            }
            case 'eval_start': {
              updateMessage(assistantId, m => ({
                ...m,
                evalState: {
                  loading: true,
                  alignmentScore: null,
                  confidenceScore: null,
                  verdict: null,
                  feedback: null,
                  judge: null,
                  nodes: makeEmptyEvalNodes(),
                },
              }));
              break;
            }
            case 'eval_node': {
              const node = evt.node as string;
              updateMessage(assistantId, m => {
                if (!m.evalState) return m;
                const nodes = { ...m.evalState.nodes };
                if (node === 'semantic_alignment') {
                  nodes.semantic_alignment = { score: (evt.alignment_score ?? null) as number | null, done: true };
                } else if (node === 'grounding_score') {
                  nodes.grounding_score = { score: (evt.confidence_score ?? null) as number | null, done: true };
                } else if (node === 'llm_judge') {
                  nodes.llm_judge = { result: (evt.judge ?? null) as JudgeResult | null, done: true };
                }
                return { ...m, evalState: { ...m.evalState, nodes } };
              });
              break;
            }
            case 'eval_done': {
              updateMessage(assistantId, m => ({
                ...m,
                evalState: {
                  loading:         false,
                  alignmentScore:  (evt.alignment_score  ?? null) as number | null,
                  confidenceScore: (evt.confidence_score ?? null) as number | null,
                  verdict:         (evt.verdict   ?? null) as string | null,
                  feedback:        (evt.feedback  ?? null) as string | null,
                  judge:           (evt.judge     ?? null) as JudgeResult | null,
                  nodes:           m.evalState?.nodes ?? makeEmptyEvalNodes(),
                },
              }));
              break;
            }
            case 'escalation': {
              // Backend has escalated to a human agent — surface the message.
              const escalationMsg = (evt.message ?? '') as string;
              updateMessage(assistantId, m => ({
                ...m,
                metadata: {
                  ...(m.metadata ?? {
                    confidence: null, retrievalMs: null, generationMs: null,
                    citationCount: null, noAnswer: false, hops: 1,
                    rewrittenQuery: null, rewriteTier: null,
                    verdict: null, feedback: null, rating: 0,
                  }),
                  escalationMessage: escalationMsg,
                } as MessageMetadata,
              }));
              break;
            }
            case 'done': {
              updateMessage(assistantId, m => ({
                ...m,
                streaming: false,
                metadata: {
                  confidence:        (evt.confidence     ?? null) as number | null,
                  retrievalMs:       (evt.retrieval_ms   ?? null) as number | null,
                  generationMs:      (evt.generation_ms  ?? null) as number | null,
                  citationCount:     (evt.citation_count ?? null) as number | null,
                  noAnswer:          !!(evt.no_answer),
                  hops:              (evt.hops ?? 1) as number,
                  rewrittenQuery:    (evt.rewritten_question ?? null) as string | null,
                  rewriteTier:       (evt.rewrite_tier       ?? null) as string | null,
                  verdict:           (evt.eval_verdict       ?? null) as string | null,
                  feedback:          (evt.eval_feedback      ?? null) as string | null,
                  escalationMessage: m.metadata?.escalationMessage ?? null,
                  rating:            m.metadata?.rating ?? 0,
                },
                evalState: m.evalState ? { ...m.evalState, loading: false } : null,
              }));
              break;
            }
          }
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split('\n\n');
          buffer = events.pop() ?? '';
          for (const block of events) {
            for (const line of block.split('\n')) {
              if (!line.startsWith('data: ')) continue;
              const s = line.slice(6).trim();
              if (s) processEvent(s);
            }
          }
        }

      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          updateMessage(assistantId, m => ({
            ...m,
            streaming: false,
            error: (err as Error).message || 'Connection failed',
          }));
        }
      } finally {
        setIsStreaming(false);
        // Persist messages to server (fire-and-forget).
        // IMPORTANT: do NOT call fetch inside a setMessages updater — React StrictMode
        // invokes state updaters twice in development, which would double-POST.
        // Use setTimeout(0) + messagesRef to read current state outside the render cycle.
        if (userId !== '__guest__' && activeSid) {
          setTimeout(() => {
            const cur = messagesRef.current;
            const toSave = cur
              .filter(m => !m.streaming && !m.error)
              .map(m => ({
                role:     m.role,
                content:  m.content,
                metadata: { sources: m.sources, evalState: m.evalState, messageMetadata: m.metadata },
              }));
            const label = cur.find(m => m.role === 'user')?.content?.slice(0, 60) ?? 'Chat';
            fetch(`${API_URL}/conversations/${activeSid}/messages`, {
              method:  'POST',
              headers: authHeaders(),
              body: JSON.stringify({ messages: toSave, label }),
            }).catch(() => {/* ignore */});
          }, 0);
        }
      }
    },
    [currentSessionId, isStreaming, settings, scope, createSession, updateMessage, userId, authHeaders]
  );

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    setIsStreaming(false);
    setMessages(prev => prev.map(m => (m.streaming ? { ...m, streaming: false } : m)));
  }, []);

  const deleteSession = useCallback((sessionId: string) => {
    setSessions(prev => prev.filter(s => s.id !== sessionId));
    if (currentSessionId === sessionId) {
      setCurrentSessionId(null);
      setMessages([]);
    }
    // Delete from server (identity comes from JWT, not query params)
    if (userId !== '__guest__') {
      fetch(`${API_URL}/conversations/${sessionId}`, {
        method: 'DELETE',
        headers: authHeaders(),
      }).catch(() => {});
    }
  }, [currentSessionId, userId, authHeaders]);

  /**
   * Submit a thumbs up (1) or thumbs down (-1) rating for an assistant message.
   * Looks up the preceding user message to supply the question text.
   * Fires POST /feedback — never throws.
   */
  const submitFeedback = useCallback((messageId: string, rating: 1 | -1) => {
    const msgs = messagesRef.current;
    const idx  = msgs.findIndex(m => m.id === messageId);
    if (idx < 0) return;
    const assistantMsg = msgs[idx];
    const userMsg      = msgs.slice(0, idx).reverse().find(m => m.role === 'user');

    // Optimistically update the UI rating
    updateMessage(messageId, m => ({
      ...m,
      metadata: m.metadata
        ? { ...m.metadata, rating }
        : null,
    }));

    fetch(`${API_URL}/feedback`, {
      method:  'POST',
      headers: JSON_HEADERS,
      body: JSON.stringify({
        question:     userMsg?.content   ?? '',
        answer:       assistantMsg.content,
        session_id:   currentSessionIdRef.current ?? '',
        rating,
        eval_verdict: assistantMsg.metadata?.verdict ?? null,
        confidence:   assistantMsg.metadata?.confidence ?? null,
        escalated:    !!assistantMsg.metadata?.escalationMessage,
      }),
    }).catch(() => { /* fire-and-forget */ });
  }, [updateMessage]);

  /**
   * Rename a session locally and persist to the server.
   */
  const renameSession = useCallback((sessionId: string, label: string) => {
    if (!label.trim()) return;
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, label } : s));
    if (userId !== '__guest__') {
      fetch(`${API_URL}/conversations/${sessionId}/label`, {
        method:  'PUT',
        headers: authHeaders(),
        body:    JSON.stringify({ label }),
      }).catch(() => {});
    }
  }, [userId, authHeaders]);

  /**
   * Generate (or retrieve) the share token for a conversation.
   * Returns the full share URL on success, null on failure.
   * Updates the local session list so the share icon renders immediately.
   */
  const shareSession = useCallback(async (sessionId: string): Promise<string | null> => {
    if (userId === '__guest__') return null;
    try {
      const res = await fetch(`${API_URL}/conversations/${sessionId}/share`, {
        method:  'POST',
        headers: authHeaders(),
      });
      if (!res.ok) return null;
      const { share_token, share_url } = await res.json() as { share_token: string; share_url: string };
      // Persist the token in local session list so the share icon stays visible
      setSessions(prev => prev.map(s =>
        s.id === sessionId ? { ...s, shareToken: share_token } : s
      ));
      return share_url;
    } catch {
      return null;
    }
  }, [userId, authHeaders]);

  return {
    messages,
    sessions,
    currentSessionId,
    settings,
    setSettings,
    scope,
    setScope,
    indexedDocs,
    refreshIndexedDocs,
    isStreaming,
    sendMessage,
    startNewChat,
    stopStreaming,
    deleteSession,
    renameSession,
    shareSession,
    submitFeedback,
    setCurrentSessionId,
    setMessages,
    loadSessionMessages,
  };
}
