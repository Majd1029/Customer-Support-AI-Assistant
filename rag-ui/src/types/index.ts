// ─── Source chunk returned from the backend ───────────────────────────────────
export interface Source {
  filename:  string;
  page:      number | null;
  section:   string | null;
  score:     number | null;
  chunkId:   string | null;
  content:   string | null;   // first ~400 chars of the chunk text
}

// ─── Source scope (which documents to search) ────────────────────────────────
export interface ScopeState {
  mode: 'all' | 'specific';
  doc:  string | null;  // filename when mode === 'specific'
}

// ─── LLM-as-a-Judge result ────────────────────────────────────────────────────
export interface DimensionScore {
  score:     number | null;
  reasoning: string;
  dimension: string;
}

export interface JudgeResult {
  overall:    number | null;
  dimensions: Record<string, DimensionScore>;
  elapsed_ms: number | null;
  model:      string | null;
  error:      string | null;
}

// ─── Progressive eval node state ──────────────────────────────────────────────
export interface EvalNodes {
  semantic_alignment: { score: number | null; done: boolean };
  grounding_score:    { score: number | null; done: boolean };
  llm_judge:          { result: JudgeResult | null; done: boolean };
}

export interface EvalState {
  loading:         boolean;
  alignmentScore:  number | null;
  confidenceScore: number | null;
  verdict:         string | null;
  feedback:        string | null;
  judge:           JudgeResult | null;
  nodes:           EvalNodes;
}

// ─── Per-message metadata ─────────────────────────────────────────────────────
export interface MessageMetadata {
  confidence:        number | null;
  retrievalMs:       number | null;
  generationMs:      number | null;
  citationCount:     number | null;
  noAnswer:          boolean;
  hops:              number;
  rewrittenQuery:    string | null;
  rewriteTier:       string | null;
  verdict:           string | null;
  feedback:          string | null;
  // Set when the backend escalates the request to a human agent
  escalationMessage: string | null;
  // User's rating: 1 = helpful, -1 = not helpful, 0 = unrated
  rating:            0 | 1 | -1;
}

// ─── Chat message ─────────────────────────────────────────────────────────────
export interface Message {
  id:        string;
  role:      'user' | 'assistant';
  content:   string;
  streaming: boolean;
  sources:   Source[];
  evalState: EvalState | null;
  metadata:  MessageMetadata | null;
  error:     string | null;
}

// ─── Sidebar session entry ────────────────────────────────────────────────────
export interface Session {
  id:          string;
  label:       string;
  createdAt:   number;
  shareToken?: string | null;   // set once the user shares this conversation
}

// ─── Global settings ─────────────────────────────────────────────────────────
export interface Settings {
  memory:   boolean;
  judge:    boolean;
  multiHop: boolean;
  limit:    number;
}

// ─── Authenticated user ───────────────────────────────────────────────────────
export interface User {
  id:       string;   // user_id (UUID for server auth, username.toLowerCase() for legacy)
  username: string;   // display name (original casing)
  email?:   string;   // email address (server auth only)
  token?:   string;   // JWT bearer token (server auth only)
}

// ─── Health status ────────────────────────────────────────────────────────────
export type HealthStatus = 'ok' | 'degraded' | 'unreachable';
