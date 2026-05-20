/**
 * errors.ts — centralised friendly-error formatting for the UI.
 *
 * All API calls should run their raw error text through `friendlyError()`
 * before showing anything to the user.  This keeps every surface consistent
 * and avoids leaking internal stack traces, JSON blobs, or Groq billing URLs.
 */

/** Map an HTTP status + raw error text → a human-readable sentence. */
export function friendlyError(status: number, raw: string = ''): string {
  const r = raw.toLowerCase();

  // ── Rate limits / quota ────────────────────────────────────────────────────
  if (
    status === 429 ||
    r.includes('rate_limit_exceeded') ||
    r.includes('tokens per day') ||
    r.includes('tokens per minute') ||
    r.includes('tpd') ||
    r.includes('tpm')
  ) {
    return '⚠️ The AI service has reached its usage limit. Please wait a few minutes and try again.';
  }

  // ── Auth / permissions ─────────────────────────────────────────────────────
  if (status === 401) return '⚠️ You are not signed in. Please sign in and try again.';
  if (status === 403) return '⚠️ You don\'t have permission to do that.';

  // ── Not found ──────────────────────────────────────────────────────────────
  if (status === 404) return '⚠️ The requested resource was not found.';

  // ── Server / service down ──────────────────────────────────────────────────
  if (
    status === 503 ||
    r.includes('unavailable') ||
    r.includes('service unavailable')
  ) {
    return '⚠️ The AI service is temporarily unavailable. Please try again in a moment.';
  }

  if (status === 504) return '⚠️ The request timed out. The file may be too large — please try again.';

  if (status === 500 || r.includes('internal server error')) {
    return '⚠️ Something went wrong on the server. Please try again.';
  }

  // ── Upload / processing specific ───────────────────────────────────────────
  if (r.includes('no space left') || r.includes('disk full')) {
    return '⚠️ Server storage is full. Please contact the administrator.';
  }
  if (r.includes('file too large') || r.includes('413')) {
    return '⚠️ The file is too large to upload. Please try a smaller file.';
  }
  if (r.includes('unsupported') || r.includes('format') || r.includes('extension')) {
    return '⚠️ This file type is not supported.';
  }

  // ── Network / connection ───────────────────────────────────────────────────
  if (
    r.includes('failed to fetch') ||
    r.includes('network error') ||
    r.includes('networkerror') ||
    r.includes('connection refused') ||
    r.includes('econnrefused')
  ) {
    return '⚠️ Could not reach the server. Check your connection and try again.';
  }

  if (r.includes('aborted') || r.includes('abort')) {
    return ''; // user-initiated cancel — don't show an error
  }

  // ── Fallback: scrub the raw text of anything that looks like JSON / keys ───
  const scrubbed = raw
    .replace(/\{[^}]*\}/g, '')            // remove JSON objects
    .replace(/gsk_[A-Za-z0-9]+/g, '***') // scrub API keys
    .replace(/https?:\/\/\S+/g, '')       // remove URLs
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 120);

  if (scrubbed.length > 10) return `⚠️ ${scrubbed}`;
  return '⚠️ An unexpected error occurred. Please try again.';
}

/** Parse a fetch Response body and return a friendly error string. */
export async function friendlyFetchError(res: Response): Promise<string> {
  const raw = await res.text().catch(() => '');
  // Try to extract .detail from JSON responses (FastAPI pattern)
  try {
    const json = JSON.parse(raw) as Record<string, unknown>;
    const detail = (json.detail ?? json.message ?? json.error ?? '') as string;
    if (detail) return friendlyError(res.status, detail);
  } catch { /* not JSON */ }
  return friendlyError(res.status, raw);
}
