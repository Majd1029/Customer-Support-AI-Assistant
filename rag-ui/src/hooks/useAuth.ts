import { useState, useCallback, useEffect } from 'react';
import type { User } from '../types';

const API_URL     = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';
const SESSION_KEY = 'docassist_current_user';

// ── Legacy localStorage helpers (backwards compat for users without server) ───
const USERS_KEY = 'docassist_users';

interface LegacyUser {
  id: string; username: string; passwordHash: string;
}

function djb2(s: string): string {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h) ^ s.charCodeAt(i), h = h >>> 0;
  return h.toString(16).padStart(8, '0');
}

function getLegacyUsers(): LegacyUser[] {
  try { return JSON.parse(localStorage.getItem(USERS_KEY) ?? '[]') as LegacyUser[]; } catch { return []; }
}
function saveLegacyUsers(users: LegacyUser[]): void { localStorage.setItem(USERS_KEY, JSON.stringify(users)); }

// ── Hook ───────────────────────────────────────────────────────────────────────
export function useAuth() {
  const [user, setUser] = useState<User | null>(() => {
    try {
      const raw = localStorage.getItem(SESSION_KEY);
      if (!raw) return null;
      const stored = JSON.parse(raw) as User;
      // Legacy localStorage users have no JWT token and cannot access any
      // server-protected route. Drop them immediately so the login screen
      // is shown — they need to sign in via the server to get a token.
      if (!stored.token) {
        localStorage.removeItem(SESSION_KEY);
        return null;
      }
      return stored;
    } catch { return null; }
  });

  // Verify the stored JWT is still valid on mount.
  // If the server returns 401 (expired / revoked), clear the session.
  // Network errors are ignored — don't log out if the server is down.
  useEffect(() => {
    if (!user?.token) return;
    fetch(`${API_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${user.token}` },
    })
      .then(r => {
        if (r.status === 401) {
          localStorage.removeItem(SESSION_KEY);
          setUser(null);
        }
      })
      .catch(() => { /* server unreachable — keep session */ });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // intentionally runs once on mount

  const _persist = (u: User) => {
    localStorage.setItem(SESSION_KEY, JSON.stringify(u));
    setUser(u);
  };

  // ── Server-side login (JWT) ────────────────────────────────────────────────
  const login = useCallback(async (username: string, password: string): Promise<string | null> => {
    // Try server first
    try {
      const res = await fetch(`${API_URL}/auth/login`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ username_or_email: username, password }),
      });
      if (res.ok) {
        const data = await res.json() as { user_id: string; username: string; email: string; token: string };
        _persist({ id: data.user_id, username: data.username, email: data.email, token: data.token });
        return null;
      }
      if (res.status === 401 || res.status === 400) {
        const err = await res.json().catch(() => ({ detail: 'Login failed' })) as { detail?: string };
        // If server says "account not found", fall through to legacy
        if (err.detail?.includes('not found')) {
          // fallthrough to legacy
        } else {
          return err.detail ?? 'Login failed';
        }
      }
    } catch {
      // Server unavailable — fall back to localStorage
    }

    // Legacy localStorage fallback
    const users  = getLegacyUsers();
    const found  = users.find(u => u.id === username.trim().toLowerCase());
    if (!found)  return 'Account not found. Did you mean to register?';
    if (found.passwordHash !== djb2(password)) return 'Incorrect password';
    _persist({ id: found.id, username: found.username });
    return null;
  }, []);

  // ── Server-side register (JWT) ─────────────────────────────────────────────
  const register = useCallback(async (username: string, password: string, email: string = ''): Promise<string | null> => {
    const trimmed = username.trim();
    if (!trimmed || trimmed.length < 2) return 'Username must be at least 2 characters';
    if (password.length < 4) return 'Password must be at least 4 characters';

    // Try server first
    try {
      const res = await fetch(`${API_URL}/auth/register`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ username: trimmed, password, email }),
      });
      if (res.ok) {
        const data = await res.json() as { user_id: string; username: string; email: string; token: string };
        _persist({ id: data.user_id, username: data.username, email: data.email, token: data.token });
        return null;
      }
      const err = await res.json().catch(() => ({ detail: 'Registration failed' })) as { detail?: string };
      if (res.status !== 503) return err.detail ?? 'Registration failed';
      // 503 = server unavailable → localStorage fallback
    } catch {
      // Server unavailable — fall back to localStorage
    }

    // Legacy localStorage fallback
    const users = getLegacyUsers();
    if (users.find(u => u.id === trimmed.toLowerCase())) return 'That username is already taken';
    const newUser: LegacyUser = { id: trimmed.toLowerCase(), username: trimmed, passwordHash: djb2(password) };
    saveLegacyUsers([...users, newUser]);
    _persist({ id: newUser.id, username: newUser.username });
    return null;
  }, []);

  // ── Google OAuth (popup) ───────────────────────────────────────────────────
  const loginWithGoogle = useCallback((): void => {
    const popup = window.open(
      `${API_URL}/auth/google`,
      'google_oauth',
      'width=520,height=620,left=100,top=100,resizable=yes,scrollbars=yes',
    );
    if (!popup) {
      alert('Please allow popups for this site to use Google Sign-In');
      return;
    }

    const handler = (e: MessageEvent) => {
      if (e.origin !== new URL(API_URL).origin && e.origin !== window.location.origin) return;
      const msg = e.data as { type?: string; user?: { user_id: string; username: string; email: string; token: string }; error?: string };
      if (msg?.type !== 'google_auth') return;
      window.removeEventListener('message', handler);
      if (msg.error) {
        alert(`Google sign-in failed: ${msg.error}`);
        return;
      }
      if (msg.user) {
        _persist({ id: msg.user.user_id, username: msg.user.username, email: msg.user.email, token: msg.user.token });
      }
    };
    window.addEventListener('message', handler);

    // Also handle the sessionStorage fallback redirect
    const pollTimer = setInterval(() => {
      try {
        if (popup.closed) {
          clearInterval(pollTimer);
          window.removeEventListener('message', handler);
          // Check sessionStorage fallback
          const stored = sessionStorage.getItem('google_auth_result');
          if (stored) {
            sessionStorage.removeItem('google_auth_result');
            const parsed = JSON.parse(stored) as { type?: string; user?: { user_id: string; username: string; email: string; token: string } };
            if (parsed?.user) {
              _persist({ id: parsed.user.user_id, username: parsed.user.username, email: parsed.user.email, token: parsed.user.token });
            }
          }
        }
      } catch { clearInterval(pollTimer); }
    }, 500);
  }, []);

  // ── Check sessionStorage on mount (for redirect fallback) ─────────────────
  const checkGoogleRedirect = useCallback((): void => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('google_auth') === '1') {
      const stored = sessionStorage.getItem('google_auth_result');
      if (stored) {
        sessionStorage.removeItem('google_auth_result');
        try {
          const parsed = JSON.parse(stored) as { user?: { user_id: string; username: string; email: string; token: string } };
          if (parsed?.user) {
            _persist({ id: parsed.user.user_id, username: parsed.user.username, email: parsed.user.email, token: parsed.user.token });
          }
        } catch { /* ignore */ }
      }
      // Clean up URL
      window.history.replaceState({}, '', window.location.pathname);
    }
  }, []);

  // Check on every render (safe — only fires once due to sessionStorage clear)
  checkGoogleRedirect();

  const logout = useCallback((): void => {
    localStorage.removeItem(SESSION_KEY);
    setUser(null);
  }, []);

  return { user, login, register, logout, loginWithGoogle };
}
