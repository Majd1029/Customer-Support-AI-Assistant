import { useState, useCallback } from 'react';
import { BookOpen, Eye, EyeOff } from 'lucide-react';

interface LoginScreenProps {
  onLogin:       (username: string, password: string) => Promise<string | null>;
  onRegister:    (username: string, password: string) => Promise<string | null>;
  onGoogleLogin: () => void;
}

type Tab = 'login' | 'register';

export default function LoginScreen({ onLogin, onRegister, onGoogleLogin }: LoginScreenProps) {
  const [tab,       setTab]      = useState<Tab>('login');
  const [username,  setUsername] = useState('');
  const [password,  setPassword] = useState('');
  const [showPwd,   setShowPwd]  = useState(false);
  const [error,     setError]    = useState<string | null>(null);
  const [loading,   setLoading]  = useState(false);

  const reset = (newTab: Tab) => {
    setTab(newTab);
    setError(null);
    setPassword('');
  };

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setError(null);
      setLoading(true);

      try {
        const err = tab === 'login'
          ? await onLogin(username, password)
          : await onRegister(username, password);
        if (err) setError(err);
      } finally {
        setLoading(false);
      }
    },
    [tab, username, password, onLogin, onRegister]
  );

  return (
    <div
      className="flex items-center justify-center h-full px-4"
      style={{ background: 'var(--bg-base)' }}
    >
      <div
        className="w-full max-w-sm rounded-2xl border p-8 shadow-2xl"
        style={{
          background:   'var(--bg-sidebar)',
          borderColor:  'var(--border-dark)',
        }}
      >
        {/* Logo */}
        <div className="flex items-center justify-center gap-2.5 mb-7">
          <div
            className="flex items-center justify-center w-10 h-10 rounded-xl"
            style={{ background: 'var(--accent)' }}
          >
            <BookOpen size={18} color="white" />
          </div>
          <span
            className="text-xl font-bold"
            style={{ color: 'var(--text-on-dark)' }}
          >
            CustomerAssist
          </span>
        </div>

        {/* Tab bar */}
        <div
          className="flex rounded-xl p-1 mb-6"
          style={{ background: 'rgba(255,255,255,0.05)' }}
        >
          {(['login', 'register'] as Tab[]).map(t => (
            <button
              key={t}
              onClick={() => reset(t)}
              className="flex-1 py-2 rounded-lg text-sm font-medium capitalize transition-all duration-150"
              style={
                tab === t
                  ? {
                      background: 'var(--accent)',
                      color:      'white',
                    }
                  : {
                      background: 'transparent',
                      color:      'var(--text-muted)',
                    }
              }
            >
              {t === 'login' ? 'Sign in' : 'Create account'}
            </button>
          ))}
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Username */}
          <div>
            <label
              htmlFor="auth-username"
              className="block text-xs font-medium mb-1.5"
              style={{ color: 'var(--text-muted)' }}
            >
              Username
            </label>
            <input
              id="auth-username"
              type="text"
              autoComplete="username"
              autoFocus
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="e.g. john_doe"
              className="w-full px-3 py-2.5 rounded-lg border text-sm outline-none transition-colors"
              style={{
                background:   'rgba(255,255,255,0.06)',
                borderColor:  error ? 'var(--danger)' : 'var(--border-dark)',
                color:        'var(--text-on-dark)',
              }}
              onFocus={e  => (e.currentTarget.style.borderColor = 'var(--accent)')}
              onBlur={e   => (e.currentTarget.style.borderColor = error ? 'var(--danger)' : 'var(--border-dark)')}
            />
          </div>

          {/* Password */}
          <div>
            <label
              htmlFor="auth-password"
              className="block text-xs font-medium mb-1.5"
              style={{ color: 'var(--text-muted)' }}
            >
              Password
            </label>
            <div className="relative">
              <input
                id="auth-password"
                type={showPwd ? 'text' : 'password'}
                autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder={tab === 'register' ? 'At least 4 characters' : '••••••••'}
                className="w-full px-3 py-2.5 pr-10 rounded-lg border text-sm outline-none transition-colors"
                style={{
                  background:   'rgba(255,255,255,0.06)',
                  borderColor:  error ? 'var(--danger)' : 'var(--border-dark)',
                  color:        'var(--text-on-dark)',
                }}
                onFocus={e => (e.currentTarget.style.borderColor = 'var(--accent)')}
                onBlur={e  => (e.currentTarget.style.borderColor = error ? 'var(--danger)' : 'var(--border-dark)')}
              />
              <button
                type="button"
                onClick={() => setShowPwd(v => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2"
                style={{ color: 'var(--text-muted)' }}
                aria-label={showPwd ? 'Hide password' : 'Show password'}
              >
                {showPwd ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
            </div>
          </div>

          {/* Error */}
          {error && (
            <p
              className="text-xs px-3 py-2 rounded-lg"
              style={{ color: 'var(--danger)', background: 'rgba(239,68,68,0.1)' }}
            >
              {error}
            </p>
          )}

          {/* Submit */}
          <button
            type="submit"
            disabled={loading || !username.trim() || !password}
            className="w-full py-2.5 rounded-lg text-sm font-semibold transition-all duration-150 mt-1"
            style={{
              background: loading || !username.trim() || !password
                ? 'var(--border-dark)'
                : 'var(--accent)',
              color: loading || !username.trim() || !password
                ? 'var(--text-muted)'
                : 'white',
            }}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <span
                  className="w-4 h-4 rounded-full border-2 border-t-transparent animate-spin"
                  style={{ borderColor: 'var(--text-muted)', borderTopColor: 'transparent' }}
                />
                Please wait…
              </span>
            ) : tab === 'login' ? 'Sign in' : 'Create account'}
          </button>
        </form>

        {/* ── Divider ───────────────────────────────────────────────── */}
        <div className="flex items-center gap-3 my-4">
          <div className="flex-1 h-px" style={{ background: 'var(--border-dark)' }} />
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>or</span>
          <div className="flex-1 h-px" style={{ background: 'var(--border-dark)' }} />
        </div>

        {/* ── Google Sign-In ─────────────────────────────────────────── */}
        <button
          type="button"
          onClick={onGoogleLogin}
          className="w-full flex items-center justify-center gap-2.5 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 border"
          style={{
            background:  'rgba(255,255,255,0.04)',
            borderColor: 'var(--border-dark)',
            color:       'var(--text-on-dark)',
          }}
          onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.08)')}
          onMouseLeave={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.04)')}
        >
          {/* Google G logo SVG */}
          <svg width="16" height="16" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
            <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
            <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
            <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
            <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
            <path fill="none" d="M0 0h48v48H0z"/>
          </svg>
          Continue with Google
        </button>

        {/* Footer hint */}
        <p className="text-center text-xs mt-5" style={{ color: 'var(--text-muted)' }}>
          {tab === 'login'
            ? "Don't have an account? "
            : 'Already have an account? '}
          <button
            className="underline hover:no-underline transition-all"
            style={{ color: 'var(--accent)' }}
            onClick={() => reset(tab === 'login' ? 'register' : 'login')}
          >
            {tab === 'login' ? 'Create one' : 'Sign in'}
          </button>
        </p>

        <p className="text-center text-xs mt-3" style={{ color: '#3a3d50', fontSize: '0.68rem' }}>
          Accounts are stored locally in this browser only.
        </p>

      </div>
    </div>
  );
}
