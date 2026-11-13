import { useState } from 'react';
import { Navigate, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext.jsx';
import { getApiBase, setApiBase } from '../api/client.js';
import { health } from '../api/endpoints.js';
import { ErrorNote, Modal, Spinner } from '../components/ui.jsx';
import { Logo } from '../components/Layout.jsx';

function RoomArt() {
  // an open-topped room seen from above: floor, two back walls, two windows, and a "you are here" marker
  return (
    <svg viewBox="0 0 320 250" className="room-art" aria-hidden="true">
      <polygon points="160,125 270,180 160,235 50,180" fill="var(--sun-soft)" />
      <polygon points="50,180 160,125 160,40 50,95" fill="var(--lake)" />
      <polygon points="160,125 270,180 270,95 160,40" fill="var(--leaf)" />
      <polygon points="72,150 118,127 118,88 72,111" fill="var(--sun)" />
      <polygon points="202,127 248,150 248,111 202,88" fill="var(--sun)" opacity=".9" />
      <polygon points="118,176 158,156 190,172 150,192" fill="var(--leaf-deep)" opacity=".85" />
      <circle cx="160" cy="190" r="9" fill="var(--lake)" /><circle cx="160" cy="190" r="3.5" fill="#fff" />
    </svg>
  );
}

export function LoginPage() {
  const { user, login, signup } = useAuth();
  const nav = useNavigate();
  const loc = useLocation();
  const [mode, setMode] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [showServer, setShowServer] = useState(false);

  if (user) return <Navigate to={loc.state?.from || '/projects'} replace />;

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      await (mode === 'login' ? login : signup)(email.trim(), password);
      nav(loc.state?.from || '/projects', { replace: true });
    } catch (err) { setError(err); } finally { setBusy(false); }
  };

  return (
    <div className="login">
      <section className="login-hero">
        <Logo size={44} />
        <h1>Walk through a room once.<br />Step back into it any time.</h1>
        <p>Film a slow lap of a room, upload it, and Nook turns it into a 3D space you can wander around in your browser.</p>
        <RoomArt />
      </section>

      <section className="login-card">
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={mode === 'login'} className={mode === 'login' ? 'on' : ''} onClick={() => { setMode('login'); setError(null); }}>Log in</button>
          <button role="tab" aria-selected={mode === 'signup'} className={mode === 'signup' ? 'on' : ''} onClick={() => { setMode('signup'); setError(null); }}>Create account</button>
        </div>
        <form onSubmit={submit} noValidate>
          <label>Email
            <input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
          </label>
          <label>Password
            <input type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} required minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} placeholder="At least 8 characters" />
          </label>
          <ErrorNote error={error} />
          <button className="btn btn-primary wide" disabled={busy || !email || password.length < 1}>
            {busy ? <Spinner /> : mode === 'login' ? 'Log in' : 'Create my account'}
          </button>
        </form>
        <button type="button" className="linklike server-link" onClick={() => setShowServer(true)}>Server: {getApiBase()}</button>
      </section>
      {showServer && <ServerDialog onClose={() => setShowServer(false)} />}
    </div>
  );
}

/** The Cloudflare tunnel URL changes every run, so it can be changed here without rebuilding. */
export function ServerDialog({ onClose }) {
  const [url, setUrl] = useState(getApiBase());
  const [result, setResult] = useState(null);
  const [checking, setChecking] = useState(false);
  const test = async () => {
    setApiBase(url); setChecking(true); setResult(null);
    try { const h = await health(); setResult({ ok: true, text: `Connected — backend v${h.version}` }); } catch (e) { setResult({ ok: false, text: e.message }); } finally { setChecking(false); }
  };
  return (
    <Modal title="Backend server" onClose={onClose}>
      <p className="muted">Use <code>http://localhost:8000</code> locally, or your Cloudflare tunnel URL.</p>
      <label>Server address
        <input value={url} onChange={(e) => setUrl(e.target.value)} spellCheck={false} />
      </label>
      {result && <div className={`note ${result.ok ? 'note-good' : 'note-bad'}`} role="status">{result.text}</div>}
      <div className="row end">
        <button className="btn btn-quiet" onClick={test} disabled={checking}>{checking ? <Spinner /> : 'Save & test'}</button>
        <button className="btn btn-primary" onClick={() => { setApiBase(url); onClose(); }}>Done</button>
      </div>
    </Modal>
  );
}
