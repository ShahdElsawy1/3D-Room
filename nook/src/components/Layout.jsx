import { Link, NavLink, Outlet, useMatch, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext.jsx';

export function Logo({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="logo-mark">
      <rect x="3" y="11" width="18" height="18" rx="5" fill="var(--lake)" />
      <rect x="11" y="3" width="18" height="18" rx="5" fill="var(--sun)" opacity=".95" />
      <rect x="11" y="13" width="18" height="16" rx="5" fill="var(--leaf)" opacity=".92" />
    </svg>
  );
}

export function Layout() {
  const { user, logout } = useAuth();
  const nav = useNavigate();
  const wide = Boolean(useMatch('/projects/:id/viewer')); // the 3D view gets the full window
  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/projects" className="brand"><Logo /><span>Nook</span></Link>
        <nav aria-label="Main">
          <NavLink to="/projects">Your rooms</NavLink>
        </nav>
        <div className="who">
          <span className="who-email">{user?.email}</span>
          <button type="button" className="btn btn-quiet" onClick={() => { logout(); nav('/login'); }}>Log out</button>
        </div>
      </header>
      <main className={`page ${wide ? 'wide' : ''}`}><Outlet /></main>
    </div>
  );
}
