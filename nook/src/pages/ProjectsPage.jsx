import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { projects as api } from '../api/endpoints.js';
import { ErrorNote, Spinner, StatusBadge } from '../components/ui.jsx';
import { timeAgo } from '../lib/format.js';

export function ProjectsPage() {
  const nav = useNavigate();
  const [list, setList] = useState(null);
  const [error, setError] = useState(null);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);

  const load = () => api.list().then(setList).catch(setError);
  useEffect(() => { load(); }, []);

  const create = async (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    setBusy(true); setError(null);
    try { const p = await api.create(name.trim()); nav(`/projects/${p.id}`); } catch (err) { setError(err); } finally { setBusy(false); }
  };

  const remove = async (p) => {
    if (!window.confirm(`Delete “${p.name}” and all its files? This cannot be undone.`)) return;
    try { await api.remove(p.id); load(); } catch (err) { setError(err); }
  };
 
  const target = (p) => {
    if (p.latest_task_status === 'done') return `/projects/${p.id}/viewer`;
    if (p.latest_task_id && ['pending', 'running'].includes(p.latest_task_status)) return `/projects/${p.id}/progress/${p.latest_task_id}`;
    return `/projects/${p.id}`;
  };
  const cta = (p) => (p.latest_task_status === 'done' ? 'Walk in' : ['pending', 'running'].includes(p.latest_task_status) ? 'See progress' : p.has_video ? 'Start 3D scan' : 'Add a video');

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Your rooms</h1>
        <p className="muted">Each room is a project: one video in, one 3D space out.</p>
      </div>

      <form className="card inline-form" onSubmit={create}>
        <label className="grow">Name a new room
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Living room, May" maxLength={200} />
        </label>
        <button className="btn btn-primary" disabled={busy || !name.trim()}>{busy ? <Spinner /> : 'Create room'}</button>
      </form>

      <ErrorNote error={error} onRetry={load} />

      {!list && !error && <p><Spinner /> Loading your rooms…</p>}
      {list && list.length === 0 && (
        <div className="empty card">
          <h2>Nothing here yet</h2>
          <p>Name your first room above, then upload a slow 2–5 minute video of it.</p>
        </div>
      )}
      <ul className="project-grid">
        {list?.map((p) => (
          <li key={p.id} className="card project">
            <div className="project-top">
              <h2><Link to={`/projects/${p.id}`}>{p.name}</Link></h2>
              <StatusBadge status={p.latest_task_status} />
            </div>
            <p className="muted small">
              {p.video ? `${p.video.filename} · ${Math.round(p.video.duration_s)} s` : 'No video yet'} · updated {timeAgo(p.updated_at)}
            </p>
            <div className="row">
              <Link className="btn btn-primary" to={target(p)}>{cta(p)}</Link>
              <button className="btn btn-quiet" onClick={() => remove(p)} aria-label={`Delete ${p.name}`}>Delete</button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
