import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { projects as api } from '../api/endpoints.js';
import { ErrorNote, Spinner } from '../components/ui.jsx';
import { EditPanel } from '../edit/EditPanel.jsx';

// three.js + the splat renderer are big: only load them on this page
const SplatViewer = lazy(() => import('../viewer/SplatViewer.jsx').then((m) => ({ default: m.SplatViewer })));

export function ViewerPage() {
  const { id } = useParams();
  const [params, setParams] = useSearchParams();
  const viewerRef = useRef(null);
  const [scene, setScene] = useState(null);
  const [version, setVersion] = useState('original');
  const [versions, setVersions] = useState([{ version: 'original' }]);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('orbit');
  const [speed, setSpeed] = useState(1);
  const [pointsOnly, setPointsOnly] = useState(false);
  const [editing, setEditing] = useState(Boolean(params.get('edit')));
  const [pick, setPick] = useState(null);
  const [touring, setTouring] = useState(false);

  const editId = params.get('edit');
  const setEditId = useCallback((v) => {
    setParams((p) => { const n = new URLSearchParams(p); if (v) n.set('edit', v); else n.delete('edit'); return n; }, { replace: true });
  }, [setParams]);

  const loadVersions = useCallback(() => api.sceneVersions(id).then(setVersions).catch(() => {}), [id]);
  useEffect(() => { loadVersions(); }, [loadVersions]);

  // (re)load the scene description whenever the version changes
  useEffect(() => {
    let cancelled = false;
    api.scene(id, version).then((s) => { if (!cancelled) { setScene(s); setError(null); } }).catch((e) => { if (!cancelled) setError(e); });
    return () => { cancelled = true; };
  }, [id, version]);

  // signed file URLs last 6 h; refresh the description a little before they expire
  useEffect(() => {
    if (!scene?.urls_expire_at) return undefined;
    const ms = new Date(scene.urls_expire_at).getTime() - Date.now() - 5 * 60 * 1000;
    if (ms <= 0) return undefined;
    const t = setTimeout(() => api.scene(id, version).then(setScene).catch(() => {}), ms);
    return () => clearTimeout(t);
  }, [scene?.urls_expire_at, id, version]);

  const onResult = useCallback(async (v) => { await loadVersions(); setVersion(v); }, [loadVersions]);

  const tour = async () => {
    if (touring) { viewerRef.current?.stopCameraPath(); return; }
    setTouring(true);
    await viewerRef.current?.playCameraPath(scene.cameras, { duration: Math.max(8000, scene.cameras.length * 350) });
    setTouring(false);
  };

  if (error) {
    const inProgress = error.status === 404;
    return (
      <div className="stack narrow">
        <div className="page-head"><p className="crumbs"><Link to="/projects">Your rooms</Link> /</p><h1>This room isn’t ready</h1></div>
        <ErrorNote error={error} />
        {inProgress && <p><Link className="btn btn-primary" to={`/projects/${id}`}>Go to the room page</Link></p>}
      </div>
    );
  }
  if (!scene) return <p><Spinner /> Opening the room…</p>;

  const hasAfter = versions.length > 1;
  const meta = scene.meta || {};

  return (
    <div className={`viewer-page ${editing ? 'with-panel' : ''}`}>
      <div className="viewer-col">
        <div className="viewer-toolbar">
          <Link to={`/projects/${id}`} className="btn btn-quiet">← Room</Link>

          <div className="seg" role="group" aria-label="Camera mode">
            <button type="button" className={mode === 'orbit' ? 'on' : ''} aria-pressed={mode === 'orbit'} onClick={() => setMode('orbit')}>Orbit</button>
            <button type="button" className={mode === 'fly' ? 'on' : ''} aria-pressed={mode === 'fly'} onClick={() => setMode('fly')}>Fly (WASD)</button>
          </div>

          <label className="speed">Speed
            <input type="range" min="0.25" max="4" step="0.25" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Fly speed" />
            <span>{speed}×</span>
          </label>

          <button type="button" className="btn btn-quiet" onClick={() => viewerRef.current?.resetView()}>Reset view</button>
          <button type="button" className="btn btn-quiet" onClick={tour} disabled={!scene.cameras?.length}>{touring ? 'Stop tour' : 'Replay the walk-through'}</button>

          {hasAfter && (
            <div className="seg" role="group" aria-label="Before and after">
              {versions.map((v) => (
                <button key={v.version} type="button" className={version === v.version ? 'on' : ''} aria-pressed={version === v.version} onClick={() => setVersion(v.version)} title={v.prompt || undefined}>
                  {v.version === 'original' ? 'Before' : versions.filter((x) => x.version !== 'original').length > 1 ? `After: ${(v.prompt || '').slice(0, 18)}` : 'After'}
                </button>
              ))}
            </div>
          )}

          <span className="grow" />
          <label className="check compact"><input type="checkbox" checked={pointsOnly} onChange={(e) => setPointsOnly(e.target.checked)} /> Point cloud</label>
          <button type="button" className={`btn ${editing ? 'btn-primary' : 'btn-accent'}`} aria-pressed={editing} onClick={() => setEditing(!editing)}>
            {editing ? 'Editing…' : 'Edit object'}
          </button>
        </div>

        <div className="viewer-frame">
          <Suspense fallback={<div className="viewer-overlay"><Spinner /> Loading the 3D engine…</div>}>
            <SplatViewer ref={viewerRef} scene={scene} mode={mode} speed={speed} pointsOnly={pointsOnly} onPick={editing ? setPick : undefined} />
          </Suspense>
          <div className="viewer-hint" aria-hidden="true">
            {mode === 'fly' ? 'Drag to look · W A S D to move · Q / E down / up · Shift = faster' : 'Drag to orbit · scroll to zoom · right-drag to pan'}
            {editing && ' · click an object to select it'}
          </div>
        </div>
        <p className="muted small viewer-meta">
          {meta.num_gaussians ? `${meta.num_gaussians.toLocaleString()} gaussians` : ''}{meta.num_cameras ? ` · ${meta.num_cameras} photos` : ''}{meta.heldout?.psnr ? ` · PSNR ${meta.heldout.psnr}` : ''}
        </p>
      </div>

      {editing && (
        <EditPanel
          projectId={id} cameras={scene.cameras || []} viewerRef={viewerRef} pick={pick}
          editId={editId} setEditId={setEditId} onResult={onResult} onClose={() => setEditing(false)}
        />
      )}
    </div>
  );
}
