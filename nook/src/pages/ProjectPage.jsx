import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { projects as api } from '../api/endpoints.js';
import { fetchBlobUrl } from '../api/client.js';
import { ErrorNote, Spinner, StatusBadge } from '../components/ui.jsx';
import { formatBytes, formatDuration, timeAgo } from '../lib/format.js';
import { isActive } from '../lib/taskText.js';

export const VIDEO_EXT = ['.mp4', '.mov', '.m4v', '.mkv', '.webm', '.avi'];
export const hasVideoExt = (name) => VIDEO_EXT.some((e) => name.toLowerCase().endsWith(e));

/** "Upload with submit": pick a video → upload (real progress) → submit the 3D reconstruction task. */
export function ProjectPage() {
  const { id } = useParams();
  const nav = useNavigate();
  const [project, setProject] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [error, setError] = useState(null);
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null); // {progress, state}
  const [warnings, setWarnings] = useState([]);
  const [quality, setQuality] = useState(false);
  const [advanced, setAdvanced] = useState({ method: '', fps: '', long_side: '' });
  const [submitting, setSubmitting] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const abortRef = useRef(null);
  const inputRef = useRef(null);

  const load = useCallback(async () => {
    try {
      const [p, t] = await Promise.all([api.get(id), api.tasks(id)]);
      setProject(p); setTasks(t); setError(null);
    } catch (e) { setError(e); }
  }, [id]);
  useEffect(() => { load(); }, [load]);
  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  const busyTask = tasks.find((t) => isActive(t.status));

  const choose = (f) => {
    setError(null);
    if (!f) return;
    if (!hasVideoExt(f.name)) { setError(`That file type isn’t supported. Use one of: ${VIDEO_EXT.join(' ')}`); return; }
    setFile(f); setUpload(null);
  };

  const doUpload = async () => {
    abortRef.current = new AbortController();
    setUpload({ progress: 0, state: 'uploading' }); setError(null);
    try {
      const res = await api.uploadVideo(id, file, {
        signal: abortRef.current.signal,
        onProgress: (p) => setUpload({ progress: p, state: 'uploading' }),
      });
      setWarnings(res.video.warnings || []);
      setUpload({ progress: 1, state: 'done' });
      setFile(null);
      await load();
      return true;
    } catch (e) {
      setUpload(null);
      if (e.name !== 'AbortError') setError(e);
      return false;
    }
  };

  const submit = async () => {
    setSubmitting(true); setError(null);
    try {
      if (file && !(await doUpload())) return;
      const body = {};
      if (quality) body.train_mode = 'full';
      if (advanced.method) body.method = advanced.method;
      if (advanced.fps) body.fps = advanced.fps;
      if (advanced.long_side) body.long_side = Number(advanced.long_side);
      const task = await api.reconstruct(id, body);
      nav(`/projects/${id}/progress/${task.id}`);
    } catch (e) { setError(e); } finally { setSubmitting(false); }
  };

  const showVideo = async () => {
    try { setPreviewUrl(await fetchBlobUrl(api.videoPath(id))); } catch (e) { setError(e); }
  };

  if (!project && !error) return <p><Spinner /> Loading…</p>;
  if (!project) return <ErrorNote error={error} onRetry={load} />;

  const v = project.video;
  const canSubmit = (file || project.has_video) && !busyTask && !submitting;
  const scanned = project.latest_task_status === 'done';

  return (
    <div className="stack">
      <div className="page-head">
        <p className="crumbs"><Link to="/projects">Your rooms</Link> / </p>
        <h1>{project.name}</h1>
        <StatusBadge status={project.latest_task_status} />
      </div>

      <ErrorNote error={error} />

      {busyTask && (
        <div className="note note-info">
          A scan is already {busyTask.status === 'pending' ? 'waiting' : 'running'} for this room.{' '}
          <Link to={`/projects/${id}/progress/${busyTask.id}`}>See its progress</Link>
        </div>
      )}
      {scanned && (
        <div className="note note-good">
          This room is ready. <Link to={`/projects/${id}/viewer`}>Walk in</Link>. Uploading a new video and scanning again replaces it.
        </div>
      )}

      <section className="card">
        <h2>1. The video</h2>
        {v && !file && (
          <div className="videometa">
            <div>
              <strong>{v.filename}</strong>
              <p className="muted small">{formatDuration(v.duration_s)} · {v.width}×{v.height}{v.fps ? ` · ${Math.round(v.fps)} fps` : ''} · {formatBytes(v.size_bytes)}</p>
              {v.warnings?.map((w) => <p key={w} className="note note-warn">{w}</p>)}
            </div>
            {!previewUrl && <button className="btn btn-quiet" onClick={showVideo}>Preview</button>}
          </div>
        )}
        {previewUrl && <video className="preview" src={previewUrl} controls />}

        <div
          className={`drop ${file ? 'has-file' : ''}`}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); choose(e.dataTransfer.files?.[0]); }}
        >
          <input ref={inputRef} type="file" accept={VIDEO_EXT.join(',')} hidden onChange={(e) => choose(e.target.files?.[0])} />
          {file ? (
            <p><strong>{file.name}</strong> · {formatBytes(file.size)}</p>
          ) : (
            <p>{v ? 'Want a different take? ' : ''}Drop a video here, or</p>
          )}
          <button type="button" className="btn btn-quiet" onClick={() => inputRef.current?.click()}>{file ? 'Choose another' : 'Choose a file'}</button>
          <p className="muted small">One room, slow camera, good light, 2–5 minutes. Keep it under 100 MB if you use the Cloudflare tunnel.</p>
        </div>

        {upload && (
          <div className="miniprogress" role="status">
            <div className="stage-track single"><span style={{ width: `${Math.round(upload.progress * 100)}%` }} /></div>
            <div className="stage-caption"><span>{upload.state === 'done' ? 'Uploaded' : 'Uploading…'}</span><span className="pct">{Math.round(upload.progress * 100)}%</span></div>
          </div>
        )}
        {warnings.map((w) => <p key={w} className="note note-warn">{w}</p>)}
      </section>

      <section className="card">
        <h2>2. Make it 3D</h2>
        <label className="check">
          <input type="checkbox" checked={quality} onChange={(e) => setQuality(e.target.checked)} />
          <span>High quality <span className="muted">(about 30k training steps; a lot slower)</span></span>
        </label>
        <details className="advanced">
          <summary>Advanced options</summary>
          <div className="grid3">
            <label>Pose method
              <select value={advanced.method} onChange={(e) => setAdvanced({ ...advanced, method: e.target.value })}>
                <option value="">Default</option><option value="vggt">VGGT (fast)</option><option value="glomap">GLOMAP (accurate)</option><option value="both">Both (comparison)</option>
              </select>
            </label>
            <label>Frames per second
              <select value={advanced.fps} onChange={(e) => setAdvanced({ ...advanced, fps: e.target.value })}>
                <option value="">Auto</option>{['1', '2', '3', '4'].map((f) => <option key={f}>{f}</option>)}
              </select>
            </label>
            <label>Frame size (px)
              <input type="number" min="640" max="960" step="10" placeholder="960" value={advanced.long_side} onChange={(e) => setAdvanced({ ...advanced, long_side: e.target.value })} />
            </label>
          </div>
        </details>
        <button className="btn btn-primary big" onClick={submit} disabled={!canSubmit}>
          {submitting ? <><Spinner /> {upload?.state === 'uploading' ? 'Uploading…' : 'Submitting…'}</> : file ? 'Upload & start 3D scan' : 'Start 3D scan'}
        </button>
        {!file && !project.has_video && <p className="muted small">Add a video first.</p>}
      </section>

      {tasks.length > 0 && (
        <section className="card">
          <h2>Earlier scans</h2>
          <ul className="history">
            {tasks.map((t) => (
              <li key={t.id}>
                <StatusBadge status={t.status} />
                <span className="muted small">{timeAgo(t.created_at)}</span>
                <Link to={`/projects/${id}/progress/${t.id}`}>Details</Link>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
