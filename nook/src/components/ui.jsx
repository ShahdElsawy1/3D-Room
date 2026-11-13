import { useEffect } from 'react';
import { currentStageIndex, describeTask } from '../lib/taskText.js';

export function Spinner({ label = 'Loading' }) {
  return <span className="spinner" role="status" aria-label={label} />;
}

export function ErrorNote({ error, onRetry }) {
  if (!error) return null;
  const text = typeof error === 'string' ? error : error.message;
  return (
    <div className="note note-bad" role="alert">
      <span>{text}</span>
      {onRetry && <button type="button" className="btn btn-quiet" onClick={onRetry}>Try again</button>}
    </div>
  );
}

const BADGE = {
  pending: ['Waiting', 'blue'], running: ['Working', 'blue'], done: ['Ready', 'green'],
  failed: ['Failed', 'red'], cancelled: ['Cancelled', 'grey'],
};
export function StatusBadge({ status }) {
  if (!status) return <span className="badge badge-grey">No scan yet</span>;
  const [label, tone] = BADGE[status] || [status, 'grey'];
  return <span className={`badge badge-${tone}`}>{label}</span>;
}

/**
 * Real progress, drawn as the backend's stage segments (extracting frames → estimating poses →
 * training → exporting). Never a fake timer: the numbers come from GET /tasks/{id}.
 */
export function StageBar({ task }) {
  const info = describeTask(task);
  const stages = task?.stages?.length ? task.stages : null;
  const current = currentStageIndex(task);
  const pct = Math.round(task?.progress ?? 0);
  return (
    <div className={`stagebar tone-${info.tone}`}>
      {stages ? (
        <ol className="stage-steps">
          {stages.map((s, i) => {
            const state = i < current ? 'past' : i === current ? 'now' : 'next';
            const fill = i < current ? 100 : i === current ? Math.max(0, Math.min(100, ((task.progress - s.start) / (s.end - s.start)) * 100)) : 0;
            return (
              <li key={s.name} className={`stage stage-${state}`} aria-current={state === 'now' ? 'step' : undefined}>
                <span className="stage-label">{s.name}</span>
                <span className="stage-track"><span style={{ width: `${fill}%` }} /></span>
              </li>
            );
          })}
        </ol>
      ) : (
        <div className="stage-track single"><span style={{ width: `${pct}%` }} /></div>
      )}
      <div className="stage-caption">
        <strong>{info.title}</strong>
        {info.showBar && <span className="pct">{pct}%</span>}
      </div>
      {info.detail && <p className="stage-detail">{info.detail}</p>}
    </div>
  );
}

/** Compact bar for edit steps (backend gives progress + a stage text like "editing frame 4 of 61"). */
export function MiniProgress({ task, fallback }) {
  const pct = Math.round(task?.progress ?? 0);
  const queued = task?.status === 'pending';
  return (
    <div className="miniprogress" role="status">
      <div className="stage-track single"><span style={{ width: `${queued ? 3 : pct}%` }} /></div>
      <div className="stage-caption">
        <span>{queued ? 'Waiting for a GPU worker…' : (task?.stage || fallback || 'Working…')}</span>
        {!queued && <span className="pct">{pct}%</span>}
      </div>
    </div>
  );
}

export function Modal({ title, onClose, children }) {
  useEffect(() => {
    const h = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={title}>
        <h2>{title}</h2>
        {children}
      </div>
    </div>
  );
}
