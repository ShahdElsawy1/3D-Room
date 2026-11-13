import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { tasks as tasksApi } from '../api/endpoints.js';
import { usePolling } from '../hooks/usePolling.js';
import { ErrorNote, StageBar } from '../components/ui.jsx';
import { isFinished } from '../lib/taskText.js';

export function ProgressPage() {
  const { id, taskId } = useParams();
  const [cancelErr, setCancelErr] = useState(null);
  const { data: task, error, refresh } = usePolling(() => tasksApi.get(taskId), {
    interval: 2500, stopWhen: (t) => isFinished(t.status), deps: [taskId],
  });

  const cancel = async () => {
    if (!window.confirm('Stop this scan?')) return;
    try { await tasksApi.cancel(taskId); refresh(); } catch (e) { setCancelErr(e); }
  };

  const done = task?.status === 'done';
  const failed = task?.status === 'failed';
  const active = task && !isFinished(task.status);

  return (
    <div className="stack narrow">
      <div className="page-head">
        <p className="crumbs"><Link to="/projects">Your rooms</Link> / <Link to={`/projects/${id}`}>Room</Link> /</p>
        <h1>{done ? 'Your room is ready' : failed ? 'The scan hit a snag' : 'Building your room'}</h1>
        {active && <p className="muted">This takes a while. You can leave this page; the scan keeps going.</p>}
      </div>

      {error && !task && <ErrorNote error={error} onRetry={refresh} />}
      {error && task && <div className="note note-warn">Lost touch with the server for a moment. Still trying…</div>}
      <ErrorNote error={cancelErr} />

      {task && (
        <section className="card">
          <StageBar task={task} />
          <div className="row">
            {done && <Link className="btn btn-primary big" to={`/projects/${id}/viewer`}>Open the viewer</Link>}
            {failed && <Link className="btn btn-primary" to={`/projects/${id}`}>Try again</Link>}
            {active && <button className="btn btn-quiet" onClick={cancel}>Cancel scan</button>}
          </div>
        </section>
      )}
      {!task && !error && <p className="muted">Checking on your scan…</p>}
    </div>
  );
}
