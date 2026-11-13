// Turns a backend Task into plain words + a tone. Follows the status/stage table in backend/docs/API.md.

export const ACTIVE = ['pending', 'running'];
export const isActive = (status) => ACTIVE.includes(status);
export const isFinished = (status) => ['done', 'failed', 'cancelled'].includes(status);

export function describeTask(task) {
  if (!task) return { tone: 'idle', title: 'Nothing running', detail: '', showBar: false };
  const { status, stage, error, queue_position: pos } = task;

  if (status === 'pending') {
    if (stage === 'queued (retry)') {
      return { tone: 'wait', title: 'Trying again', detail: error ? `The last attempt stopped: ${error}` : 'The last attempt stopped, so it is queued again.', showBar: false };
    }
    const place = pos ? ` You are number ${pos} in line.` : '';
    return { tone: 'wait', title: 'Waiting for a GPU worker', detail: `Nothing moves until the Kaggle worker is running.${place}`, showBar: false };
  }
  if (status === 'running') {
    if (stage === 'uploading results') return { tone: 'run', title: 'Saving your room', detail: 'Copying the finished files to the server.', showBar: true };
    return { tone: 'run', title: capitalize(stage || 'Working'), detail: '', showBar: true };
  }
  if (status === 'done') return { tone: 'ok', title: 'Done', detail: '', showBar: true };
  if (status === 'failed') return { tone: 'bad', title: 'This run failed', detail: error || 'No error message was reported.', showBar: false };
  if (status === 'cancelled') return { tone: 'idle', title: 'Cancelled', detail: '', showBar: false };
  return { tone: 'idle', title: status || 'Unknown', detail: '', showBar: false };
}

/** Which of the task's stage segments is the current one (-1 = none). Progress can drop on retry; that is fine. */
export function currentStageIndex(task) {
  if (!task?.stages?.length) return -1;
  if (task.status === 'done') return task.stages.length;
  const byName = task.stages.findIndex((s) => s.name === task.stage);
  if (byName >= 0) return byName;
  return task.stages.findIndex((s) => task.progress >= s.start && task.progress < s.end);
}

function capitalize(s) { return s ? s[0].toUpperCase() + s.slice(1) : s; }

/** Edit step statuses → one sentence. */
export function describeEditStatus(edit) {
  if (!edit) return '';
  const t = edit.current_task;
  if (/_failed$/.test(edit.status)) return t?.error ? `That step failed: ${t.error}` : 'That step failed.';
  if (/_cancelled$/.test(edit.status)) return 'That step was cancelled.';
  return {
    created: 'Starting…', segmenting: 'Finding the object in your frame…', mask_ready: 'Mask ready.',
    tracking: 'Following the object through the other frames…', tracked: 'Tracking finished.',
    previewing: 'Painting the first frame…', preview_ready: 'First frame ready.',
    editing: t?.stage || 'Editing the other frames…', edited: 'All frames edited.',
    rebuilding: 'Rebuilding the 3D room…', done: 'Finished.',
  }[edit.status] || edit.status;
}
