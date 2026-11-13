import { isActive } from './taskText.js';

// Edit status comes from the *latest task* (backend/app/edits.py), so after "mark a missed frame"
// (a segment task) the status says mask_ready again even though we are in the tracking step.
// The wizard therefore decides where it is from the most advanced task type that exists.
const RANK = { segment: 1, track: 2, edit_preview: 3, edit_apply: 4, rebuild: 5 };
const PHASES = ['mask', 'mask', 'track', 'preview', 'apply', 'rebuild'];
export const STEPS = [
  { key: 'pick', label: 'Pick' }, { key: 'mask', label: 'Mask' }, { key: 'track', label: 'Track' },
  { key: 'preview', label: 'Preview' }, { key: 'apply', label: 'Edit' }, { key: 'rebuild', label: 'Rebuild' },
];

export function editPhase(edit) {
  if (!edit) return 'pick';
  const types = (edit.tasks || []).map((t) => t.type);
  if (edit.current_task) types.push(edit.current_task.type);
  const top = types.reduce((m, t) => Math.max(m, RANK[t] || 0), 0);
  return PHASES[top];
}

/** Index into STEPS for the stepper; once every frame is edited we are really at "Rebuild". */
export function stepIndex(edit) {
  const phase = editPhase(edit);
  if (phase === 'apply' && edit?.status === 'edited') return 5;
  return STEPS.findIndex((s) => s.key === phase);
}

const BUSY_STATUS = ['created', 'segmenting', 'tracking', 'previewing', 'editing', 'rebuilding'];
export const isEditBusy = (edit) => Boolean(edit) && (BUSY_STATUS.includes(edit.status) || isActive(edit.current_task?.status));
export const isEditFailed = (edit) => Boolean(edit) && /_(failed|cancelled)$/.test(edit.status);
