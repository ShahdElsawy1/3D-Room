import { describeTask, currentStageIndex, describeEditStatus } from '../../src/lib/taskText.js';
import { editPhase, isEditBusy, isEditFailed, stepIndex } from '../../src/lib/editFlow.js';
import { clientToFrame, frameToPercent } from '../../src/lib/clicks.js';
import { maskToOverlay } from '../../src/lib/mask.js';
import { formatBytes, formatDuration, timeAgo } from '../../src/lib/format.js';

const stages = [
  { name: 'extracting frames', start: 0, end: 15 }, { name: 'estimating poses', start: 15, end: 45 },
  { name: 'training', start: 45, end: 90 }, { name: 'exporting', start: 90, end: 100 },
];

describe('describeTask (status table from API.md)', () => {
  it('pending/queued shows the queue position and no bar', () => {
    const d = describeTask({ status: 'pending', stage: 'queued', queue_position: 2 });
    expect(d.title).toMatch(/waiting for a gpu worker/i); expect(d.detail).toMatch(/number 2/); expect(d.showBar).toBe(false);
  });
  it('pending/queued (retry) shows why the last attempt failed', () => {
    const d = describeTask({ status: 'pending', stage: 'queued (retry)', error: 'CUDA out of memory' });
    expect(d.title).toMatch(/again/i); expect(d.detail).toMatch(/CUDA out of memory/);
  });
  it('running shows the stage; uploading results is its own message', () => {
    expect(describeTask({ status: 'running', stage: 'training' }).title).toBe('Training');
    expect(describeTask({ status: 'running', stage: 'uploading results' }).title).toBe('Saving your room');
  });
  it('does NOT show a stale error while running (it may be a retried attempt)', () => {
    expect(describeTask({ status: 'running', stage: 'training', error: 'old failure' }).detail).toBe('');
  });
  it('failed shows the error; done and cancelled are handled', () => {
    expect(describeTask({ status: 'failed', error: 'boom' })).toMatchObject({ tone: 'bad', detail: 'boom' });
    expect(describeTask({ status: 'done' }).tone).toBe('ok');
    expect(describeTask({ status: 'cancelled' }).title).toBe('Cancelled');
  });
});

describe('currentStageIndex', () => {
  it('finds the stage by name, falls back to progress, and is past-the-end when done', () => {
    expect(currentStageIndex({ stages, stage: 'training', progress: 50, status: 'running' })).toBe(2);
    expect(currentStageIndex({ stages, stage: 'queued', progress: 20, status: 'running' })).toBe(1);
    expect(currentStageIndex({ stages, status: 'done', progress: 100 })).toBe(4);
    expect(currentStageIndex({ status: 'running' })).toBe(-1);
  });
});

describe('edit wizard phase (status comes from the latest task, so use the task list)', () => {
  const t = (type, status = 'done') => ({ type, status, stage: 'x', progress: 100 });
  it('no edit → pick', () => { expect(editPhase(null)).toBe('pick'); });
  it('segment → mask, track → track, preview → preview, apply → apply, rebuild → rebuild', () => {
    expect(editPhase({ tasks: [t('segment')], current_task: t('segment') })).toBe('mask');
    expect(editPhase({ tasks: [t('segment'), t('track')], current_task: t('track') })).toBe('track');
    expect(editPhase({ tasks: [t('segment'), t('track'), t('edit_preview')], current_task: t('edit_preview') })).toBe('preview');
    expect(editPhase({ tasks: [t('edit_apply')], current_task: t('edit_apply') })).toBe('apply');
    expect(editPhase({ tasks: [t('rebuild')], current_task: t('rebuild') })).toBe('rebuild');
  });
  it('marking a missed frame (a new segment task) stays in the tracking step', () => {
    const e = { status: 'mask_ready', tasks: [t('segment'), t('track'), t('segment')], current_task: t('segment') };
    expect(editPhase(e)).toBe('track');
  });
  it('regenerating the preview stays in preview', () => {
    expect(editPhase({ tasks: [t('segment'), t('track'), t('edit_preview'), t('edit_preview')], current_task: t('edit_preview') })).toBe('preview');
  });
  it('a failed track still shows the tracking step (so it can be retried)', () => {
    expect(editPhase({ status: 'track_failed', tasks: [t('segment'), t('track', 'failed')], current_task: t('track', 'failed') })).toBe('track');
  });
  it('stepIndex moves to Rebuild once every frame is edited', () => {
    expect(stepIndex({ status: 'edited', tasks: [t('edit_apply')], current_task: t('edit_apply') })).toBe(5);
    expect(stepIndex({ status: 'editing', tasks: [t('edit_apply', 'running')], current_task: t('edit_apply', 'running') })).toBe(4);
  });
  it('busy / failed detection', () => {
    expect(isEditBusy({ status: 'tracking', current_task: { status: 'running' } })).toBe(true);
    expect(isEditBusy({ status: 'tracked', current_task: { status: 'done' } })).toBe(false);
    expect(isEditFailed({ status: 'edit_apply_failed' })).toBe(true);
    expect(isEditFailed({ status: 'edit_apply_cancelled' })).toBe(true);
    expect(isEditFailed({ status: 'edited' })).toBe(false);
  });
  it('describeEditStatus surfaces the task error on failure and the stage text while editing', () => {
    expect(describeEditStatus({ status: 'track_failed', current_task: { error: 'no GPU' } })).toMatch(/no GPU/);
    expect(describeEditStatus({ status: 'editing', current_task: { stage: 'editing frame 4 of 61' } })).toBe('editing frame 4 of 61');
  });
});

describe('click coordinates', () => {
  const rect = { left: 100, top: 50, width: 480, height: 270 }; // frame shown at half size
  it('maps a click on the displayed image to the frame’s own pixels', () => {
    expect(clientToFrame(rect, 960, 540, 340, 185)).toEqual({ x: 480, y: 270 });
    expect(clientToFrame(rect, 960, 540, 100, 50)).toEqual({ x: 0, y: 0 });
  });
  it('clamps to the frame (the backend rejects clicks outside it)', () => {
    expect(clientToFrame(rect, 960, 540, 9999, 9999)).toEqual({ x: 959, y: 539 });
    expect(clientToFrame(rect, 960, 540, -50, -50)).toEqual({ x: 0, y: 0 });
  });
  it('frameToPercent places the dot back where it was clicked', () => {
    expect(frameToPercent(480, 270, 960, 540)).toEqual({ left: '50%', top: '50%' });
  });
});

describe('maskToOverlay', () => {
  const px = (r, g, b, a) => [r, g, b, a];
  it('white-on-black (no alpha) → yellow where white', () => {
    const out = maskToOverlay(new Uint8ClampedArray([...px(255, 255, 255, 255), ...px(0, 0, 0, 255)]));
    expect(out[3]).toBeGreaterThan(0); expect(out[7]).toBe(0); expect([out[0], out[1], out[2]]).toEqual([245, 184, 61]);
  });
  it('white-on-transparent → overlay where opaque', () => {
    const out = maskToOverlay(new Uint8ClampedArray([...px(255, 255, 255, 255), ...px(0, 0, 0, 0)]));
    expect(out[3]).toBeGreaterThan(0); expect(out[7]).toBe(0);
  });
});

describe('format', () => {
  it('bytes, durations, relative time', () => {
    expect(formatBytes(33731776)).toBe('32.2 MB'); expect(formatBytes(512)).toBe('512 B');
    expect(formatDuration(182.4)).toBe('3 min 02 s'); expect(formatDuration(45)).toBe('45 s');
    const now = Date.parse('2026-11-07T12:00:00Z');
    expect(timeAgo('2026-11-07T11:30:00Z', now)).toBe('30 min ago'); expect(timeAgo('2026-11-05T12:00:00Z', now)).toBe('2 d ago');
  });
});
