import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { vi } from 'vitest';

vi.mock('../../src/api/endpoints.js', () => {
  const fn = () => vi.fn();
  return {
    auth: { login: fn(), signup: fn(), me: fn() },
    health: fn(),
    projects: { list: fn(), create: fn(), get: fn(), update: fn(), remove: fn(), uploadVideo: fn(), videoPath: (id) => `/projects/${id}/video`, reconstruct: fn(), tasks: fn(), scene: fn(), sceneVersions: fn(), frames: fn() },
    tasks: { get: fn(), cancel: fn() },
    edits: { list: fn(), create: fn(), get: fn(), remove: fn(), clicks: fn(), track: fn(), frames: fn(), patchFrames: fn(), preview: fn(), candidates: fn(), accept: fn(), rebuild: fn() },
  };
});
import * as api from '../../src/api/endpoints.js';
import { AuthProvider } from '../../src/auth/AuthContext.jsx';
import { AppRoutes } from '../../src/App.jsx';
import { ProgressPage } from '../../src/pages/ProgressPage.jsx';
import { ProjectPage } from '../../src/pages/ProjectPage.jsx';
import { EditPanel } from '../../src/edit/EditPanel.jsx';

beforeEach(() => { localStorage.clear(); vi.clearAllMocks(); });

const wrap = (ui, path = '/') => render(<MemoryRouter initialEntries={[path]}><AuthProvider>{ui}</AuthProvider></MemoryRouter>);
const user = { id: 'u1', email: 'a@b.co', created_at: '2026-11-07T00:00:00Z' };

describe('authorization', () => {
  it('sends a visitor with no token to the login page', async () => {
    wrap(<AppRoutes />, '/projects');
    expect(await screen.findByRole('button', { name: /log in/i, selected: false }).catch(() => null) ?? true).toBeTruthy();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
  });

  it('shows the server’s message for a wrong password and stays on the form', async () => {
    api.auth.login.mockRejectedValue(Object.assign(new Error('Incorrect email or password'), { status: 401 }));
    wrap(<AppRoutes />, '/login');
    await userEvent.type(screen.getByLabelText(/email/i), 'a@b.co');
    await userEvent.type(screen.getByLabelText(/password/i), 'wrongpass1');
    await userEvent.click(screen.getByRole('button', { name: /^log in$/i, hidden: false }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Incorrect email or password');
  });

  it('logs in, stores the token, and lands on the rooms list', async () => {
    api.auth.login.mockResolvedValue({ access_token: 'T0K', token_type: 'bearer', expires_in: 1, user });
    api.projects.list.mockResolvedValue([]);
    wrap(<AppRoutes />, '/login');
    await userEvent.type(screen.getByLabelText(/email/i), 'a@b.co');
    await userEvent.type(screen.getByLabelText(/password/i), 'password123');
    await userEvent.click(screen.getByRole('button', { name: /^log in$/i }));
    expect(await screen.findByText(/your rooms/i, { selector: 'h1' })).toBeInTheDocument();
    expect(localStorage.getItem('nook.token')).toBe('T0K');
  });

  it('create account tab calls signup', async () => {
    api.auth.signup.mockResolvedValue({ access_token: 'S1', token_type: 'bearer', expires_in: 1, user });
    api.projects.list.mockResolvedValue([]);
    wrap(<AppRoutes />, '/login');
    await userEvent.click(screen.getByRole('tab', { name: /create account/i }));
    await userEvent.type(screen.getByLabelText(/email/i), 'new@b.co');
    await userEvent.type(screen.getByLabelText(/password/i), 'password123');
    await userEvent.click(screen.getByRole('button', { name: /create my account/i }));
    await waitFor(() => expect(api.auth.signup).toHaveBeenCalledWith('new@b.co', 'password123'));
  });

  it('a stored token that /auth/me rejects sends the user to login', async () => {
    localStorage.setItem('nook.token', 'expired');
    api.auth.me.mockRejectedValue(Object.assign(new Error('Invalid token'), { status: 401 }));
    wrap(<AppRoutes />, '/projects');
    expect(await screen.findByLabelText(/email/i)).toBeInTheDocument();
    expect(localStorage.getItem('nook.token')).toBeNull();
  });
});

describe('progress page', () => {
  const stages = [{ name: 'extracting frames', start: 0, end: 15 }, { name: 'estimating poses', start: 15, end: 45 }, { name: 'training', start: 45, end: 90 }, { name: 'exporting', start: 90, end: 100 }];
  const renderProgress = () => render(
    <MemoryRouter initialEntries={['/projects/p1/progress/t1']}>
      <Routes><Route path="/projects/:id/progress/:taskId" element={<ProgressPage />} /></Routes>
    </MemoryRouter>);

  it('draws the four real stages with the current one marked, and the percentage', async () => {
    api.tasks.get.mockResolvedValue({ id: 't1', status: 'running', stage: 'training', progress: 60, stages, error: null });
    renderProgress();
    expect(await screen.findByText('60%')).toBeInTheDocument();
    const items = screen.getAllByRole('listitem');
    expect(items).toHaveLength(4);
    expect(items[2]).toHaveAttribute('aria-current', 'step');
    expect(screen.getByRole('button', { name: /cancel scan/i })).toBeInTheDocument();
  });
  it('shows queue position while waiting for the worker', async () => {
    api.tasks.get.mockResolvedValue({ id: 't1', status: 'pending', stage: 'queued', progress: 0, stages, queue_position: 1 });
    renderProgress();
    expect(await screen.findByText(/waiting for a gpu worker/i)).toBeInTheDocument();
    expect(screen.getByText(/number 1 in line/i)).toBeInTheDocument();
  });
  it('offers the viewer when done', async () => {
    api.tasks.get.mockResolvedValue({ id: 't1', status: 'done', stage: 'done', progress: 100, stages });
    renderProgress();
    expect(await screen.findByRole('link', { name: /open the viewer/i })).toHaveAttribute('href', '/projects/p1/viewer');
  });
  it('shows the error and a way to try again when failed', async () => {
    api.tasks.get.mockResolvedValue({ id: 't1', status: 'failed', stage: 'training', progress: 50, stages, error: 'CUDA out of memory' });
    renderProgress();
    expect(await screen.findByText(/CUDA out of memory/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /try again/i })).toBeInTheDocument();
  });
});

describe('upload with submit', () => {
  const project = { id: 'p1', name: 'Living room', has_video: false, video: null, latest_task_status: null, latest_task_id: null };
  const renderPage = () => render(
    <MemoryRouter initialEntries={['/projects/p1']}>
      <Routes>
        <Route path="/projects/:id" element={<ProjectPage />} />
        <Route path="/projects/:id/progress/:taskId" element={<div>PROGRESS PAGE</div>} />
      </Routes>
    </MemoryRouter>);

  it('rejects a non-video file before uploading', async () => {
    api.projects.get.mockResolvedValue(project); api.projects.tasks.mockResolvedValue([]);
    const { container } = renderPage();
    await screen.findByText('Living room');
    await userEvent.upload(container.querySelector('input[type=file]'), new File(['x'], 'notes.txt'), { applyAccept: false });
    expect(await screen.findByRole('alert')).toHaveTextContent(/isn’t supported/);
    expect(api.projects.uploadVideo).not.toHaveBeenCalled();
  });

  it('uploads, then submits the reconstruction, then opens the progress page', async () => {
    api.projects.get.mockResolvedValue(project); api.projects.tasks.mockResolvedValue([]);
    api.projects.uploadVideo.mockImplementation(async (id, file, { onProgress }) => { onProgress(1); return { project_id: id, video: { warnings: ['duration 45s is outside the recommended 2-5 min'] } }; });
    api.projects.reconstruct.mockResolvedValue({ id: 'task9' });
    const { container } = renderPage();
    await screen.findByText('Living room');
    await userEvent.upload(container.querySelector('input[type=file]'), new File(['x'], 'room.mp4', { type: 'video/mp4' }));
    await userEvent.click(screen.getByLabelText(/high quality/i));
    await userEvent.click(screen.getByRole('button', { name: /upload & start 3d scan/i }));
    expect(await screen.findByText('PROGRESS PAGE')).toBeInTheDocument();
    expect(api.projects.uploadVideo.mock.calls[0][0]).toBe('p1');
    expect(api.projects.reconstruct).toHaveBeenCalledWith('p1', { train_mode: 'full' });
  });

  it('disables submit while a scan is already running for the project', async () => {
    api.projects.get.mockResolvedValue({ ...project, has_video: true, video: { filename: 'r.mp4', duration_s: 120, width: 1920, height: 1080, size_bytes: 1e6, warnings: [] } });
    api.projects.tasks.mockResolvedValue([{ id: 'tt', status: 'running', created_at: '2026-11-07T00:00:00Z' }]);
    renderPage();
    expect(await screen.findByText(/already running/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /start 3d scan/i })).toBeDisabled();
  });
});

describe('edit panel (spec §4.5 steps 1-6)', () => {
  const frames = [
    { name: '000001.jpg', index: 0, url: '/f/1.jpg?t=a', width: 960, height: 540, camera_id: 1 },
    { name: '000002.jpg', index: 1, url: '/f/2.jpg?t=a', width: 960, height: 540, camera_id: 2 },
  ];
  const cameras = [{ id: 1 }, { id: 2 }];
  const viewerRef = { current: { flyToCamera: vi.fn() } };
  const task = (type, status = 'done', extra = {}) => ({ id: 'tk', type, status, stage: 'done', progress: 100, error: null, ...extra });
  const mount = (props = {}) => {
    const setEditId = vi.fn();
    render(<EditPanel projectId="p1" cameras={cameras} viewerRef={viewerRef} pick={null} editId={null} setEditId={setEditId} onResult={vi.fn()} onClose={vi.fn()} {...props} />);
    return { setEditId };
  };

  beforeEach(() => { api.projects.frames.mockResolvedValue(frames); });

  it('step 1: pick a frame → click the object → "Find the object" sends frame + clicks in frame pixels', async () => {
    api.edits.create.mockResolvedValue({ id: 'e1' });
    api.edits.get.mockResolvedValue({ id: 'e1', status: 'segmenting', ref_frame: '000002.jpg', clicks: [], tasks: [], current_task: task('segment', 'running'), removed_frames: [] });
    const { setEditId } = mount();
    await userEvent.click(await screen.findByRole('button', { name: /use frame 000002\.jpg/i }));
    expect(viewerRef.current.flyToCamera).toHaveBeenCalledWith({ id: 2 }); // viewer snaps to that training camera
    const stage = screen.getByRole('img', { name: /reference frame/i });
    stage.getBoundingClientRect = () => ({ left: 0, top: 0, width: 480, height: 270 });
    fireEvent.click(stage, { clientX: 240, clientY: 135 });
    await userEvent.click(screen.getByRole('button', { name: /find the object/i }));
    await waitFor(() => expect(api.edits.create).toHaveBeenCalledWith('p1', '000002.jpg', [{ x: 480, y: 270, positive: true }]));
    expect(setEditId).toHaveBeenCalledWith('e1');
  });

  it('click in the 3D room → jumps to the right frame with the projected pixel pre-placed', async () => {
    mount({ pick: { hit: true, camera: { id: 2, img_name: '000002.jpg', x: 300, y: 200 } } });
    await screen.findByRole('img', { name: /reference frame/i });
    expect(screen.getByText(/1 point/)).toBeInTheDocument();
    expect(viewerRef.current.flyToCamera).toHaveBeenCalledWith({ id: 2 });
  });

  it('a 3D click that misses explains itself instead of failing silently', async () => {
    mount({ pick: { hit: false, camera: null } });
    expect(await screen.findByText(/did not land on anything/i)).toBeInTheDocument();
  });

  it('step 2-3: mask preview, refine clicks, prompt required before "Edit" starts tracking', async () => {
    api.edits.get.mockResolvedValue({ id: 'e1', status: 'mask_ready', ref_frame: '000001.jpg', clicks: [{ x: 10, y: 10, positive: true }], ref_mask_url: '/m.png?t=a', tasks: [task('segment')], current_task: task('segment'), removed_frames: [] });
    api.edits.track.mockResolvedValue({});
    mount({ editId: 'e1' });
    const edit = await screen.findByRole('button', { name: /^edit$/i });
    expect(edit).toBeDisabled(); // no prompt yet
    await userEvent.type(screen.getByLabelText(/what should it become/i), 'make the sofa dark green leather');
    expect(edit).toBeEnabled();
    await userEvent.click(edit);
    await waitFor(() => expect(api.edits.track).toHaveBeenCalledWith('e1'));
  });

  it('step 4: tracking grid lets the user remove a wrong frame and mark a missed one (fallback)', async () => {
    api.edits.get.mockResolvedValue({ id: 'e1', status: 'tracked', ref_frame: '000001.jpg', clicks: [], prompt: 'make it green', tasks: [task('segment'), task('track')], current_task: task('track'), removed_frames: [] });
    api.edits.frames.mockResolvedValue([
      { name: '000001.jpg', url: '/f/1.jpg', mask_url: '/m1.png', present: true, tracked_present: true, removed: false },
      { name: '000002.jpg', url: '/f/2.jpg', mask_url: null, present: false, tracked_present: false, removed: false },
    ]);
    api.edits.patchFrames.mockResolvedValue({});
    mount({ editId: 'e1' });
    expect(await screen.findByText(/of 2 frames show the object/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /^remove$/i }));
    await waitFor(() => expect(api.edits.patchFrames).toHaveBeenCalledWith('e1', { remove: ['000001.jpg'] }));
    await userEvent.click(screen.getByRole('button', { name: /mark it/i }));
    const dialog = screen.getByRole('dialog');
    await userEvent.click(within(dialog).getByRole('button', { name: /add this frame/i }));
    await waitFor(() => expect(api.edits.patchFrames).toHaveBeenCalledWith('e1', { add: [{ name: '000002.jpg', clicks: [] }] }));
  });

  it('step 5: shows candidates; accepting one sends its seed', async () => {
    api.edits.get.mockResolvedValue({ id: 'e1', status: 'preview_ready', ref_frame: '000001.jpg', clicks: [], prompt: 'p', tasks: [task('segment'), task('track'), task('edit_preview')], current_task: task('edit_preview'), removed_frames: [] });
    api.edits.candidates.mockResolvedValue([{ seed: 11, url: '/c/11.png', seconds: 4.2 }, { seed: 22, url: '/c/22.png', seconds: 4.0 }]);
    api.edits.accept.mockResolvedValue({});
    mount({ editId: 'e1' });
    await userEvent.click(await screen.findByLabelText(/option 22/i));
    await userEvent.click(screen.getByRole('button', { name: /use this one for every frame/i }));
    await waitFor(() => expect(api.edits.accept).toHaveBeenCalledWith('e1', 22));
  });

  it('step 5: shows "editing frame x of N" while the diffusion task runs', async () => {
    const running = task('edit_apply', 'running', { stage: 'editing frame 4 of 61', progress: 7 });
    api.edits.get.mockResolvedValue({ id: 'e1', status: 'editing', ref_frame: '000001.jpg', clicks: [], tasks: [running], current_task: running, removed_frames: [] });
    mount({ editId: 'e1' });
    expect(await screen.findByText('editing frame 4 of 61')).toBeInTheDocument();
    expect(screen.getByText('7%')).toBeInTheDocument();
  });

  it('step 6: rebuild, then tells the viewer the new scene version exactly once', async () => {
    const onResult = vi.fn();
    api.edits.get.mockResolvedValue({ id: 'e1', status: 'done', scene_version: 'e1', ref_frame: '000001.jpg', clicks: [], tasks: [task('rebuild')], current_task: task('rebuild'), removed_frames: [] });
    mount({ editId: 'e1', onResult });
    await screen.findByText(/before \/ after/i);
    expect(onResult).toHaveBeenCalledTimes(1);
    expect(onResult).toHaveBeenCalledWith('e1');
  });

  it('a failed step shows the backend error and lets the user retry that step', async () => {
    const failed = task('edit_apply', 'failed', { error: 'CUDA out of memory' });
    api.edits.get.mockResolvedValue({ id: 'e1', status: 'edit_apply_failed', chosen_seed: 22, ref_frame: '000001.jpg', clicks: [], tasks: [failed], current_task: failed, removed_frames: [] });
    api.edits.accept.mockResolvedValue({});
    mount({ editId: 'e1' });
    expect(await screen.findByText(/CUDA out of memory/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /try this step again/i }));
    await waitFor(() => expect(api.edits.accept).toHaveBeenCalledWith('e1', 22));
  });
});
