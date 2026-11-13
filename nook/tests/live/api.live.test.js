// Runs the REAL api client + endpoint modules against a REAL running backend (no mocks).
//   BACKEND_URL=http://localhost:8000 BACKEND_WORKER_TOKEN_FILE=../backend/data/.worker_token npm run test:live
// Skipped automatically when the backend is not reachable.
import { execFileSync, execSync } from 'node:child_process';
import { existsSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { setApiBase, clearToken, setToken, getToken, absUrl } from '../../src/api/client.js';
import * as api from '../../src/api/endpoints.js';

const BASE = process.env.BACKEND_URL || 'http://localhost:8000';
const WTOKEN = process.env.BACKEND_WORKER_TOKEN_FILE || '/home/claude/bdata/.worker_token';
let up = false;
try { up = (await fetch(`${BASE}/health`)).ok; } catch { up = false; }

describe.skipIf(!up)('live backend contract', () => {
  const email = `live-${Date.now()}@example.com`;
  const pw = 'password123';
  let project; let seeded;

  beforeAll(() => { setApiBase(BASE); localStorage.clear(); });

  it('health + signup returns a token and user', async () => {
    expect((await api.health()).status).toBe('ok');
    const out = await api.auth.signup(email, pw);
    expect(out.access_token).toBeTruthy(); expect(out.user.email).toBe(email);
    setToken(out.access_token);
    expect((await api.auth.me()).email).toBe(email);
  });

  it('duplicate signup → 409, short password → 422 with a readable message, wrong password → 401', async () => {
    await expect(api.auth.signup(email, pw)).rejects.toMatchObject({ status: 409 });
    await expect(api.auth.signup('x@example.com', 'short')).rejects.toMatchObject({ status: 422, message: expect.stringMatching(/password/) });
    await expect(api.auth.login(email, 'nope-nope-1')).rejects.toMatchObject({ status: 401, message: 'Incorrect email or password' });
    const ok = await api.auth.login(email, pw); expect(ok.access_token).toBeTruthy();
  });

  it('a bad token is a 401 (the UI turns this into "go to login")', async () => {
    const good = getToken(); setToken('garbage');
    await expect(api.projects.list()).rejects.toMatchObject({ status: 401 });
    setToken(good);
  });

  it('create + list projects (own only)', async () => {
    project = await api.projects.create('Live test room');
    expect(project.id).toMatch(/^[0-9a-f]{32}$/); expect(project.has_video).toBe(false);
    expect((await api.projects.list()).map((p) => p.id)).toContain(project.id);
  });

  it('upload: rejects the wrong extension, accepts a real video via XHR with progress', async () => {
    await expect(api.projects.uploadVideo(project.id, new File(['hi'], 'notes.txt'))).rejects.toMatchObject({ status: 415 });
    const dir = mkdtempSync(path.join(tmpdir(), 'nook-'));
    const mp4 = path.join(dir, 'room.mp4');
    execFileSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc=duration=8:size=640x360:rate=10', '-pix_fmt', 'yuv420p', mp4]);
    const { readFileSync } = await import('node:fs');
    const file = new File([readFileSync(mp4)], 'room.mp4', { type: 'video/mp4' });
    const progress = [];
    const res = await api.projects.uploadVideo(project.id, file, { onProgress: (p) => progress.push(p) });
    expect(res.video.width).toBe(640); expect(res.video.duration_s).toBeGreaterThan(7);
    expect(res.video.warnings.join(' ')).toMatch(/recommended/);
    expect(progress.at(-1)).toBe(1);
    expect((await api.projects.get(project.id)).has_video).toBe(true);
  });

  it('reconstruct → pending/queued task with the four real stages; a 2nd submit is a 409; scene is 404 "in progress"', async () => {
    const t = await api.projects.reconstruct(project.id, { train_mode: 'quick' });
    expect(t.status).toBe('pending'); expect(t.stages.map((s) => s.name)).toEqual(['extracting frames', 'estimating poses', 'training', 'exporting']);
    const polled = await api.tasks.get(t.id);
    expect(polled.stage).toBe('queued'); expect(polled.queue_position).toBeGreaterThan(0);
    await expect(api.projects.reconstruct(project.id, {})).rejects.toMatchObject({ status: 409 });
    await expect(api.projects.scene(project.id)).rejects.toMatchObject({ status: 404, message: expect.stringMatching(/in progress/i) });
    await expect(api.projects.remove(project.id)).rejects.toMatchObject({ status: 409 }); // cannot delete while a task is pending
    const c = await api.tasks.cancel(t.id); expect(c.status).toBe('cancelled');
    expect((await api.projects.get(project.id)).latest_task_status).toBe('cancelled');
  });

  it('a different user cannot see this project (404)', async () => {
    const mine = getToken();
    const other = await api.auth.signup(`other-${Date.now()}@example.com`, pw); setToken(other.access_token);
    await expect(api.projects.get(project.id)).rejects.toMatchObject({ status: 404 });
    expect((await api.projects.list()).map((p) => p.id)).not.toContain(project.id);
    setToken(mine);
  });

  describe('a finished scene (synthetic room uploaded by tests/live/seed_scene.py acting as the worker)', () => {
    let scene; let frames;
    beforeAll(() => {
      if (!existsSync(WTOKEN)) throw new Error(`worker token file not found: ${WTOKEN}`);
      const out = execSync(`python3 tests/live/seed_scene.py --api ${BASE} --worker-token-file ${WTOKEN} --drain`, { encoding: 'utf8' });
      seeded = JSON.parse(out.trim().split('\n').pop());
      setToken(seeded.token);
    }, 120000);

    it('project shows latest_task_status done (the list links straight to the viewer)', async () => {
      const p = await api.projects.get(seeded.project_id);
      expect(p.latest_task_status).toBe('done'); expect(p.latest_task_id).toBe(seeded.task_id);
    });

    it('GET /scene has everything the viewer needs', async () => {
      scene = await api.projects.scene(seeded.project_id);
      expect(scene.up_vector).toEqual([0, -1, 0]);
      expect(scene.cameras).toHaveLength(24);
      expect(scene.cameras[0]).toMatchObject({ id: 1, img_name: '000001.jpg', width: 960, height: 540 });
      expect(scene.cameras[0].rotation).toHaveLength(3);
      expect(scene.meta.bbox_min).toBeDefined(); expect(scene.meta.num_gaussians).toBe(scene.splat.size_bytes / 32);
      expect(scene.points?.url).toBeTruthy(); expect(new Date(scene.urls_expire_at) > new Date()).toBe(true);
    });

    it('signed splat URL downloads WITHOUT an auth header, as 32-byte gaussians, with Range support', async () => {
      const url = absUrl(scene.splat.url);
      const res = await fetch(url); expect(res.status).toBe(200);
      const buf = new Uint8Array(await res.arrayBuffer());
      expect(buf.length).toBe(scene.splat.size_bytes); expect(buf.length % 32).toBe(0);
      const dv = new DataView(buf.buffer);
      expect(Number.isFinite(dv.getFloat32(0, true))).toBe(true); expect(buf[27]).toBe(255); // first gaussian alpha
      const part = await fetch(url, { headers: { Range: 'bytes=0-63' } });
      expect(part.status).toBe(206); expect((await part.arrayBuffer()).byteLength).toBe(64);
    });

    it('points.ply (the fallback) is a coloured PLY', async () => {
      const head = new TextDecoder().decode(new Uint8Array(await (await fetch(absUrl(scene.points.url))).arrayBuffer()).slice(0, 200));
      expect(head).toMatch(/^ply/); expect(head).toMatch(/property uchar red/);
    });

    it('a tampered file token is refused', async () => {
      const bad = absUrl(scene.splat.url).replace(/t=.{6}/, 't=XXXXXX');
      expect((await fetch(bad)).status).toBeGreaterThanOrEqual(401);
    });

    it('frames list (frame picker) matches the scene cameras and its images load via signed URL', async () => {
      frames = await api.projects.frames(seeded.project_id);
      expect(frames).toHaveLength(24);
      expect(frames[0]).toMatchObject({ name: '000001.jpg', camera_id: 1 });
      expect(scene.cameras.find((c) => c.id === frames[0].camera_id).img_name).toBe(frames[0].name);
      const img = await fetch(absUrl(frames[0].url)); expect(img.status).toBe(200); expect(img.headers.get('content-type')).toMatch(/image/);
    });

    it('Week 2: create an edit → segment task is queued; out-of-frame click is a 422; steps in the wrong order are 409', async () => {
      await expect(api.edits.create(seeded.project_id, '000001.jpg', [{ x: 5000, y: 5, positive: true }])).rejects.toMatchObject({ status: 422 });
      const e = await api.edits.create(seeded.project_id, '000001.jpg', [{ x: 480, y: 270, positive: true }]);
      expect(e.status).toBe('segmenting'); expect(e.current_task.type).toBe('segment'); expect(e.ref_frame).toBe('000001.jpg');
      await expect(api.edits.create(seeded.project_id, '000002.jpg', [{ x: 1, y: 1, positive: true }])).rejects.toMatchObject({ status: 409 }); // one task at a time
      await api.tasks.cancel(e.current_task.id);
      const after = await api.edits.get(e.id); expect(after.status).toBe('segment_cancelled');
      await expect(api.edits.track(e.id)).rejects.toMatchObject({ status: 409 }); // no mask yet
      expect((await api.edits.list(seeded.project_id)).map((x) => x.id)).toContain(e.id);
      await api.edits.remove(e.id);
      expect((await api.projects.sceneVersions(seeded.project_id)).map((v) => v.version)).toEqual(['original']);
    });
  });

  afterAll(() => clearToken());
});
