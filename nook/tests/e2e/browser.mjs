// Real-browser end-to-end test: real Chromium (software WebGL2) + real backend + built frontend.
//   1. npm run build && npm run preview      (http://127.0.0.1:5173)
//   2. backend running on :8000
//   3. node tests/e2e/browser.mjs
import chromium from '@sparticuz/chromium';
import { chromium as pw } from 'playwright-core';
import { execSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';

const APP = process.env.APP_URL || 'http://127.0.0.1:5173';
const API = process.env.BACKEND_URL || 'http://localhost:8000';
const WT = process.env.BACKEND_WORKER_TOKEN_FILE || '/home/claude/bdata/.worker_token';
const SHOTS = process.env.SHOTS || '/home/claude/shots';
mkdirSync(SHOTS, { recursive: true });

let passed = 0; const failures = [];
const check = (name, ok, extra = '') => { if (ok) { passed++; console.log(`  ok   ${name}`); } else { failures.push(name); console.log(`  FAIL ${name} ${extra}`); } };

const seeded = JSON.parse(execSync(`python3 tests/live/seed_scene.py --api ${API} --worker-token-file ${WT} --drain`, { encoding: 'utf8' }).trim().split('\n').pop());

const browser = await pw.launch({
  executablePath: await chromium.executablePath(), headless: true,
  args: [...chromium.args, '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
const ctx = await browser.newContext({ viewport: { width: 1360, height: 860 } });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
// Software WebGL is slow: stop the render loop, draw one frame, screenshot, resume.
const shot = async (n) => {
  const hasEngine = await page.evaluate(() => {
    const e = document.querySelector('.viewer-host')?.__engine; if (!e) return false;
    e.renderer.setAnimationLoop(null); e.renderer.render(e.scene, e.camera); return true;
  });
  await page.screenshot({ path: `${SHOTS}/${n}.png`, timeout: 120000 });
  if (hasEngine) await page.evaluate(() => { const e = document.querySelector('.viewer-host').__engine; e.clock.getDelta(); e.renderer.setAnimationLoop(() => e._frame()); });
};
const camPos = () => page.evaluate(() => document.querySelector('.viewer-host').__engine.camera.position.toArray());
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

console.log('A. login page, wrong password, real login');
await page.goto(`${APP}/#/projects`);
await page.waitForSelector('input[type=email]');
check('unauthenticated visit to /projects lands on the login page', page.url().includes('#/login'));
await shot('01-login');
await page.fill('input[type=email]', seeded.email); await page.fill('input[type=password]', 'wrong-password');
await page.click('button.btn-primary');
await page.waitForSelector('.note-bad', { timeout: 10000 });
check('wrong password shows the backend message', await page.locator('.note-bad', { hasText: 'Incorrect email or password' }).count() === 1);
await page.fill('input[type=password]', seeded.password); await page.click('button.btn-primary');
await page.waitForSelector('h1:has-text("Your rooms")');
check('login lands on the rooms list', true);
check('the seeded room shows as Ready with a "Walk in" button', await page.locator('.badge-green', { hasText: 'Ready' }).count() >= 1 && await page.locator('a:has-text("Walk in")').count() >= 1);
await shot('02-projects');

console.log('B. new room → upload → submit → progress page (real queue, no worker running)');
await page.fill('input[placeholder^="e.g."]', 'Browser test room'); await page.click('button:has-text("Create room")');
await page.waitForSelector('h2:has-text("1. The video")');
execSync(`ffmpeg -v error -y -f lavfi -i testsrc=duration=8:size=640x360:rate=10 -pix_fmt yuv420p /tmp/e2e-room.mp4`);
await page.setInputFiles('input[type=file]', '/tmp/e2e-room.mp4');
check('submit button says "Upload & start 3D scan"', await page.locator('button:has-text("Upload & start 3D scan")').isEnabled());
await page.click('button:has-text("Upload & start 3D scan")');
await page.waitForURL(/progress\//);
await page.waitForSelector('.stage-steps');
check('progress page shows the 4 real stages', await page.locator('.stage-steps li').count() === 4);
check('progress page says it is waiting for a GPU worker', await page.locator('text=Waiting for a GPU worker').count() === 1);
await shot('03-progress-queued');
page.once('dialog', (d) => d.accept());
await page.click('button:has-text("Cancel scan")');

console.log('C. viewer on the seeded room (real .splat through Spark in WebGL2)');
await page.goto(`${APP}/#/projects/${seeded.project_id}/viewer`);
await page.waitForSelector('.viewer-canvas', { timeout: 30000 });
await page.waitForFunction(() => !document.querySelector('.viewer-overlay'), null, { timeout: 90000 });
check('splat loaded with no overlay left on screen', true);
check('rendered with the splat renderer, not the point-cloud fallback', await page.locator('.viewer-banner').count() === 0);
check('engine reports splat kind', await page.evaluate(() => document.querySelector('.viewer-host').__engine.kind) === 'splat');
await page.waitForTimeout(2500);
const fwd0 = await page.evaluate(() => { const e = document.querySelector('.viewer-host').__engine; const v = [0, 0, -1]; const q = e.camera.quaternion; // rotate (0,0,-1) by q
  const x = q.x, y = q.y, z = q.z, w = q.w; const ux = 2 * (y * -1 * 0 - z * 0) ; return [2 * (x * z + w * y) * -1, 2 * (y * z - w * x) * -1, (1 - 2 * (x * x + y * y)) * -1]; });
check(`first view faces the same way as training camera 1 (forward ${fwd0.map((n) => n.toFixed(2))}, expected 0,0,1)`, Math.abs(fwd0[2] - 1) < 0.02 && Math.abs(fwd0[1]) < 0.02);
await shot('04-viewer-splat');
const px = await page.evaluate(() => {
  const e = document.querySelector('.viewer-host').__engine; e.renderer.render(e.scene, e.camera);
  const c = document.querySelector('.viewer-canvas'); const t = document.createElement('canvas'); t.width = 64; t.height = 36;
  const x = t.getContext('2d'); x.drawImage(c, 0, 0, 64, 36); const d = x.getImageData(0, 0, 64, 36).data; const set = new Set();
  for (let i = 0; i < d.length; i += 4) set.add(`${d[i] >> 5},${d[i + 1] >> 5},${d[i + 2] >> 5}`);
  return set.size;
});
check(`canvas shows real content (${px} distinct colours, not a blank frame)`, px > 8);

console.log('D. orbit, free-fly (WASD) and speed');
// Software WebGL drops to a few fps while the view is moving, so wall-clock distances are meaningless here.
// Real key events go through the real FlyControls; we then advance its own update() in fixed 1/60 s steps.
const step = (n) => page.evaluate((n) => { const f = document.querySelector('.viewer-host').__engine.fly; for (let i = 0; i < n; i++) f.update(1 / 60); }, n);
const start = await camPos();
await page.mouse.move(600, 450); await page.mouse.down(); await page.mouse.move(700, 450, { steps: 8 }); await page.mouse.up();
await page.waitForTimeout(3500);
check('orbit: dragging the mouse moves the camera', dist(start, await camPos()) > 0.05);
await page.click('button:has-text("Fly (WASD)")');
await page.locator('.speed input').fill('2');
check('speed slider changes the engine speed', await page.evaluate(() => document.querySelector('.viewer-host').__engine.speedMultiplier) === 2);
await page.locator('.viewer-canvas').focus();
const before = await camPos();
await page.keyboard.down('KeyW'); await step(60); await page.keyboard.up('KeyW');
await step(60);
const moved = dist(before, await camPos());
const expected = await page.evaluate(() => document.querySelector('.viewer-host').__engine.fly.getSpeed(false));
check(`fly: W for 1 s moves ≈ one speed-unit (${moved.toFixed(2)} vs ${expected.toFixed(2)} u/s configured)`, moved > expected * 0.7 && moved < expected * 1.3);
const b2 = await camPos();
await page.keyboard.down('Shift'); await page.keyboard.down('KeyW'); await step(60); await page.keyboard.up('KeyW'); await page.keyboard.up('Shift');
await step(60);
const movedShift = dist(b2, await camPos());
check(`fly: Shift is ~3x faster (${movedShift.toFixed(2)} vs ${moved.toFixed(2)})`, movedShift > moved * 2.2);
const bz = await camPos();
await page.keyboard.down('KeyE'); await step(30); await page.keyboard.up('KeyE'); await step(30);
check('fly: E moves up (world up, not along the view)', (await camPos())[1] > bz[1] + 0.2);
const yawBefore = await page.evaluate(() => document.querySelector('.viewer-host').__engine.fly.yaw);
await page.mouse.move(500, 400); await page.mouse.down(); await page.mouse.move(600, 400, { steps: 6 }); await page.mouse.up();
check('fly: dragging turns the view', Math.abs((await page.evaluate(() => document.querySelector('.viewer-host').__engine.fly.yaw)) - yawBefore) > 0.1);
await shot('05-viewer-fly');

console.log('E. camera hooks: fly to a training camera, play the walk-through path');
await page.evaluate(() => window.__scene = null);
const cams = await page.evaluate(async ({ api, token, pid }) => (await (await fetch(`${api}/projects/${pid}/scene`, { headers: { Authorization: `Bearer ${token}` } })).json()).cameras, { api: API, token: seeded.token, pid: seeded.project_id });
await page.evaluate((cam) => document.querySelector('.viewer-host').__engine.flyToCamera(cam, { duration: 300 }), cams[6]);
await page.waitForTimeout(900);
const q = await page.evaluate(() => { const e = document.querySelector('.viewer-host').__engine; return e.camera.position.toArray(); });
check('flyToCamera lands at the training camera position (up-axis converted)', Math.abs(q[0]) < 0.01 && Math.abs(q[1] - -0.2) < 0.01 && Math.abs(q[2]) < 0.01, JSON.stringify(q));
await page.click('button:has-text("Replay the walk-through")');
await page.waitForTimeout(1500);
check('walk-through is playing', await page.locator('button:has-text("Stop tour")').count() === 1);
await page.click('button:has-text("Stop tour")');
await page.waitForSelector('button:has-text("Replay the walk-through")');
check('tour can be stopped', true);

console.log('F. point-cloud fallback');
await page.locator('label:has-text("Point cloud") input').check();
await page.waitForSelector('.viewer-banner', { timeout: 30000 });
check('fallback banner marks the known limitation', (await page.locator('.viewer-banner').innerText()).includes('Known limitation'));
check('engine switched to points', await page.evaluate(() => document.querySelector('.viewer-host').__engine.kind) === 'points');
await page.waitForTimeout(1200); await shot('06-viewer-points');
await page.locator('label:has-text("Point cloud") input').uncheck();
await page.waitForFunction(() => document.querySelector('.viewer-host').__engine.kind === 'splat', null, { timeout: 60000 });
check('switching back restores the splat', true);

console.log('G. Edit object: click in the 3D room → training frame + pixel');
await page.click('button:has-text("Edit object")');
await page.waitForSelector('.editpanel');
check('edit panel opens with the 6-step stepper', await page.locator('.stepper li').count() === 6);
await page.click('button:has-text("Reset view")'); await page.waitForTimeout(600);
// aim at the sofa through the real hit-test path (engine.pickAt), the same function a mouse click calls
const picked = await page.evaluate(() => {
  const e = document.querySelector('.viewer-host').__engine; const r = e.renderer.domElement.getBoundingClientRect();
  const out = [];
  for (let gy = 0.35; gy <= 0.8; gy += 0.1) for (let gx = 0.2; gx <= 0.8; gx += 0.1) { const p = e.pickAt(r.left + r.width * gx, r.top + r.height * gy); if (p?.hit && p.camera) out.push(p); }
  return out.length ? out[0] : null;
});
check('a click on the splat hits a Gaussian and resolves to a training frame + pixel', Boolean(picked?.hit && picked.camera?.img_name), JSON.stringify(picked));
await page.waitForSelector('.clickcanvas', { timeout: 15000 });
check('the panel jumped to that frame with the click pre-placed', await page.locator('.clickcanvas .dot.pos').count() === 1);
await shot('07-edit-picked');
await page.click('button:has-text("Find the object")');
await page.waitForSelector('.stage-track', { timeout: 15000 });
check('"Find the object" creates an edit and shows its progress', await page.locator('text=Waiting for a GPU worker').count() >= 1);
await shot('08-edit-segmenting');

console.log('H. responsive layout');
await page.setViewportSize({ width: 390, height: 800 }); await page.goto(`${APP}/#/projects`); await page.waitForSelector('h1:has-text("Your rooms")');
check('no horizontal scroll on a phone-sized screen (rooms list)', await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
await page.goto(`${APP}/#/projects/${seeded.project_id}/viewer`); await page.waitForSelector('.viewer-canvas'); await page.waitForTimeout(1500);
check('no horizontal scroll on a phone-sized screen (viewer)', await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
await shot('09-mobile-projects');

const real = errors.filter((e) => !/favicon|Failed to load resource: the server responded with a status of (404|409|401|422)/.test(e));
check('no unexpected console/page errors', real.length === 0, JSON.stringify(real.slice(0, 4)));
await browser.close();
console.log(`\n${passed} checks passed, ${failures.length} failed`);
if (failures.length) { console.log('Failed:', failures); process.exit(1); }
