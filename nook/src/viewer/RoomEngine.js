import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { absUrl } from '../api/client.js';
import { cameraToThree, pickBestCamera, threeToColmapPoint, upQuaternion, DEFAULT_UP } from './cameraMath.js';
import { CameraPath, easeInOut } from './cameraPath.js';
import { FlyControls, flySpeed } from './flyControls.js';

/**
 * Everything that touches WebGL lives here, so React components stay small.
 *
 * Public hooks for Week 2 (spec §4.5):
 *   onPick({x, y, width, height, hit, point, camera})   user clicked at screen position (x, y)
 *   flyToCamera(cam) / setCameraPose(cam)               load a specific camera pose (API camera object)
 *   playCameraPath(cameras, {duration, loop})           load a camera path
 */
const PICK_RADII = [0, 4, 8, 14, 22]; // CSS pixels

export class RoomEngine {
  constructor(container, { onPick, onStatus } = {}) {
    this.container = container;
    this.onPick = onPick;
    this.onStatus = onStatus || (() => {});

    this.scene = new THREE.Scene();
    this.world = new THREE.Group(); // holds the splat/points; rotated so the room is upright
    this.scene.add(this.world);
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.01, 500);
    this.camera.position.set(0, 0.5, 3);

    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.domElement.className = 'viewer-canvas';
    this.renderer.domElement.tabIndex = 0; // so keyboard events (WASD) reach it
    container.appendChild(this.renderer.domElement);

    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.08;

    this.speedMultiplier = 1;
    this.sceneSize = 5;
    this.fly = new FlyControls(this.camera, this.renderer.domElement, {
      getSpeed: (boost) => flySpeed(this.sceneSize, this.speedMultiplier, boost),
    });

    this.mode = 'orbit';
    this.Q = upQuaternion(DEFAULT_UP);
    this.cameras = [];
    this.meshes = { splat: null, points: null };
    this.kind = null; // 'splat' | 'points'
    this.path = null; this.pathState = null;
    this.tween = null;
    this.home = null;
    this.disposed = false;
    this.clock = new THREE.Clock();
    this.raycaster = new THREE.Raycaster();

    this._bindPick();
    this._resizeObserver = new ResizeObserver(() => this.resize());
    this._resizeObserver.observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => this._frame());
  }

  // ------------------------------------------------------------------ loading
  /**
   * scene = the backend's SceneOut with URLs already made absolute.
   * keepCamera: used by the before/after toggle so the view does not jump.
   */
  async loadScene(scene, { preferPoints = false, keepCamera = false, onProgress } = {}) {
    this.Q = upQuaternion(scene.up_vector || DEFAULT_UP);
    this.world.quaternion.copy(this.Q);
    this.cameras = scene.cameras || [];
    this._frameBounds(scene.meta);

    let kind = null; let reason = null;
    let nextMesh = null;
    if (!preferPoints) {
      try {
        nextMesh = await this._loadSplat(scene.splat.url, scene.splat.size_bytes, onProgress);
        kind = 'splat';
      } catch (err) {
        reason = err?.message || String(err);
        console.warn('Splat rendering failed, falling back to points:', err);
      }
    }
    if (!kind) {
      if (!scene.points?.url) throw new Error(reason ? `Splat rendering failed (${reason}) and this scene has no point cloud.` : 'This scene has no point cloud.');
      nextMesh = await this._loadPoints(scene.points.url);
      kind = 'points';
    }
    if (this.disposed) { this._disposeMesh(nextMesh); return { kind }; }

    // swap meshes only once the new one is ready, so there is never an empty frame
    this._disposeMesh(this.meshes.splat); this._disposeMesh(this.meshes.points);
    this.meshes = { splat: kind === 'splat' ? nextMesh : null, points: kind === 'points' ? nextMesh : null };
    this.world.add(nextMesh);
    this.kind = kind;
    if (!keepCamera || !this.home) this.resetView();
    this.onStatus({ type: 'ready', kind, fallbackReason: kind === 'points' ? (reason || (preferPoints ? 'requested' : null)) : null });
    return { kind, fallbackReason: reason };
  }

  async _loadSplat(url, sizeBytes, onProgress) {
    const gl = document.createElement('canvas').getContext('webgl2');
    if (!gl) throw new Error('WebGL2 is not available in this browser');
    const { SparkRenderer, SplatMesh } = await import('@sparkjsdev/spark'); // lazy: only viewer visitors download it
    if (!this.spark) {
      this.spark = new SparkRenderer({ renderer: this.renderer });
      this.scene.add(this.spark);
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not download the splat (${res.status})`);
    const bytes = await readWithProgress(res, sizeBytes, onProgress);
    const mesh = new SplatMesh({ fileBytes: bytes, fileType: 'splat', raycastable: true });
    await mesh.initialized;
    return mesh;
  }

  async _loadPoints(url) {
    const geometry = await new Promise((resolve, reject) => new PLYLoader().load(url, resolve, undefined, reject));
    const hasColor = Boolean(geometry.getAttribute('color'));
    const material = new THREE.PointsMaterial({ size: 3, sizeAttenuation: false, vertexColors: hasColor, color: hasColor ? 0xffffff : 0x2563b0 });
    const points = new THREE.Points(geometry, material);
    this.raycaster.params.Points.threshold = this.sceneSize * 0.01;
    return points;
  }

  _disposeMesh(m) {
    if (!m) return;
    this.world.remove(m);
    m.dispose?.();
    m.geometry?.dispose?.();
    m.material?.dispose?.();
  }

  _frameBounds(meta = {}) {
    const lo = meta.bbox_min; const hi = meta.bbox_max;
    if (lo && hi) {
      const size = new THREE.Vector3(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
      this.sceneSize = Math.max(0.5, size.length() / 1.7);
    }
    const mid = meta.median_position || (lo && hi ? lo.map((v, i) => (v + hi[i]) / 2) : [0, 0, 0]);
    this.center = new THREE.Vector3(...mid).applyQuaternion(this.Q);
    this.camera.near = Math.max(0.005, this.sceneSize * 0.002);
    this.camera.far = this.sceneSize * 60;
    this.camera.updateProjectionMatrix();
  }

  // ------------------------------------------------------------------ view control
  /** Start from the first training camera: it is inside the room, so you never begin in the void. */
  resetView() {
    if (this.cameras.length) this.setCameraPose(this.cameras[0]);
    else {
      this.camera.position.copy(this.center).add(new THREE.Vector3(0, this.sceneSize * 0.1, this.sceneSize * 0.8));
      this.camera.lookAt(this.center);
    }
    if (this.cameras.length) this._aimOrbitAhead(); else { this.orbit.target.copy(this.center); this.orbit.update(); }
    this.home = { position: this.camera.position.clone(), quaternion: this.camera.quaternion.clone() };
    if (this.mode === 'fly') this.fly.syncFromCamera();
  }

  /** Orbit around a point straight ahead of the camera, so switching to orbit never changes the view direction. */
  _aimOrbitAhead() {
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
    this.orbit.target.copy(this.camera.position).addScaledVector(fwd, Math.max(this.sceneSize * 0.25, 0.5));
    this.orbit.update();
  }

  setMode(mode) {
    if (mode === this.mode) return;
    this.stopCameraPath();
    this.mode = mode;
    if (mode === 'fly') {
      this.orbit.enabled = false;
      this.fly.enabled = true;
      this.fly.syncFromCamera();
      this.renderer.domElement.focus({ preventScroll: true });
    } else {
      this.fly.enabled = false;
      this._aimOrbitAhead();
      this.orbit.enabled = true;
    }
    this.onStatus({ type: 'mode', mode });
  }

  setSpeed(multiplier) { this.speedMultiplier = multiplier; }

  /** Jump to a training camera (API camera object). */
  setCameraPose(cam) {
    const p = cameraToThree(cam, this.Q);
    this._applyPose(p);
  }

  _applyPose({ position, quaternion, fov }) {
    this.camera.position.copy(position);
    this.camera.quaternion.copy(quaternion);
    if (fov && Math.abs(this.camera.fov - fov) > 0.01) { this.camera.fov = fov; this.camera.updateProjectionMatrix(); }
    if (this.mode === 'fly') this.fly.syncFromCamera();
  }

  /** Smoothly move to a training camera. Returns a promise that resolves when it arrives. */
  flyToCamera(cam, { duration = 900 } = {}) {
    this.stopCameraPath();
    const target = cameraToThree(cam, this.Q);
    const from = { position: this.camera.position.clone(), quaternion: this.camera.quaternion.clone(), fov: this.camera.fov };
    return new Promise((resolve) => {
      this.tween = { start: performance.now(), duration, from, to: target, resolve };
    });
  }

  /** Load a camera path: play through API camera objects with smooth interpolation. */
  playCameraPath(cameras, { duration = 20000, loop = false } = {}) {
    if (!cameras?.length) return Promise.resolve();
    this.stopCameraPath();
    this.path = CameraPath.fromCameras(cameras, this._up());
    return new Promise((resolve) => {
      this.pathState = { start: performance.now(), duration, loop, resolve };
      this.orbit.enabled = false;
      this.fly.enabled = false;
      this.onStatus({ type: 'path', playing: true });
    });
  }

  stopCameraPath() {
    if (this.tween) { this.tween.resolve?.(); this.tween = null; }
    if (!this.pathState) return;
    const { resolve } = this.pathState;
    this.pathState = null; this.path = null;
    // hand control back to the current mode
    if (this.mode === 'fly') { this.fly.enabled = true; this.fly.syncFromCamera(); }
    else {
      this.orbit.enabled = true; this._aimOrbitAhead();
    }
    this.onStatus({ type: 'path', playing: false });
    resolve?.();
  }

  _up() { return new THREE.Vector3(0, 1, 0).applyQuaternion(this.Q.clone().invert()).toArray(); }

  // ------------------------------------------------------------------ picking
  _bindPick() {
    const dom = this.renderer.domElement;
    let down = null;
    this._onDown = (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; };
    this._onUp = (e) => {
      if (!down || e.button !== 0) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      const quick = performance.now() - down.t < 500;
      down = null;
      if (moved < 5 && quick) this.pickAt(e.clientX, e.clientY);
    };
    dom.addEventListener('pointerdown', this._onDown);
    dom.addEventListener('pointerup', this._onUp);
  }

  /** The "user clicked at screen position (x, y)" hook. x/y are CSS pixels inside the canvas. */
  pickAt(clientX, clientY) {
    if (!this.onPick) return null;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const x = clientX - rect.left; const y = clientY - rect.top;
    // Gaussians are tiny, so a click can fall between two of them. Try the exact pixel first, then
    // widen the search in rings (a few px at a time) and take the first hit.
    let hitPoint = null;
    const target = this.meshes.splat || this.meshes.points;
    if (target) {
      outer: for (const radius of PICK_RADII) {
        const steps = radius === 0 ? 1 : 8;
        for (let k = 0; k < steps; k += 1) {
          const ang = (k / steps) * Math.PI * 2;
          const nx = ((x + Math.cos(ang) * radius) / rect.width) * 2 - 1;
          const ny = -((y + Math.sin(ang) * radius) / rect.height) * 2 + 1;
          this.raycaster.setFromCamera(new THREE.Vector2(nx, ny), this.camera);
          try {
            const hits = this.raycaster.intersectObject(target, false);
            if (hits.length) { hitPoint = hits[0].point; break outer; }
          } catch (err) { console.warn('raycast failed', err); break outer; }
        }
      }
    }
    const point = hitPoint ? threeToColmapPoint(hitPoint, this.Q).toArray() : null;
    const best = point ? pickBestCamera(this.cameras, point) : null;
    const result = {
      x: Math.round(x), y: Math.round(y), width: Math.round(rect.width), height: Math.round(rect.height),
      hit: Boolean(hitPoint), point,
      camera: best ? { id: best.camera.id, img_name: best.camera.img_name, x: best.x, y: best.y, width: best.camera.width, height: best.camera.height } : null,
    };
    this.onPick(result);
    return result;
  }

  // ------------------------------------------------------------------ loop
  resize() {
    const w = this.container.clientWidth || 1; const h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _frame() {
    if (this.disposed) return;
    const dt = Math.min(this.clock.getDelta(), 0.1);
    const now = performance.now();

    if (this.tween) {
      const { start, duration, from, to, resolve } = this.tween;
      const t = Math.min(1, (now - start) / duration);
      const e = easeInOut(t);
      this.camera.position.lerpVectors(from.position, to.position, e);
      this.camera.quaternion.slerpQuaternions(from.quaternion, to.quaternion, e);
      if (to.fov) { this.camera.fov = from.fov + (to.fov - from.fov) * e; this.camera.updateProjectionMatrix(); }
      if (t >= 1) { this.tween = null; if (this.mode === 'fly') this.fly.syncFromCamera(); else this._aimOrbitAhead(); resolve(); }
    } else if (this.pathState && this.path) {
      const { start, duration, loop } = this.pathState;
      let u = (now - start) / duration;
      if (u >= 1 && loop) u %= 1;
      const s = this.path.sample(easeInOut(Math.min(1, u)));
      this._applyPose(s);
      if (u >= 1 && !loop) this.stopCameraPath();
    } else if (this.mode === 'fly') {
      this.fly.update(dt);
    } else {
      this.orbit.update();
    }
    this.renderer.render(this.scene, this.camera);
  }

  // ------------------------------------------------------------------ cleanup
  dispose() {
    this.disposed = true;
    this.renderer.setAnimationLoop(null);
    this._resizeObserver.disconnect();
    const dom = this.renderer.domElement;
    dom.removeEventListener('pointerdown', this._onDown);
    dom.removeEventListener('pointerup', this._onUp);
    this.fly.dispose();
    this.orbit.dispose();
    this._disposeMesh(this.meshes.splat); this._disposeMesh(this.meshes.points);
    this.spark?.dispose?.();
    this.renderer.dispose();
    dom.remove();
  }
}

/** Download with a progress callback (fetch + stream), falling back to a plain read. */
async function readWithProgress(res, expectedBytes, onProgress) {
  const total = Number(res.headers.get('content-length')) || expectedBytes || 0;
  if (!res.body || !onProgress) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = []; let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); received += value.length;
    onProgress(total ? Math.min(1, received / total) : 0);
  }
  const out = new Uint8Array(received); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

/** Make the backend's relative file URLs absolute before handing a scene to the engine. */
export function withAbsoluteUrls(scene) {
  const fix = (ref) => (ref ? { ...ref, url: absUrl(ref.url) } : ref);
  return { ...scene, splat: fix(scene.splat), ply: fix(scene.ply), points: fix(scene.points), cameras_url: absUrl(scene.cameras_url) };
}
