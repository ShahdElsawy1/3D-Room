import * as THREE from 'three';

/** Sensible speed: a fraction of the room's size per second, scaled by the slider; Shift = boost. */
export function flySpeed(sceneSize, multiplier = 1, boost = false) {
  const base = Math.max(0.05, (sceneSize || 5) * 0.12);
  return base * multiplier * (boost ? 3 : 1);
}

const MOVE_KEYS = {
  KeyW: 'forward', ArrowUp: 'forward', KeyS: 'back', ArrowDown: 'back',
  KeyA: 'left', ArrowLeft: 'left', KeyD: 'right', ArrowRight: 'right',
  KeyE: 'up', Space: 'up', KeyQ: 'down', KeyC: 'down',
};

/** WASD + drag-to-look (y-up world). Q/E or Space/C go down/up. Mouse wheel dollies forward/back. */
export class FlyControls {
  constructor(camera, dom, { getSpeed = () => 1, lookSpeed = 0.0025 } = {}) {
    this.camera = camera; this.dom = dom;
    this.getSpeed = getSpeed; this.lookSpeed = lookSpeed;
    this.enabled = false;
    this.keys = new Set(); this.boost = false;
    this.velocity = new THREE.Vector3();
    this.yaw = 0; this.pitch = 0;
    this.dragging = false; this._last = null;

    this.onKeyDown = (e) => {
      if (!this.enabled || isTyping(e.target)) return;
      if (e.code === 'ShiftLeft' || e.code === 'ShiftRight') { this.boost = true; return; }
      const dir = MOVE_KEYS[e.code];
      if (dir) { this.keys.add(dir); e.preventDefault(); }
    };
    this.onKeyUp = (e) => {
      if (e.code === 'ShiftLeft' || e.code === 'ShiftRight') { this.boost = false; return; }
      const dir = MOVE_KEYS[e.code];
      if (dir) this.keys.delete(dir);
    };
    this.onBlur = () => { this.keys.clear(); this.boost = false; this.dragging = false; };
    this.onPointerDown = (e) => {
      if (!this.enabled || e.button !== 0) return;
      this.dragging = true; this._last = { x: e.clientX, y: e.clientY };
      this.dom.focus?.({ preventScroll: true });
      this.dom.setPointerCapture?.(e.pointerId);
    };
    this.onPointerMove = (e) => {
      if (!this.enabled || !this.dragging) return;
      this.yaw -= (e.clientX - this._last.x) * this.lookSpeed;
      this.pitch -= (e.clientY - this._last.y) * this.lookSpeed;
      this.pitch = Math.max(-Math.PI / 2 + 0.02, Math.min(Math.PI / 2 - 0.02, this.pitch));
      this._last = { x: e.clientX, y: e.clientY };
      this.applyLook();
    };
    this.onPointerUp = (e) => { this.dragging = false; this.dom.releasePointerCapture?.(e.pointerId); };
    this.onWheel = (e) => {
      if (!this.enabled) return;
      e.preventDefault();
      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
      this.camera.position.addScaledVector(fwd, -Math.sign(e.deltaY) * this.getSpeed(false) * 0.15);
    };

    dom.addEventListener('keydown', this.onKeyDown);
    dom.addEventListener('keyup', this.onKeyUp);
    dom.addEventListener('blur', this.onBlur);
    dom.addEventListener('pointerdown', this.onPointerDown);
    dom.addEventListener('pointermove', this.onPointerMove);
    dom.addEventListener('pointerup', this.onPointerUp);
    dom.addEventListener('pointercancel', this.onPointerUp);
    dom.addEventListener('wheel', this.onWheel, { passive: false });
  }

  /** Read yaw/pitch from the camera's current orientation (call when switching into fly mode). */
  syncFromCamera() {
    const e = new THREE.Euler().setFromQuaternion(this.camera.quaternion, 'YXZ');
    this.yaw = e.y; this.pitch = e.x;
    this.velocity.set(0, 0, 0);
  }

  applyLook() {
    this.camera.quaternion.setFromEuler(new THREE.Euler(this.pitch, this.yaw, 0, 'YXZ'));
  }

  update(dt) {
    if (!this.enabled) return;
    const want = new THREE.Vector3();
    const q = this.camera.quaternion;
    if (this.keys.has('forward')) want.add(new THREE.Vector3(0, 0, -1).applyQuaternion(q));
    if (this.keys.has('back')) want.add(new THREE.Vector3(0, 0, 1).applyQuaternion(q));
    if (this.keys.has('right')) want.add(new THREE.Vector3(1, 0, 0).applyQuaternion(q));
    if (this.keys.has('left')) want.add(new THREE.Vector3(-1, 0, 0).applyQuaternion(q));
    if (this.keys.has('up')) want.y += 1;
    if (this.keys.has('down')) want.y -= 1;
    if (want.lengthSq() > 0) want.normalize().multiplyScalar(this.getSpeed(this.boost));
    // ease in/out so movement feels like walking, not teleporting
    this.velocity.lerp(want, 1 - Math.exp(-dt * 10));
    if (this.velocity.lengthSq() < 1e-8) this.velocity.set(0, 0, 0);
    this.camera.position.addScaledVector(this.velocity, dt);
  }

  dispose() {
    const d = this.dom;
    d.removeEventListener('keydown', this.onKeyDown);
    d.removeEventListener('keyup', this.onKeyUp);
    d.removeEventListener('blur', this.onBlur);
    d.removeEventListener('pointerdown', this.onPointerDown);
    d.removeEventListener('pointermove', this.onPointerMove);
    d.removeEventListener('pointerup', this.onPointerUp);
    d.removeEventListener('pointercancel', this.onPointerUp);
    d.removeEventListener('wheel', this.onWheel);
  }
}

function isTyping(el) {
  return el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
}
