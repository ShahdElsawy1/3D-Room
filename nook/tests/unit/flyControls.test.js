import * as THREE from 'three';
import { FlyControls, flySpeed } from '../../src/viewer/flyControls.js';

function setup(speed = 1) {
  const dom = document.createElement('canvas');
  document.body.appendChild(dom);
  const camera = new THREE.PerspectiveCamera();
  const fly = new FlyControls(camera, dom, { getSpeed: () => speed });
  fly.enabled = true; fly.syncFromCamera();
  return { dom, camera, fly };
}
const key = (dom, type, code) => dom.dispatchEvent(new KeyboardEvent(type, { code, bubbles: true, cancelable: true }));
const run = (fly, seconds, dt = 1 / 60) => { for (let t = 0; t < seconds; t += dt) fly.update(dt); };

describe('flySpeed', () => {
  it('scales with room size, slider and Shift', () => {
    expect(flySpeed(10, 1)).toBeCloseTo(1.2);
    expect(flySpeed(10, 2)).toBeCloseTo(2.4);
    expect(flySpeed(10, 1, true)).toBeCloseTo(3.6);
  });
  it('never drops to zero for tiny scenes', () => { expect(flySpeed(0.01, 0.25)).toBeGreaterThan(0); });
});

describe('FlyControls', () => {
  it('W moves forward (toward -z) at roughly the configured speed', () => {
    const { dom, camera, fly } = setup(2);
    key(dom, 'keydown', 'KeyW'); run(fly, 2);
    expect(camera.position.z).toBeLessThan(-3); // ~2 u/s for 2 s, minus ease-in
    expect(Math.abs(camera.position.x)).toBeLessThan(1e-6);
  });
  it('S, A, D and Q/E move the other ways', () => {
    const { dom, camera, fly } = setup(2);
    key(dom, 'keydown', 'KeyD'); run(fly, 1); key(dom, 'keyup', 'KeyD');
    expect(camera.position.x).toBeGreaterThan(1);
    key(dom, 'keydown', 'KeyE'); run(fly, 1); key(dom, 'keyup', 'KeyE');
    expect(camera.position.y).toBeGreaterThan(1);
  });
  it('stops (eases out) after the key is released', () => {
    const { dom, camera, fly } = setup(2);
    key(dom, 'keydown', 'KeyW'); run(fly, 1); key(dom, 'keyup', 'KeyW'); run(fly, 1);
    const z = camera.position.z; run(fly, 1);
    expect(camera.position.z).toBeCloseTo(z, 3);
  });
  it('ignores keys while typing in an input', () => {
    const { camera, fly } = setup(2);
    const input = document.createElement('input'); document.body.appendChild(input);
    input.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyW', bubbles: true }));
    run(fly, 1);
    expect(camera.position.length()).toBe(0);
  });
  it('does nothing when disabled (orbit mode)', () => {
    const { dom, camera, fly } = setup(2);
    fly.enabled = false; key(dom, 'keydown', 'KeyW'); run(fly, 1);
    expect(camera.position.length()).toBe(0);
  });
  it('dragging looks around and clamps pitch so you cannot flip over', () => {
    const { dom, camera, fly } = setup();
    dom.dispatchEvent(new PointerEvent('pointerdown', { clientX: 100, clientY: 100, button: 0, bubbles: true }));
    dom.dispatchEvent(new PointerEvent('pointermove', { clientX: 100, clientY: -100000, bubbles: true }));
    expect(fly.pitch).toBeLessThanOrEqual(Math.PI / 2);
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    expect(fwd.y).toBeGreaterThan(0.99);
  });
  it('moves where you look (flying, not walking)', () => {
    const { dom, camera, fly } = setup(2);
    fly.pitch = Math.PI / 4; fly.applyLook();
    key(dom, 'keydown', 'KeyW'); run(fly, 1);
    expect(camera.position.y).toBeGreaterThan(0.5);
  });
  it('dispose removes listeners', () => {
    const { dom, camera, fly } = setup(2);
    fly.dispose(); key(dom, 'keydown', 'KeyW'); run(fly, 1);
    expect(camera.position.length()).toBe(0);
  });
});
