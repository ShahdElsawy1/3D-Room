import * as THREE from 'three';
import { CameraPath, easeInOut } from '../../src/viewer/cameraPath.js';

const pose = (x, yaw = 0) => ({ position: new THREE.Vector3(x, 0, 0), quaternion: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, yaw, 0)), fov: 60 });

describe('CameraPath', () => {
  it('starts and ends exactly on the first and last pose', () => {
    const path = new CameraPath([pose(0), pose(1, 1), pose(3, 2)]);
    expect(path.sample(0).position.x).toBeCloseTo(0);
    expect(path.sample(1).position.x).toBeCloseTo(3);
    expect(path.sample(1).quaternion.angleTo(pose(3, 2).quaternion)).toBeCloseTo(0, 5);
  });
  it('passes through the middle keyframe at u = 0.5 (three poses)', () => {
    const path = new CameraPath([pose(0), pose(1, 1), pose(3, 2)]);
    expect(path.sample(0.5).position.x).toBeCloseTo(1, 5);
  });
  it('interpolates orientation smoothly', () => {
    const s = new CameraPath([pose(0, 0), pose(2, 1)]).sample(0.5);
    expect(s.quaternion.angleTo(pose(0, 0.5).quaternion)).toBeCloseTo(0, 5);
  });
  it('handles a single pose and clamps u', () => {
    const path = new CameraPath([pose(4)]);
    expect(path.sample(0.3).position.x).toBe(4);
    expect(new CameraPath([pose(0), pose(1)]).sample(5).position.x).toBeCloseTo(1);
  });
  it('rejects an empty path', () => { expect(() => new CameraPath([])).toThrow(); });
  it('easeInOut is 0 → 1 and monotonic', () => {
    expect(easeInOut(0)).toBe(0); expect(easeInOut(1)).toBe(1);
    expect(easeInOut(0.25)).toBeLessThan(easeInOut(0.75));
  });
});
