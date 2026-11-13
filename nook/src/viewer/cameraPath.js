import * as THREE from 'three';
import { cameraToThree, upQuaternion } from './cameraMath.js';

export const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

/** Smooth path through camera poses: Catmull-Rom for position, slerp for orientation. */
export class CameraPath {
  /** poses: [{position: Vector3, quaternion: Quaternion, fov?: number}] in the y-up three.js world */
  constructor(poses) {
    if (!poses?.length) throw new Error('A camera path needs at least one pose');
    this.poses = poses;
    this.curve = poses.length > 1
      ? new THREE.CatmullRomCurve3(poses.map((p) => p.position.clone()), false, 'centripetal')
      : null;
  }

  get length() { return this.poses.length; }

  /** u in [0,1] → {position, quaternion, fov} */
  sample(u) {
    const n = this.poses.length;
    const c = Math.min(1, Math.max(0, u));
    if (n === 1) { const p = this.poses[0]; return { position: p.position.clone(), quaternion: p.quaternion.clone(), fov: p.fov }; }
    const f = c * (n - 1);
    const i = Math.min(n - 2, Math.floor(f));
    const a = this.poses[i]; const b = this.poses[i + 1];
    const t = f - i;
    return {
      position: this.curve.getPoint(c),
      quaternion: a.quaternion.clone().slerp(b.quaternion, t),
      fov: a.fov != null && b.fov != null ? a.fov + (b.fov - a.fov) * t : a.fov,
    };
  }

  /** Build from backend camera objects (COLMAP world). */
  static fromCameras(cameras, up) {
    const Q = upQuaternion(up);
    return new CameraPath(cameras.map((c) => cameraToThree(c, Q)));
  }
}
