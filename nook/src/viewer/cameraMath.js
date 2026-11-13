import * as THREE from 'three';

/*
 * Coordinate notes (see backend/docs/API.md, "scene"):
 *  - The backend works in COLMAP/OpenCV world space: +y points DOWN. It gives us `up_vector` (0,-1,0).
 *  - three.js is y-up and its cameras look down -z; OpenCV cameras look down +z with y down.
 *  - So everything shown in three.js lives inside a group rotated by Q (up_vector → +Y).
 *    Functions here convert between the two worlds so the rest of the app can keep speaking COLMAP.
 */

export const DEFAULT_UP = [0, -1, 0];
const FLIP_YZ = new THREE.Matrix4().makeScale(1, -1, -1); // OpenCV camera axes → three camera axes

export function upQuaternion(up = DEFAULT_UP) {
  const u = new THREE.Vector3(...up).normalize();
  return new THREE.Quaternion().setFromUnitVectors(u, new THREE.Vector3(0, 1, 0));
}

export function fovFromIntrinsics(cam) {
  if (!cam?.fy || !cam?.height) return 60;
  return (2 * Math.atan(cam.height / (2 * cam.fy)) * 180) / Math.PI;
}

/** A training camera from the API (COLMAP world) → a pose in the y-up three.js world. */
export function cameraToThree(cam, Q = upQuaternion()) {
  const R = cam.rotation; // camera-to-world, rows
  const m = new THREE.Matrix4().set(
    R[0][0], R[0][1], R[0][2], 0,
    R[1][0], R[1][1], R[1][2], 0,
    R[2][0], R[2][1], R[2][2], 0,
    0, 0, 0, 1,
  ).multiply(FLIP_YZ);
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(m).premultiply(Q).normalize();
  const position = new THREE.Vector3(...cam.position).applyQuaternion(Q);
  return { position, quaternion, fov: fovFromIntrinsics(cam), aspect: cam.width && cam.height ? cam.width / cam.height : undefined };
}

export function threeToColmapPoint(p, Q = upQuaternion()) {
  return p.clone().applyQuaternion(Q.clone().invert());
}

/** Project a COLMAP-world point into a training camera. depth > 0 means in front of it. */
export function projectToCamera(cam, point) {
  const R = cam.rotation;
  const d = [point[0] - cam.position[0], point[1] - cam.position[1], point[2] - cam.position[2]];
  // world→camera = Rᵀ · d
  const x = R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2];
  const y = R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2];
  const z = R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2];
  if (z <= 1e-6) return { x: NaN, y: NaN, depth: z, inFrame: false };
  const u = (cam.fx * x) / z + cam.cx;
  const v = (cam.fy * y) / z + cam.cy;
  const inFrame = u >= 0 && u < cam.width && v >= 0 && v < cam.height;
  return { x: u, y: v, depth: z, inFrame };
}

/**
 * Spec §4.1, option 2: given a 3D point under the user's click, find the training frame that
 * sees it best (most central, in front of the camera) and the 2D pixel to hand to the segmenter.
 */
export function pickBestCamera(cameras, point, { margin = 0.04 } = {}) {
  let best = null;
  for (const cam of cameras || []) {
    const p = projectToCamera(cam, point);
    if (p.depth <= 0 || !Number.isFinite(p.x)) continue;
    const mx = cam.width * margin; const my = cam.height * margin;
    if (p.x < mx || p.x > cam.width - mx || p.y < my || p.y > cam.height - my) continue;
    const dx = (p.x - cam.width / 2) / cam.width;
    const dy = (p.y - cam.height / 2) / cam.height;
    const score = Math.hypot(dx, dy);
    if (!best || score < best.score) best = { camera: cam, x: Math.round(p.x), y: Math.round(p.y), depth: p.depth, score };
  }
  return best;
}
