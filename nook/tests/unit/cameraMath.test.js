import * as THREE from 'three';
import { cameraToThree, fovFromIntrinsics, pickBestCamera, projectToCamera, threeToColmapPoint, upQuaternion } from '../../src/viewer/cameraMath.js';

const cam = (over = {}) => ({
  id: 1, img_name: '000001.jpg', width: 960, height: 540, position: [0, 0, 0],
  rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], fx: 800, fy: 800, cx: 480, cy: 270, ...over,
});

describe('upQuaternion', () => {
  it('turns COLMAP up (0,-1,0) into three.js +Y', () => {
    const v = new THREE.Vector3(0, -1, 0).applyQuaternion(upQuaternion([0, -1, 0]));
    expect(v.x).toBeCloseTo(0); expect(v.y).toBeCloseTo(1); expect(v.z).toBeCloseTo(0);
  });
  it('is the identity when the scene is already y-up', () => {
    const q = upQuaternion([0, 1, 0]);
    expect(q.angleTo(new THREE.Quaternion())).toBeCloseTo(0);
  });
});

describe('fovFromIntrinsics', () => {
  it('matches 2*atan(h / 2fy)', () => {
    expect(fovFromIntrinsics(cam({ fy: 270, height: 540 }))).toBeCloseTo(90, 5);
  });
  it('falls back to 60 when intrinsics are missing', () => { expect(fovFromIntrinsics({})).toBe(60); });
});

describe('projectToCamera', () => {
  it('puts a point straight ahead at the principal point', () => {
    const p = projectToCamera(cam(), [0, 0, 2]);
    expect(p.x).toBeCloseTo(480); expect(p.y).toBeCloseTo(270); expect(p.depth).toBeCloseTo(2); expect(p.inFrame).toBe(true);
  });
  it('+x is right and +y is down in the image (OpenCV)', () => {
    const p = projectToCamera(cam(), [0.5, 0.25, 2]);
    expect(p.x).toBeCloseTo(480 + 800 * 0.25); expect(p.y).toBeCloseTo(270 + 800 * 0.125);
  });
  it('rejects points behind the camera', () => {
    const p = projectToCamera(cam(), [0, 0, -1]);
    expect(p.inFrame).toBe(false); expect(Number.isNaN(p.x)).toBe(true);
  });
  it('accounts for camera position and rotation', () => {
    // camera at (1,0,0) looking along world +x (camera z → world x), camera x → world -z? use a proper rotation:
    const R = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]; // columns: camX=(0,0,-1), camY=(0,1,0), camZ=(1,0,0)
    const p = projectToCamera(cam({ position: [1, 0, 0], rotation: R }), [4, 0, 0]);
    expect(p.depth).toBeCloseTo(3); expect(p.x).toBeCloseTo(480); expect(p.y).toBeCloseTo(270);
  });
});

describe('cameraToThree', () => {
  it('a point in front of an OpenCV camera ends up on the three.js camera’s -z axis', () => {
    const Q = upQuaternion([0, -1, 0]);
    // an arbitrary tilted camera
    const e = new THREE.Euler(0.3, -0.7, 0.2);
    const m = new THREE.Matrix4().makeRotationFromEuler(e);
    const R = [0, 1, 2].map((r) => [0, 1, 2].map((c) => m.elements[c * 4 + r]));
    const c = cam({ position: [1, 2, 3], rotation: R });
    const world = new THREE.Vector3(1, 2, 3).add(new THREE.Vector3(R[0][2], R[1][2], R[2][2]).multiplyScalar(2)); // 2 m along the optical axis
    const pose = cameraToThree(c, Q);
    const cam3 = new THREE.PerspectiveCamera(); cam3.position.copy(pose.position); cam3.quaternion.copy(pose.quaternion); cam3.updateMatrixWorld(true);
    const inCam = world.applyQuaternion(Q).applyMatrix4(cam3.matrixWorldInverse);
    expect(inCam.x).toBeCloseTo(0, 5); expect(inCam.y).toBeCloseTo(0, 5); expect(inCam.z).toBeCloseTo(-2, 5);
  });
  it('keeps the camera upright for the y-down COLMAP world', () => {
    const pose = cameraToThree(cam()); // identity OpenCV camera in a y-down world
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(pose.quaternion);
    expect(up.y).toBeGreaterThan(0.99); // camera's own "up" points up in the viewer
  });
  it('round-trips a point between worlds', () => {
    const Q = upQuaternion();
    const p = new THREE.Vector3(1, -2, 3);
    const back = threeToColmapPoint(p.clone().applyQuaternion(Q), Q);
    expect(back.distanceTo(p)).toBeCloseTo(0, 6);
  });
});

describe('pickBestCamera (click in 3D → frame + pixel)', () => {
  const cameras = [
    cam({ id: 1, position: [0, 0, 0] }),
    cam({ id: 2, position: [3, 0, 0] }), // sees the point off-centre
    cam({ id: 3, position: [0, 0, 10], rotation: [[-1, 0, 0], [0, 1, 0], [0, 0, -1]] }), // looks back at the origin side, 180° yaw
  ];
  it('chooses the camera where the point is most central', () => {
    const best = pickBestCamera(cameras, [0, 0, 4]);
    expect(best.camera.id).toBe(1);
    expect(best.x).toBe(480); expect(best.y).toBe(270);
  });
  it('ignores cameras that do not see the point', () => {
    expect(pickBestCamera([cam({ position: [0, 0, 10] })], [0, 0, 0])).toBeNull(); // point is behind it
  });
  it('ignores points projecting outside the frame', () => {
    expect(pickBestCamera([cam()], [10, 0, 1])).toBeNull();
  });
});
