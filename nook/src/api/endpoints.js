// One tiny function per backend route (see backend/docs/API.md). Pages never build URLs by hand.
import { request, uploadFile } from './client.js';

export const auth = {
  signup: (email, password) => request('POST', '/auth/signup', { body: { email, password }, auth: false }),
  login: (email, password) => request('POST', '/auth/login', { body: { email, password }, auth: false }),
  me: () => request('GET', '/auth/me'),
};

export const health = () => request('GET', '/health', { auth: false });

export const projects = {
  list: () => request('GET', '/projects'),
  create: (name, description = '') => request('POST', '/projects', { body: { name, description } }),
  get: (id) => request('GET', `/projects/${id}`),
  update: (id, patch) => request('PATCH', `/projects/${id}`, { body: patch }),
  remove: (id) => request('DELETE', `/projects/${id}`),
  uploadVideo: (id, file, opts) => uploadFile(`/projects/${id}/video`, file, opts),
  videoPath: (id) => `/projects/${id}/video`,
  reconstruct: (id, options = {}) => request('POST', `/projects/${id}/reconstruct`, { body: options }),
  tasks: (id) => request('GET', `/projects/${id}/tasks`),
  scene: (id, version) => request('GET', `/projects/${id}/scene${version && version !== 'original' ? `?version=${encodeURIComponent(version)}` : ''}`),
  sceneVersions: (id) => request('GET', `/projects/${id}/scene/versions`),
  frames: (id) => request('GET', `/projects/${id}/frames`),
};

export const tasks = {
  get: (id) => request('GET', `/tasks/${id}`),
  cancel: (id) => request('POST', `/tasks/${id}/cancel`),
};

export const edits = {
  list: (projectId) => request('GET', `/projects/${projectId}/edits`),
  create: (projectId, ref_frame, clicks) => request('POST', `/projects/${projectId}/edits`, { body: { ref_frame, clicks } }),
  get: (id) => request('GET', `/edits/${id}`),
  remove: (id) => request('DELETE', `/edits/${id}`),
  clicks: (id, clicks) => request('POST', `/edits/${id}/clicks`, { body: { clicks } }),
  track: (id) => request('POST', `/edits/${id}/track`),
  frames: (id) => request('GET', `/edits/${id}/frames`),
  patchFrames: (id, patch) => request('PATCH', `/edits/${id}/frames`, { body: patch }),
  preview: (id, prompt, num_seeds = 3) => request('POST', `/edits/${id}/preview`, { body: { prompt, num_seeds } }),
  candidates: (id) => request('GET', `/edits/${id}/candidates`),
  accept: (id, seed) => request('POST', `/edits/${id}/accept`, { body: { seed } }),
  rebuild: (id, mode = 'finetune') => request('POST', `/edits/${id}/rebuild`, { body: { mode } }),
};
