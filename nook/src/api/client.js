// Single place that knows how to talk to the backend: base URL, token, errors, uploads.
// Every other file imports from here and never calls fetch() directly.

const TOKEN_KEY = 'nook.token';
const BASE_KEY = 'nook.apiBase';
const DEFAULT_BASE = 'http://localhost:8000';

const store = {
  get(k) { try { return window.localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { window.localStorage.setItem(k, v); } catch { /* private mode */ } },
  del(k) { try { window.localStorage.removeItem(k); } catch { /* private mode */ } },
};

export function getApiBase() {
  const fromUser = store.get(BASE_KEY);
  const fromEnv = import.meta.env?.VITE_API_URL;
  return (fromUser || fromEnv || DEFAULT_BASE).replace(/\/+$/, '');
}
export function setApiBase(url) {
  const clean = (url || '').trim().replace(/\/+$/, '');
  if (clean) store.set(BASE_KEY, clean); else store.del(BASE_KEY);
}
export const getToken = () => store.get(TOKEN_KEY);
export const setToken = (t) => store.set(TOKEN_KEY, t);
export const clearToken = () => store.del(TOKEN_KEY);

let unauthorizedHandler = null;
/** AuthContext registers a callback here so any 401 sends the user back to the login page. */
export function setUnauthorizedHandler(fn) { unauthorizedHandler = fn; }

/** Backend URLs for files are relative ("/projects/…/files/…?t=…"): make them absolute. */
export function absUrl(path) {
  if (!path) return null;
  if (/^(https?:|blob:|data:)/i.test(path)) return path;
  return getApiBase() + (path.startsWith('/') ? path : '/' + path);
}

export class ApiError extends Error {
  constructor(status, message, detail) {
    super(message);
    this.name = 'ApiError';
    this.status = status; // 0 = never reached the server
    this.detail = detail;
  }
}

/** FastAPI sends {detail: "text"} or, for 422, {detail: [{loc, msg}]}. Make both readable. */
export function messageFromDetail(detail, fallback) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => {
      const field = Array.isArray(d.loc) ? d.loc.filter((x) => x !== 'body').join('.') : '';
      const msg = String(d.msg || '').replace(/^Value error, /, '');
      return field ? `${field}: ${msg}` : msg;
    }).join('; ');
  }
  return fallback;
}

async function parse(res) {
  if (res.status === 204) return null;
  const text = await res.text();
  if (!text) return null;
  try { return JSON.parse(text); } catch { return text; }
}

export async function request(method, path, { body, signal, auth = true } = {}) {
  const headers = {};
  const token = getToken();
  if (auth && token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers['Content-Type'] = 'application/json';

  let res;
  try {
    res = await fetch(getApiBase() + path, {
      method, headers, signal,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    throw new ApiError(0, `Can't reach the server at ${getApiBase()}. Is the backend running?`, null);
  }

  const data = await parse(res);
  if (!res.ok) {
    const msg = messageFromDetail(data?.detail, `Request failed (${res.status})`);
    if (res.status === 401 && auth && token && unauthorizedHandler) unauthorizedHandler();
    throw new ApiError(res.status, msg, data?.detail);
  }
  return data;
}

/** multipart upload with a real progress callback (fetch cannot report upload progress). */
export function uploadFile(path, file, { onProgress, signal } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', getApiBase() + path);
    const token = getToken();
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* not json */ }
      if (xhr.status >= 200 && xhr.status < 300) return resolve(data);
      if (xhr.status === 401 && unauthorizedHandler) unauthorizedHandler();
      reject(new ApiError(xhr.status, messageFromDetail(data?.detail, `Upload failed (${xhr.status})`), data?.detail));
    };
    xhr.onerror = () => reject(new ApiError(0, `Can't reach the server at ${getApiBase()}. Is the backend running?`, null));
    xhr.onabort = () => reject(new DOMException('Upload cancelled', 'AbortError'));
    if (signal) signal.addEventListener('abort', () => xhr.abort());
    const fd = new FormData();
    fd.append('file', file);
    xhr.send(fd);
  });
}

/** Fetch a protected file (e.g. the original video) as a blob URL. Caller must revoke it. */
export async function fetchBlobUrl(path, { signal } = {}) {
  const token = getToken();
  const res = await fetch(getApiBase() + path, {
    signal, headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) throw new ApiError(res.status, `Could not load file (${res.status})`);
  return URL.createObjectURL(await res.blob());
}
