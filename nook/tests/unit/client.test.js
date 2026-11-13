import { ApiError, absUrl, clearToken, getApiBase, messageFromDetail, request, setApiBase, setToken, setUnauthorizedHandler } from '../../src/api/client.js';

const json = (status, body) => new Response(body === undefined ? null : JSON.stringify(body), { status });

beforeEach(() => { localStorage.clear(); setApiBase('http://api.test'); setUnauthorizedHandler(null); });
afterEach(() => { vi.restoreAllMocks(); });

describe('base url', () => {
  it('strips trailing slashes and can be overridden at runtime', () => {
    setApiBase('https://abc.trycloudflare.com///');
    expect(getApiBase()).toBe('https://abc.trycloudflare.com');
  });
  it('absUrl makes backend-relative file URLs absolute and leaves absolute ones alone', () => {
    expect(absUrl('/projects/1/files/x.splat?t=abc')).toBe('http://api.test/projects/1/files/x.splat?t=abc');
    expect(absUrl('https://cdn.example/x')).toBe('https://cdn.example/x');
    expect(absUrl(null)).toBeNull();
  });
});

describe('request', () => {
  it('sends the bearer token and JSON body', async () => {
    setToken('tok123');
    const spy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(json(200, { ok: 1 }));
    await request('POST', '/projects', { body: { name: 'A' } });
    const [url, init] = spy.mock.calls[0];
    expect(url).toBe('http://api.test/projects');
    expect(init.headers.Authorization).toBe('Bearer tok123');
    expect(init.headers['Content-Type']).toBe('application/json');
    expect(init.body).toBe('{"name":"A"}');
  });
  it('does not send a token to auth routes', async () => {
    setToken('tok123');
    const spy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(json(200, {}));
    await request('POST', '/auth/login', { body: {}, auth: false });
    expect(spy.mock.calls[0][1].headers.Authorization).toBeUndefined();
  });
  it('returns null for 204', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(null, { status: 204 }));
    expect(await request('DELETE', '/projects/1')).toBeNull();
  });
  it('turns {detail: "text"} into ApiError with that message', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(json(409, { detail: 'A task is already running' }));
    await expect(request('POST', '/x')).rejects.toMatchObject({ name: 'ApiError', status: 409, message: 'A task is already running' });
  });
  it('makes FastAPI 422 validation lists readable', () => {
    const msg = messageFromDetail([{ loc: ['body', 'password'], msg: 'String should have at least 8 characters', type: 'x' }]);
    expect(msg).toBe('password: String should have at least 8 characters');
  });
  it('401 with a token clears the session via the handler', async () => {
    setToken('old');
    const handler = vi.fn(); setUnauthorizedHandler(handler);
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(json(401, { detail: 'Invalid token' }));
    await expect(request('GET', '/projects')).rejects.toBeInstanceOf(ApiError);
    expect(handler).toHaveBeenCalledOnce();
  });
  it('wrong password on /auth/login (no token yet) does not trigger logout handling', async () => {
    const handler = vi.fn(); setUnauthorizedHandler(handler);
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(json(401, { detail: 'Incorrect email or password' }));
    await expect(request('POST', '/auth/login', { body: {}, auth: false })).rejects.toThrow('Incorrect email or password');
    expect(handler).not.toHaveBeenCalled();
  });
  it('explains an unreachable server in plain words (status 0)', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'));
    await expect(request('GET', '/projects')).rejects.toMatchObject({ status: 0, message: expect.stringContaining('http://api.test') });
  });
  it('lets aborts through untouched', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(Object.assign(new Error('aborted'), { name: 'AbortError' }));
    await expect(request('GET', '/x')).rejects.toMatchObject({ name: 'AbortError' });
  });
});
afterAll(() => clearToken());
