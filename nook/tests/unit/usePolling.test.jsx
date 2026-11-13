import { act, renderHook } from '@testing-library/react';
import { usePolling } from '../../src/hooks/usePolling.js';

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());
const flush = async (ms) => { await act(async () => { await vi.advanceTimersByTimeAsync(ms); }); };

describe('usePolling', () => {
  it('polls on an interval and stops when stopWhen is true', async () => {
    const seq = [{ s: 'running' }, { s: 'running' }, { s: 'done' }, { s: 'never' }];
    const fetcher = vi.fn(() => Promise.resolve(seq.shift()));
    const { result } = renderHook(() => usePolling(fetcher, { interval: 1000, stopWhen: (d) => d.s === 'done' }));
    await flush(0); expect(result.current.data.s).toBe('running');
    await flush(1000); await flush(1000);
    expect(result.current.data.s).toBe('done');
    await flush(5000);
    expect(fetcher).toHaveBeenCalledTimes(3);
  });
  it('keeps trying after a network error and reports it', async () => {
    const fetcher = vi.fn()
      .mockRejectedValueOnce(Object.assign(new Error('down'), { status: 0 }))
      .mockResolvedValue({ s: 'running' });
    const { result } = renderHook(() => usePolling(fetcher, { interval: 1000 }));
    await flush(0); expect(result.current.error.message).toBe('down');
    await flush(1000); expect(result.current.data.s).toBe('running'); expect(result.current.error).toBeNull();
  });
  it('gives up on 404 / 401 (retrying cannot help)', async () => {
    const fetcher = vi.fn().mockRejectedValue(Object.assign(new Error('gone'), { status: 404 }));
    renderHook(() => usePolling(fetcher, { interval: 1000 }));
    await flush(0); await flush(5000);
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it('stops polling on unmount', async () => {
    const fetcher = vi.fn(() => Promise.resolve({}));
    const { unmount } = renderHook(() => usePolling(fetcher, { interval: 1000 }));
    await flush(0); unmount(); await flush(5000);
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it('does not poll when disabled', async () => {
    const fetcher = vi.fn(() => Promise.resolve({}));
    renderHook(() => usePolling(fetcher, { interval: 1000, enabled: false }));
    await flush(3000); expect(fetcher).not.toHaveBeenCalled();
  });
  it('refresh() fetches immediately', async () => {
    const fetcher = vi.fn(() => Promise.resolve({ n: 1 }));
    const { result } = renderHook(() => usePolling(fetcher, { interval: 60000, stopWhen: () => true }));
    await flush(0); expect(fetcher).toHaveBeenCalledTimes(1);
    await act(async () => { await result.current.refresh(); });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
});
