import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Call `fetcher` every `interval` ms until `stopWhen(data)` is true (or enabled=false / unmounted).
 * Network hiccups do not stop the loop: the error is exposed and polling continues,
 * because a tunnel that blinks for a second should not kill the progress page.
 */
export function usePolling(fetcher, { interval = 2500, stopWhen, enabled = true, deps = [] } = {}) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const fetcherRef = useRef(fetcher);
  const stopRef = useRef(stopWhen);
  fetcherRef.current = fetcher;
  stopRef.current = stopWhen;
  const timer = useRef(null);
  const alive = useRef(true);
  const tick = useRef(null);

  useEffect(() => {
    alive.current = true;
    if (!enabled) { setState((s) => ({ ...s, loading: false })); return undefined; }
    let cancelled = false;

    const run = async () => {
      try {
        const data = await fetcherRef.current();
        if (cancelled) return;
        setState({ data, error: null, loading: false });
        if (stopRef.current?.(data)) return;
      } catch (error) {
        if (cancelled) return;
        setState((s) => ({ ...s, error, loading: false }));
        if (error?.status === 401 || error?.status === 404) return; // retrying will not help
      }
      timer.current = setTimeout(run, interval);
    };
    tick.current = run;
    run();
    return () => { cancelled = true; alive.current = false; clearTimeout(timer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, interval, ...deps]);

  /** Fetch right now (e.g. after the user pressed a button) and resume the loop. */
  const refresh = useCallback(() => {
    clearTimeout(timer.current);
    return tick.current?.();
  }, []);

  return { ...state, refresh };
}
