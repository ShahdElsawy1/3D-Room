import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { RoomEngine, withAbsoluteUrls } from './RoomEngine.js';

/**
 * React wrapper around RoomEngine.
 *
 * props
 *   scene        backend SceneOut (relative URLs are fine)
 *   mode         'orbit' | 'fly'
 *   speed        fly speed multiplier (0.25 – 4)
 *   pointsOnly   force the point-cloud fallback
 *   onPick       hook: user clicked at screen position (x, y)  → see RoomEngine.pickAt
 *   onStatus     {type:'ready'|'path'|'mode', ...}
 *
 * ref methods (the "hooks" for Week 2)
 *   flyToCamera(cam) · setCameraPose(cam) · playCameraPath(cams, opts) · stopCameraPath() · resetView()
 */
export const SplatViewer = forwardRef(function SplatViewer({ scene, mode = 'orbit', speed = 1, pointsOnly = false, onPick, onStatus }, ref) {
  const hostRef = useRef(null);
  const engineRef = useRef(null);
  const pickRef = useRef(onPick);
  const statusRef = useRef(onStatus);
  pickRef.current = onPick; statusRef.current = onStatus;
  const [state, setState] = useState({ loading: true, progress: 0, error: null, kind: null, reason: null });
  const loadedKey = useRef(null);

  // create / destroy the engine once
  useEffect(() => {
    const engine = new RoomEngine(hostRef.current, {
      onPick: (p) => pickRef.current?.(p),
      onStatus: (s) => statusRef.current?.(s),
    });
    engineRef.current = engine;
    hostRef.current.__engine = engine; // read by browser tests (tests/e2e); harmless otherwise
    return () => { engine.dispose(); engineRef.current = null; loadedKey.current = null; };
  }, []);

  // load (or swap) the scene whenever the splat file or the rendering choice changes
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !scene) return undefined;
    const key = `${scene.splat.url}|${pointsOnly}`;
    if (loadedKey.current === key) return undefined;
    let cancelled = false;
    const keepCamera = loadedKey.current !== null; // second load = before/after toggle: don't move the camera
    setState((s) => ({ ...s, loading: true, progress: 0, error: null }));
    engine.loadScene(withAbsoluteUrls(scene), {
      preferPoints: pointsOnly, keepCamera,
      onProgress: (p) => { if (!cancelled) setState((s) => ({ ...s, progress: p })); },
    }).then(({ kind, fallbackReason }) => {
      if (cancelled) return;
      loadedKey.current = key;
      setState({ loading: false, progress: 1, error: null, kind, reason: fallbackReason || null });
    }).catch((err) => {
      if (!cancelled) setState({ loading: false, progress: 0, error: err.message || String(err), kind: null, reason: null });
    });
    return () => { cancelled = true; };
  }, [scene, pointsOnly]);

  useEffect(() => { engineRef.current?.setMode(mode); }, [mode, state.kind]);
  useEffect(() => { engineRef.current?.setSpeed(speed); }, [speed]);

  useImperativeHandle(ref, () => ({
    flyToCamera: (cam, o) => engineRef.current?.flyToCamera(cam, o),
    setCameraPose: (cam) => engineRef.current?.setCameraPose(cam),
    playCameraPath: (cams, o) => engineRef.current?.playCameraPath(cams, o),
    stopCameraPath: () => engineRef.current?.stopCameraPath(),
    resetView: () => engineRef.current?.resetView(),
    focus: () => engineRef.current?.renderer.domElement.focus(),
  }), []);

  return (
    <div className="viewer-host" ref={hostRef}>
      {state.loading && (
        <div className="viewer-overlay" role="status">
          <div className="viewer-loader">
            <div className="bar"><span style={{ width: `${Math.round(state.progress * 100)}%` }} /></div>
            <p>{state.progress > 0 && state.progress < 1 ? `Unpacking the room… ${Math.round(state.progress * 100)}%` : 'Getting the room ready…'}</p>
          </div>
        </div>
      )}
      {state.error && (
        <div className="viewer-overlay" role="alert">
          <div className="viewer-loader error"><strong>The room would not open.</strong><p>{state.error}</p></div>
        </div>
      )}
      {!state.loading && state.kind === 'points' && (
        <div className="viewer-banner" role="note">
          Showing the sparse point cloud instead of the full splat.
          <span className="muted"> Known limitation{state.reason && state.reason !== 'requested' ? `: ${state.reason}` : ''}.</span>
        </div>
      )}
    </div>
  );
});
