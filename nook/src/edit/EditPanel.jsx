import { useEffect, useRef, useState } from 'react';
import { edits as editsApi, projects as projectsApi } from '../api/endpoints.js';
import { usePolling } from '../hooks/usePolling.js';
import { ErrorNote, MiniProgress, Spinner } from '../components/ui.jsx';
import { absUrl } from '../api/client.js';
import { describeEditStatus } from '../lib/taskText.js';
import { STEPS, editPhase, isEditBusy, isEditFailed, stepIndex } from '../lib/editFlow.js';
import { ClickCanvas } from './ClickCanvas.jsx';
import { TrackingGrid } from './TrackingGrid.jsx';
import { Candidates } from './Candidates.jsx';

/**
 * "Edit object" mode (spec §4.5, steps 1-6). One panel, one step at a time:
 *   pick a frame (or click in the room) → mask preview + prompt → tracking grid →
 *   first-frame candidates → rebuild → before/after in the viewer.
 */
export function EditPanel({ projectId, cameras, viewerRef, pick, editId, setEditId, onResult, onClose }) {
  const [frames, setFrames] = useState(null);
  const [framesError, setFramesError] = useState(null);
  const [refFrame, setRefFrame] = useState(null);
  const [clicks, setClicks] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [error, setError] = useState(null);
  const [acting, setActing] = useState(false);
  const [editFrames, setEditFrames] = useState(null);
  const [candidates, setCandidates] = useState(null);
  const [rebuildMode, setRebuildMode] = useState('finetune');
  const [pickNote, setPickNote] = useState(null);
  const announced = useRef(null);

  useEffect(() => { projectsApi.frames(projectId).then(setFrames).catch(setFramesError); }, [projectId]);

  const { data: polled, error: pollError, refresh } = usePolling(() => editsApi.get(editId), {
    enabled: Boolean(editId), interval: 2000, stopWhen: (e) => !isEditBusy(e), deps: [editId],
  });
  const edit = editId && polled?.id === editId ? polled : null;
  const phase = editPhase(edit);
  const busy = isEditBusy(edit);
  const failed = isEditFailed(edit);
  const frameOf = (name) => frames?.find((f) => f.name === name);

  // resume: when an existing edit is opened (e.g. from ?edit=), restore its clicks and prompt
  useEffect(() => {
    if (!edit || announced.current === `load:${edit.id}`) return;
    announced.current = `load:${edit.id}`;
    setClicks(edit.clicks?.map((c) => ({ x: c.x, y: c.y, positive: c.positive !== false })) || []);
    if (edit.prompt) setPrompt(edit.prompt);
  }, [edit]);

  // a click in the 3D room → the best training frame + the pixel to start from (spec §4.1 option 2).
  // Waits for the frame list (the click can arrive first) and handles each click only once.
  const handledPick = useRef(null);
  useEffect(() => {
    if (!pick || editId || handledPick.current === pick) return;
    if (!pick.hit || !pick.camera) {
      handledPick.current = pick;
      setPickNote(pick.hit ? 'No training photo saw that spot clearly. Try clicking nearer the middle of the object.' : 'That click did not land on anything. Click directly on the object.');
      return;
    }
    if (!frames) return; // effect re-runs when frames arrive
    handledPick.current = pick;
    const f = frameOf(pick.camera.img_name) || frames.find((x) => x.camera_id === pick.camera.id);
    if (!f) { setPickNote('Found the spot, but could not match it to a frame.'); return; }
    setPickNote(null);
    setRefFrame(f);
    setClicks([{ x: pick.camera.x, y: pick.camera.y, positive: true }]);
    const cam = cameras.find((c) => c.id === pick.camera.id);
    if (cam) viewerRef.current?.flyToCamera(cam);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pick, frames]);

  // load the tracking grid / candidates when their step has settled
  useEffect(() => {
    if (!edit || busy) return;
    if (phase === 'track' && !failed) editsApi.frames(edit.id).then(setEditFrames).catch(setError);
    if (phase === 'preview' && edit.status === 'preview_ready') editsApi.candidates(edit.id).then(setCandidates).catch(setError);
  }, [edit?.id, edit?.status, edit?.updated_at, busy, phase, failed]); // eslint-disable-line react-hooks/exhaustive-deps

  // tell the viewer when the rebuilt scene exists (once)
  useEffect(() => {
    if (edit?.status === 'done' && edit.scene_version && announced.current !== `done:${edit.id}`) {
      announced.current = `done:${edit.id}`;
      onResult(edit.scene_version);
    }
  }, [edit?.status, edit?.scene_version]); // eslint-disable-line react-hooks/exhaustive-deps

  const act = async (fn) => {
    setActing(true); setError(null);
    try { await fn(); refresh(); } catch (e) { setError(e); } finally { setActing(false); }
  };

  const startEdit = () => act(async () => {
    const e = await editsApi.create(projectId, refFrame.name, clicks);
    setEditId(e.id);
  });
  const updateMask = () => act(() => editsApi.clicks(edit.id, clicks));
  const startTracking = () => act(() => editsApi.track(edit.id));
  const startPreview = () => act(async () => { setCandidates(null); await editsApi.preview(edit.id, prompt.trim(), 3); });
  const accept = (seed) => act(() => editsApi.accept(edit.id, seed));
  const rebuild = () => act(() => editsApi.rebuild(edit.id, rebuildMode));
  const patch = (body) => act(async () => { await editsApi.patchFrames(edit.id, body); setEditFrames(await editsApi.frames(edit.id)); });

  const discard = async () => {
    if (edit) {
      try { await editsApi.remove(edit.id); } catch (e) { setError(e); return; }
    }
    setEditId(null); setRefFrame(null); setClicks([]); setEditFrames(null); setCandidates(null); announced.current = null; setPickNote(null);
  };

  const retry = () => {
    const type = edit.current_task?.type;
    if (type === 'segment') return phase === 'mask' ? updateMask() : refresh();
    if (type === 'track') return startTracking();
    if (type === 'edit_preview') return startPreview();
    if (type === 'edit_apply') return accept(edit.chosen_seed);
    if (type === 'rebuild') return rebuild();
    return refresh();
  };

  const refFrameObj = edit ? frameOf(edit.ref_frame) : refFrame;
  const current = stepIndex(edit);

  return (
    <aside className="editpanel" aria-label="Edit object">
      <div className="editpanel-head">
        <h2>Edit an object</h2>
        <button className="btn btn-quiet" onClick={onClose} aria-label="Close edit panel">Close</button>
      </div>

      <ol className="stepper" aria-label="Edit steps">
        {STEPS.map((s, i) => (
          <li key={s.key} className={i < current ? 'past' : i === current ? 'now' : ''} aria-current={i === current ? 'step' : undefined}>{s.label}</li>
        ))}
      </ol>

      <ErrorNote error={error || framesError} />
      {pollError && edit && <div className="note note-warn">Lost touch with the server. Still trying…</div>}

      {/* ---------------------------------------------------------- 1. pick */}
      {phase === 'pick' && (
        <div className="stack tight">
          {!frames && !framesError && <p><Spinner /> Loading frames…</p>}
          {frames && !refFrame && (
            <>
              <p>Click the object in the room, or choose a frame that shows it.</p>
              {pickNote && <div className="note note-warn">{pickNote}</div>}
              <ul className="frame-grid small-tiles">
                {frames.map((f) => (
                  <li key={f.name} className="tile">
                    <button type="button" className="thumb as-button" onClick={() => { setRefFrame(f); setClicks([]); const cam = cameras.find((c) => c.id === f.camera_id); if (cam) viewerRef.current?.flyToCamera(cam); }} aria-label={`Use frame ${f.name}`}>
                      <img src={absUrl(f.url)} alt="" loading="lazy" />
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
          {refFrame && (
            <>
              <p>Click the object. Add more points to be exact.</p>
              <ClickCanvas url={refFrame.url} width={refFrame.width} height={refFrame.height} clicks={clicks} onChange={setClicks} disabled={acting} />
              <div className="row">
                <button className="btn btn-primary" disabled={!clicks.some((c) => c.positive) || acting} onClick={startEdit}>{acting ? <Spinner /> : 'Find the object'}</button>
                <button className="btn btn-quiet" onClick={() => { setRefFrame(null); setClicks([]); }}>Different frame</button>
              </div>
            </>
          )}
        </div>
      )}

      {/* ---------------------------------------------------------- 2/3. mask + prompt */}
      {phase === 'mask' && edit && (
        <div className="stack tight">
          {busy ? (
            <MiniProgress task={edit.current_task} fallback={describeEditStatus(edit)} />
          ) : failed ? null : (
            <>
              <p>Here is what Nook found. Add or remove points until the yellow covers just the object.</p>
              {refFrameObj && (
                <ClickCanvas url={refFrameObj.url} width={refFrameObj.width} height={refFrameObj.height} clicks={clicks} onChange={setClicks} maskUrl={edit.ref_mask_url} disabled={acting} />
              )}
              <div className="row">
                <button className="btn btn-quiet" disabled={acting || !clicks.length} onClick={updateMask}>Update mask</button>
              </div>
              <label>What should it become?
                <textarea rows={2} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="make the sofa dark green leather" maxLength={1000} />
              </label>
              <button className="btn btn-primary" disabled={acting || !prompt.trim()} onClick={startTracking}>{acting ? <Spinner /> : 'Edit'}</button>
            </>
          )}
        </div>
      )}

      {/* ---------------------------------------------------------- 4. tracking */}
      {phase === 'track' && edit && (
        <div className="stack tight">
          {busy ? (
            <MiniProgress task={edit.current_task} fallback={describeEditStatus(edit)} />
          ) : failed ? null : editFrames ? (
            <>
              <TrackingGrid
                frames={editFrames} busy={acting}
                onRemove={(n) => patch({ remove: [n] })}
                onRestore={(n) => patch({ restore: [n] })}
                onMark={(n, c) => act(async () => { await editsApi.patchFrames(edit.id, { add: [{ name: n, clicks: c.map(({ x, y }) => ({ x, y })) }] }); })}
              />
              <button className="btn btn-primary" disabled={acting || !prompt.trim() || !editFrames.some((f) => f.present)} onClick={startPreview}>
                {acting ? <Spinner /> : 'Looks right — paint the first frame'}
              </button>
              {!prompt.trim() && <label>What should it become?<textarea rows={2} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="make the sofa dark green leather" /></label>}
            </>
          ) : <p><Spinner /> Loading frames…</p>}
        </div>
      )}

      {/* ---------------------------------------------------------- 5. diffusion preview + apply */}
      {phase === 'preview' && edit && (
        <div className="stack tight">
          {busy ? <MiniProgress task={edit.current_task} fallback="Painting the first frame…" />
            : failed ? null
              : candidates ? <Candidates key={candidates.map((c) => c.seed).join()} candidates={candidates} busy={acting} onAccept={accept} onRegenerate={startPreview} />
                : <p><Spinner /> Loading options…</p>}
        </div>
      )}
      {phase === 'apply' && edit && (
        <div className="stack tight">
          {edit.status === 'edited' ? (
            <>
              <p>Every frame is edited. Now Nook rebuilds the 3D room from them.</p>
              <fieldset className="radios">
                <legend>How</legend>
                <label><input type="radio" checked={rebuildMode === 'finetune'} onChange={() => setRebuildMode('finetune')} /> Quick: fine-tune the existing room</label>
                <label><input type="radio" checked={rebuildMode === 'scratch'} onChange={() => setRebuildMode('scratch')} /> Thorough: train again from scratch</label>
              </fieldset>
              <button className="btn btn-primary" disabled={acting} onClick={rebuild}>{acting ? <Spinner /> : 'Rebuild the room'}</button>
            </>
          ) : busy ? <MiniProgress task={edit.current_task} fallback="Editing the other frames…" /> : null}
        </div>
      )}

      {/* ---------------------------------------------------------- 6. rebuild */}
      {phase === 'rebuild' && edit && (
        <div className="stack tight">
          {busy && <MiniProgress task={edit.current_task} fallback="Rebuilding the 3D room…" />}
          {edit.status === 'done' && (
            <div className="note note-good">
              Done. Use the <strong>Before / After</strong> switch above the viewer to compare.
              <div className="row"><button className="btn btn-primary" onClick={discardAndStartNew}>Edit another object</button></div>
            </div>
          )}
        </div>
      )}

      {failed && edit && (
        <div className="note note-bad" role="alert">
          <strong>{describeEditStatus(edit)}</strong>
          <div className="row"><button className="btn btn-primary" disabled={acting} onClick={retry}>Try this step again</button></div>
        </div>
      )}

      {edit && !busy && edit.status !== 'done' && (
        <button className="linklike discard" onClick={discard}>Discard this edit and start over</button>
      )}
    </aside>
  );

  function discardAndStartNew() {
    // keep the finished edit in history; just clear the wizard
    setEditId(null); setRefFrame(null); setClicks([]); setEditFrames(null); setCandidates(null); setPrompt(''); announced.current = null; setPickNote(null);
  }
}
