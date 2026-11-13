import { useState } from 'react';
import { absUrl } from '../api/client.js';
import { MaskOverlay } from './MaskOverlay.jsx';
import { ClickCanvas } from './ClickCanvas.jsx';
import { Modal } from '../components/ui.jsx';

/**
 * Step 4: every frame with its mask. Wrong frame → "Remove". Frame the tracker missed → "Mark it"
 * (the user-feedback fallback the spec allows: the user points at the object, SAM does the rest).
 */
export function TrackingGrid({ frames, onRemove, onRestore, onMark, busy }) {
  const [marking, setMarking] = useState(null); // frame being marked
  const [clicks, setClicks] = useState([]);
  const [filter, setFilter] = useState('all');

  const present = frames.filter((f) => f.present).length;
  const shown = frames.filter((f) => filter === 'all' || (filter === 'found' ? f.present : !f.present));

  const submitMark = () => { onMark(marking.name, clicks); setMarking(null); setClicks([]); };

  return (
    <div>
      <p><strong>{present}</strong> of {frames.length} frames show the object.
        {' '}<span className="muted">Remove wrong ones; mark any it missed.</span></p>
      <div className="seg" role="group" aria-label="Filter frames">
        {[['all', 'All'], ['found', 'Found'], ['missing', 'Not found']].map(([k, label]) => (
          <button key={k} type="button" className={filter === k ? 'on' : ''} aria-pressed={filter === k} onClick={() => setFilter(k)}>{label}</button>
        ))}
      </div>
      <ul className="frame-grid">
        {shown.map((f) => {
          const state = f.removed ? 'removed' : f.present ? 'present' : 'missing';
          return (
            <li key={f.name} className={`tile tile-${state}`}>
              <div className="thumb">
                <img src={absUrl(f.url)} alt={f.name} loading="lazy" />
                {f.present && f.mask_url && <MaskOverlay url={f.mask_url} />}
              </div>
              <div className="tile-foot">
                <span className="small">{f.name.replace(/\.\w+$/, '')}{f.source === 'user' ? ' · by you' : ''}</span>
                {state === 'present' && <button type="button" className="linklike" disabled={busy} onClick={() => onRemove(f.name)}>Remove</button>}
                {state === 'removed' && <button type="button" className="linklike" disabled={busy} onClick={() => onRestore(f.name)}>Put back</button>}
                {state === 'missing' && <button type="button" className="linklike" disabled={busy} onClick={() => { setMarking(f); setClicks([]); }}>Mark it</button>}
              </div>
            </li>
          );
        })}
      </ul>
      {marking && (
        <Modal title={`Where is the object in ${marking.name}?`} onClose={() => setMarking(null)}>
          <p className="muted">Click it once or twice. Skip clicking to let Nook search the frame on its own.</p>
          <ClickCanvas url={marking.url} clicks={clicks} onChange={setClicks} allowNegative={false} />
          <div className="row end">
            <button className="btn btn-quiet" onClick={() => setMarking(null)}>Cancel</button>
            <button className="btn btn-primary" onClick={submitMark}>Add this frame</button>
          </div>
        </Modal>
      )}
    </div>
  );
}
