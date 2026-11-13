import { useState } from 'react';
import { absUrl } from '../api/client.js';
import { clientToFrame, frameToPercent } from '../lib/clicks.js';
import { MaskOverlay } from './MaskOverlay.jsx';

/**
 * A frame you can click on. Click = "this is the object" (green dot).
 * Right-click, Shift-click, or the "Not this" toggle = "not part of it" (red dot).
 * Coordinates are reported in the frame's own pixels, exactly what the backend expects.
 */
export function ClickCanvas({ url, width, height, clicks, onChange, maskUrl, disabled = false, allowNegative = true, max = 30 }) {
  const [natural, setNatural] = useState(null);
  const [negativeMode, setNegativeMode] = useState(false);
  const W = width || natural?.w; const H = height || natural?.h;

  const add = (e, positive) => {
    if (disabled || !W || !H || clicks.length >= max) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const { x, y } = clientToFrame(rect, W, H, e.clientX, e.clientY);
    onChange([...clicks, { x, y, positive }]);
  };

  return (
    <div className="clickcanvas">
      <div
        className={`frame-stage ${disabled ? 'is-disabled' : ''}`}
        style={W && H ? { aspectRatio: `${W} / ${H}` } : undefined}
        onClick={(e) => add(e, !(negativeMode || e.shiftKey))}
        onContextMenu={(e) => { if (!allowNegative) return; e.preventDefault(); add(e, false); }}
        role="img"
        aria-label="Reference frame. Click the object."
      >
        <img src={absUrl(url)} alt="" draggable={false} onLoad={(e) => setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
        {maskUrl && <MaskOverlay url={maskUrl} />}
        {W && H && clicks.map((c, i) => (
          <span key={i} className={`dot ${c.positive === false ? 'neg' : 'pos'}`} style={frameToPercent(c.x, c.y, W, H)} title={c.positive === false ? 'Not this' : 'This'} />
        ))}
      </div>
      <div className="row tools">
        {allowNegative && (
          <button type="button" className={`btn btn-quiet ${negativeMode ? 'on' : ''}`} aria-pressed={negativeMode} onClick={() => setNegativeMode(!negativeMode)}>
            {negativeMode ? 'Marking “not this”' : 'Mark “not this”'}
          </button>
        )}
        <button type="button" className="btn btn-quiet" disabled={!clicks.length || disabled} onClick={() => onChange(clicks.slice(0, -1))}>Undo</button>
        <button type="button" className="btn btn-quiet" disabled={!clicks.length || disabled} onClick={() => onChange([])}>Clear</button>
        <span className="muted small">{clicks.length ? `${clicks.length} point${clicks.length > 1 ? 's' : ''}` : 'Click the object'}</span>
      </div>
    </div>
  );
}
