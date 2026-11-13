import { useState } from 'react';
import { absUrl } from '../api/client.js';

/** Step 5: the first edited frame, in a few seeds. Accept one, or ask for new ones. */
export function Candidates({ candidates, onAccept, onRegenerate, busy }) {
  const [seed, setSeed] = useState(candidates[0]?.seed ?? null);
  return (
    <div>
      <p className="muted">Pick the version you like best. Every other frame will be matched to it.</p>
      <ul className="cand-grid" role="radiogroup" aria-label="First edited frame">
        {candidates.map((c) => (
          <li key={c.seed}>
            <label className={`cand ${seed === c.seed ? 'on' : ''}`}>
              <input type="radio" name="seed" checked={seed === c.seed} onChange={() => setSeed(c.seed)} />
              <img src={absUrl(c.url)} alt={`Option with seed ${c.seed}`} />
              <span className="small">Option {c.seed}{c.seconds != null ? ` · ${c.seconds.toFixed(1)} s` : ''}</span>
            </label>
          </li>
        ))}
      </ul>
      <div className="row">
        <button className="btn btn-primary" disabled={busy || seed == null} onClick={() => onAccept(seed)}>Use this one for every frame</button>
        <button className="btn btn-quiet" disabled={busy} onClick={onRegenerate}>Try other seeds</button>
      </div>
    </div>
  );
}
