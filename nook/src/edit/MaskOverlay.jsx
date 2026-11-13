import { useEffect, useRef, useState } from 'react';
import { absUrl } from '../api/client.js';
import { maskToOverlay } from '../lib/mask.js';

/**
 * Draws a mask PNG as a translucent yellow layer on top of its frame.
 * Reads the mask's pixels via a canvas; if the browser refuses (CORS), it falls back to a plain
 * blended <img>, which looks slightly different but still shows where the mask is.
 */
export function MaskOverlay({ url }) {
  const ref = useRef(null);
  const [fallback, setFallback] = useState(false);
  const full = absUrl(url);

  useEffect(() => {
    if (!full) return undefined;
    setFallback(false);
    let cancelled = false;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (cancelled || !ref.current) return;
      try {
        const c = ref.current;
        c.width = img.naturalWidth; c.height = img.naturalHeight;
        const ctx = c.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        const src = ctx.getImageData(0, 0, c.width, c.height);
        ctx.putImageData(new ImageData(maskToOverlay(src.data), c.width, c.height), 0, 0);
      } catch { setFallback(true); }
    };
    img.onerror = () => { if (!cancelled) setFallback(true); };
    img.src = full;
    return () => { cancelled = true; };
  }, [full]);

  if (!full) return null;
  if (fallback) return <img className="mask-layer mask-fallback" src={full} alt="" />;
  return <canvas ref={ref} className="mask-layer" aria-hidden="true" />;
}
