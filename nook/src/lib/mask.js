/**
 * Convert a mask PNG's pixels into an RGBA overlay. Masks may be white-on-black (no alpha)
 * or white-on-transparent; handle both so the UI does not depend on the writer's choice.
 * `rgb` is the overlay colour, `strength` its max opacity (0-255).
 */
export function maskToOverlay(data, rgb = [245, 184, 61], strength = 150) {
  const out = new Uint8ClampedArray(data.length);
  let hasAlpha = false;
  for (let i = 3; i < data.length; i += 4) if (data[i] < 255) { hasAlpha = true; break; }
  for (let i = 0; i < data.length; i += 4) {
    const v = hasAlpha ? data[i + 3] : (data[i] + data[i + 1] + data[i + 2]) / 3;
    const on = v > 127;
    out[i] = rgb[0]; out[i + 1] = rgb[1]; out[i + 2] = rgb[2];
    out[i + 3] = on ? strength : 0;
  }
  return out;
}
