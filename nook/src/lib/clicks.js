/** Mouse position on a displayed image → pixel in the frame's own resolution (origin top-left, clamped). */
export function clientToFrame(rect, frameW, frameH, clientX, clientY) {
  const nx = (clientX - rect.left) / rect.width;
  const ny = (clientY - rect.top) / rect.height;
  const x = Math.min(frameW - 1, Math.max(0, Math.round(nx * frameW)));
  const y = Math.min(frameH - 1, Math.max(0, Math.round(ny * frameH)));
  return { x, y };
}

/** A frame pixel → percentage position for drawing a dot on the displayed image. */
export function frameToPercent(x, y, frameW, frameH) {
  return { left: `${(x / frameW) * 100}%`, top: `${(y / frameH) * 100}%` };
}
