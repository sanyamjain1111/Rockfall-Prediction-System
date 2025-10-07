export function getRiskColor(p) {
  if (p < 0.3) return '#10b981';
  if (p < 0.5) return '#22c55e';
  if (p < 0.7) return '#f59e0b';
  if (p < 0.85) return '#f97316';
  return '#ef4444';
}
export function getAlertLevelColor(level) {
  if (level === 3) return '#ef4444';
  if (level === 2) return '#f59e0b';
  return '#3b82f6';
}
export function hexToRgba(hex, a=1.0){
  const c = hex.replace('#',''); const n=parseInt(c,16);
  return `rgba(${(n>>16)&255}, ${(n>>8)&255}, ${n&255}, ${a})`;
}
