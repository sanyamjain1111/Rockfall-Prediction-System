export function formatTimestamp(ts) {
  const d = typeof ts === 'string' ? new Date(ts) : ts;
  if (!d || isNaN(d.getTime())) return '—';
  const diff = Math.floor((Date.now() - d.getTime()) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff/3600)}h ago`;
  return d.toLocaleString();
}
export function truncateText(text, maxLen) { if (!text) return ''; return text.length<=maxLen?text:(text.slice(0,maxLen-1)+'…'); }
