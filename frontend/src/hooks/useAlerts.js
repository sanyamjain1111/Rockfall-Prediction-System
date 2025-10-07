import { useEffect, useMemo, useState } from 'react';

export default function useAlerts() {
  const [alerts, setAlerts] = useState([]);
  const addAlert = (alert) => setAlerts(prev => [alert, ...prev]);
  const acknowledgeAlert = (id) => setAlerts(prev => prev.map(a => a.alert_id === id ? { ...a, acknowledged: true } : a));
  const clearAll = () => setAlerts([]);

  useEffect(() => {
    if (alerts.length === 0) return;
    const newest = alerts[0];
    if (newest.level === 3 && process.env.REACT_APP_ALERT_SOUND_ENABLED === 'true') {
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.connect(g); g.connect(ctx.destination);
        o.type = 'sine'; o.frequency.value = 880;
        g.gain.setValueAtTime(0.001, ctx.currentTime);
        g.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.05);
        o.start(); setTimeout(() => { g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.2); o.stop(); }, 600);
      } catch {}
    }
  }, [alerts]);

  const unacknowledgedCount = useMemo(() => alerts.filter(a => !a.acknowledged).length, [alerts]);

  return { alerts, addAlert, acknowledgeAlert, clearAll, unacknowledgedCount };
}
