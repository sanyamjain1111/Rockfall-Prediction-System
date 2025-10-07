import React, { useMemo, useState } from 'react';
import AlertItem from './AlertItem';

export default function AlertPanel({ alerts, onAcknowledge, onClearAll, unacknowledgedCount }) {
  const [filter, setFilter] = useState('all');
  const filtered = useMemo(() => {
    if (filter === 'active') return alerts.filter(a => !a.acknowledged);
    if (filter === 'acknowledged') return alerts.filter(a => a.acknowledged);
    return alerts;
  }, [alerts, filter]);

  return (
    <aside className="alert-panel">
      <div className="alert-header">
        <h3>Alerts</h3>
        <span className="alert-count">{unacknowledgedCount}</span>
      </div>
      <div className="alert-controls">
        <select value={filter} onChange={e => setFilter(e.target.value)}>
          <option value="all">All</option>
          <option value="active">Active</option>
          <option value="acknowledged">Acknowledged</option>
        </select>
        <button className="alert-ack-button" onClick={onClearAll}>Clear All</button>
      </div>
      <div className="alert-list">
        {filtered.length === 0 && <div className="empty-state">No alerts</div>}
        {filtered.map(a => (
          <AlertItem key={a.alert_id} alert={a} onAcknowledge={() => onAcknowledge(a.alert_id)} />
        ))}
      </div>
    </aside>
  );
}
