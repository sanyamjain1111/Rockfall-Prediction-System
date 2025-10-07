import React from 'react';
import { formatTimestamp, truncateText } from '../utils/formatters';
import { getAlertLevelColor } from '../utils/colors';

export default function AlertItem({ alert, onAcknowledge }) {
  const color = getAlertLevelColor(alert.level);
  const cls = alert.level === 3 ? 'warning' : (alert.level === 2 ? 'advisory' : 'watch');
  return (
    <div className={`alert-item ${cls}`} style={{ borderLeftColor: color }}>
      <div className="alert-level-badge" style={{ background: color }}>{alert.level_name}</div>
      <div className="alert-message">{truncateText(alert.message || '', 220)}</div>
      <div className="alert-time">{formatTimestamp(alert.timestamp)}</div>
      {!alert.acknowledged && (
        <button className="alert-ack-button" onClick={onAcknowledge}>Acknowledge</button>
      )}
    </div>
  );
}
