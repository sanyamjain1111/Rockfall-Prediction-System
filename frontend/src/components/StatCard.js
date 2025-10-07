import React from 'react';

export default function StatCard({ label, value, icon: Icon, color }) {
  return (
    <div className="stat-card">
      {Icon && <div className="stat-icon" style={{color}}><Icon size={20} /></div>}
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
