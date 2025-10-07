import React from 'react';
import StatCard from './StatCard';
import { Activity, AlertTriangle, CheckCircle2, Info } from 'lucide-react';

export default function Dashboard({ stats }) {
  return (
    <section className="dashboard-section">
      <div className="stats-grid">
        <StatCard label="Total Zones" value={stats?.total_zones ?? '—'} icon={Activity} color="#3b82f6" />
        <StatCard label="High Risk" value={stats?.high_risk_count ?? 0} icon={AlertTriangle} color="#ef4444" />
        <StatCard label="Moderate Risk" value={stats?.moderate_risk_count ?? 0} icon={Info} color="#f59e0b" />
        <StatCard label="Low Risk" value={stats?.low_risk_count ?? 0} icon={CheckCircle2} color="#10b981" />
        <StatCard label="Active Alerts" value={stats?.active_alerts ?? 0} icon={AlertTriangle} color="#f59e0b" />
        <StatCard label="Model Confidence" value={Math.round((stats?.model_confidence ?? 0)*100) + '%'} icon={Activity} color="#3b82f6" />
      </div>
    </section>
  );
}
