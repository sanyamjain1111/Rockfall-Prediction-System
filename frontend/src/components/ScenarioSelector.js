import React from 'react';

export default function ScenarioSelector({ scenarios, selectedId, onChange }) {
  return (
    <select className="scenario-select" value={selectedId || ''} onChange={e => onChange(parseInt(e.target.value,10))}>
      {scenarios && scenarios.map(s => (
        <option key={s.id} value={s.id}>{s.name}</option>
      ))}
    </select>
  );
}
