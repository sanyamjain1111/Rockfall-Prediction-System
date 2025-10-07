import React from 'react';
export default function MapLegend() {
  return (
    <div className="map-legend">
      <strong>Risk Level</strong>
      <div className="legend-row"><span className="swatch" style={{background:'#10b981'}}></span> Low (0–30%)</div>
      <div className="legend-row"><span className="swatch" style={{background:'#f59e0b'}}></span> Moderate (30–70%)</div>
      <div className="legend-row"><span className="swatch" style={{background:'#ef4444'}}></span> High (70–100%)</div>
    </div>
  );
}
