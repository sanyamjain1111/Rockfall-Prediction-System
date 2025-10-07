import React from 'react';
import { GeoJSON } from 'react-leaflet';
import { getRiskColor } from '../utils/colors';

export default function RiskHeatmap({ geoJsonData, onFeatureClick }) {
  const style = (feature) => {
    const p = feature.properties.risk_probability || 0;
    return { color: '#ffffff', weight: 1, fillColor: getRiskColor(p), fillOpacity: 0.6 };
  };
  const onEachFeature = (feature, layer) => {
    const props = feature.properties || {};
    layer.bindTooltip(`${props.zone_id} — ${props.risk_level}`, { sticky: true });
    layer.on('click', (e) => onFeatureClick(feature, e.latlng));
  };
  return <GeoJSON data={geoJsonData} style={style} onEachFeature={onEachFeature} />;
}
