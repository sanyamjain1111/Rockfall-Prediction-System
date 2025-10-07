import React, { useMemo, useState } from 'react';
import { MapContainer, TileLayer, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import RiskHeatmap from './RiskHeatmap';
import MapLegend from './MapLegend';

export default function MapView({ riskData, center, zoom }) {
  const [selected, setSelected] = useState(null);
  const onFeatureClick = (feature, latlng) => setSelected({ feature, latlng });
  const position = useMemo(() => center, [center]);
  return (
    <div className="map-container">
      <MapContainer center={position} zoom={zoom} scrollWheelZoom className="leaflet-container">
        <TileLayer
          attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {riskData && <RiskHeatmap geoJsonData={riskData} onFeatureClick={onFeatureClick} />}
        {selected && (
          <Popup position={selected.latlng} onClose={() => setSelected(null)}>
            <div className="custom-popup">
              <div className="popup-header">
                <strong>{selected.feature.properties.zone_id}</strong> — {selected.feature.properties.risk_level}
              </div>
              <div className="popup-body">
                <div>Probability: {(selected.feature.properties.risk_probability * 100).toFixed(1)}%</div>
                <div>Factors:</div>
                <ul>
                  {(selected.feature.properties.contributing_factors || []).map((f,i) => <li key={i}>{f}</li>)}
                </ul>
              </div>
            </div>
          </Popup>
        )}
        <MapLegend position="bottomleft" />
      </MapContainer>
    </div>
  );
}
