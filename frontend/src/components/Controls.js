import React from 'react';
import ScenarioSelector from './ScenarioSelector';
import SliderControl from './SliderControl';

export default function Controls({ conditions, scenarios, selectedScenario, onScenarioSelect, onConditionChange, onRunPrediction, onTestAlert, isLoading }) {
  return (
    <div className="controls-panel">
      <div className="control-section">
        <div className="section-title">Demo Scenarios</div>
        <ScenarioSelector
          scenarios={scenarios}
          selectedId={selectedScenario}
          onChange={onScenarioSelect}
        />
      </div>
      <div className="control-section">
        <div className="section-title">Manual Controls</div>
        <SliderControl label="Rainfall 24h" unit="mm"
          min={0} max={150} step={1}
          value={conditions.rainfall_24h}
          onChange={(v) => onConditionChange('rainfall_24h', v)} />
        <SliderControl label="Rainfall 72h" unit="mm"
          min={0} max={300} step={1}
          value={conditions.rainfall_72h}
          onChange={(v) => onConditionChange('rainfall_72h', v)} />
        <SliderControl label="Days Since Blast" unit="days"
          min={0} max={30} step={1}
          value={conditions.days_since_blast}
          onChange={(v) => onConditionChange('days_since_blast', v)} />
        <SliderControl label="Temperature" unit="°C"
          min={-20} max={50} step={1}
          value={conditions.temperature}
          onChange={(v) => onConditionChange('temperature', v)} />
        <button className="btn-primary" disabled={isLoading} onClick={onRunPrediction}>
          {isLoading ? 'Running...' : 'Run Prediction'}
        </button>
        <div style={{height:8}}/>
        <button className="alert-ack-button" onClick={onTestAlert}>Test Alert</button>
      </div>
    </div>
  );
}
