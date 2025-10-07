import React, { useEffect, useMemo, useState } from 'react';
import Header from './components/Header';
import MapView from './components/Map';
import Dashboard from './components/Dashboard';
import Controls from './components/Controls';
import AlertPanel from './components/AlertPanel';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorBoundary from './components/ErrorBoundary';
import api from './services/api';
import useRiskData from './hooks/useRiskData';
import useStats from './hooks/useStats';
import useAlerts from './hooks/useAlerts';
import { formatTimestamp } from './utils/formatters';

const DEFAULT_CONDITIONS = { slope_angle: 40, rainfall_24h: 5, rainfall_72h: 15, days_since_blast: 10, temperature: 22 };

export default function App() {
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenario, setSelectedScenario] = useState(null);
  const [conditions, setConditions] = useState(DEFAULT_CONDITIONS);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendUp, setBackendUp] = useState(false);

  const { riskData, isLoading: riskLoading, fetchRiskData, lastUpdate } = useRiskData();
  const { stats, fetchStats } = useStats();
  const { alerts, addAlert, acknowledgeAlert, clearAll, unacknowledgedCount } = useAlerts();

  const center = useMemo(() => [parseFloat(process.env.REACT_APP_MAP_CENTER_LAT || '23.5'),
                                parseFloat(process.env.REACT_APP_MAP_CENTER_LON || '85.3')], []);
  const zoom = parseInt(process.env.REACT_APP_MAP_ZOOM || '15', 10);

  useEffect(() => {
    let mounted = true;
    api.healthCheck().then(() => mounted && setBackendUp(true)).catch(() => setBackendUp(false));
    api.getScenarios().then((res) => {
      if (!mounted) return;
      setScenarios(res || []);
      if (res && res.length > 0) setSelectedScenario(res[0].id);
    }).catch(() => {});
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    fetchRiskData();
    fetchStats();
  }, [fetchRiskData, fetchStats]);

  useEffect(() => {
    const intervalMs = parseInt(process.env.REACT_APP_AUTO_REFRESH_INTERVAL || '30000', 10);
    const id = setInterval(() => {
      fetchRiskData();
      fetchStats();
    }, intervalMs);
    return () => clearInterval(id);
  }, [fetchRiskData, fetchStats]);

  const handlePrediction = async (nextConditions) => {
    setIsLoading(true);
    setError(null);
    try {
      await api.runPrediction(nextConditions);
      await Promise.all([fetchRiskData(), fetchStats()]);
    } catch (e) {
      setError(e.message || 'Prediction failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleScenarioSelect = async (scenarioId) => {
    setSelectedScenario(scenarioId);
    setIsLoading(true);
    setError(null);
    try {
      await api.loadScenario(scenarioId);
      await Promise.all([fetchRiskData(), fetchStats()]);
    } catch (e) {
      setError(e.message || 'Scenario failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTestAlert = async () => {
    try {
      const alert = await api.testAlert();
      addAlert(alert);
    } catch (e) {
      setError(e.message || 'Alert failed');
    }
  };

  const onConditionChange = (field, value) => {
    setConditions(prev => ({ ...prev, [field]: value }));
  };

  const clearError = () => setError(null);

  return (
    <ErrorBoundary>
      <div className="app-container">
        <Header backendUp={backendUp} lastUpdate={lastUpdate} />
        <div className="sidebar">
          <Controls
            conditions={conditions}
            scenarios={scenarios}
            selectedScenario={selectedScenario}
            onScenarioSelect={handleScenarioSelect}
            onConditionChange={onConditionChange}
            onRunPrediction={() => handlePrediction(conditions)}
            onTestAlert={handleTestAlert}
            isLoading={isLoading}
          />
          <Dashboard stats={stats} />
        </div>
        <div className="map-area">
          <MapView
            riskData={riskData}
            center={center}
            zoom={zoom}
          />
        </div>
        <div className="alerts-area">
          <AlertPanel
            alerts={alerts}
            onAcknowledge={acknowledgeAlert}
            onClearAll={clearAll}
            unacknowledgedCount={unacknowledgedCount}
          />
        </div>
        <footer className="footer">
          <div>Rockfall Prediction System • Last updated {lastUpdate ? formatTimestamp(lastUpdate) : '—'}</div>
        </footer>
        {(isLoading || riskLoading) && <LoadingSpinner overlay size="large" />}
        {error && (
          <div className="toast error" role="alert" onClick={clearError}>
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
}
