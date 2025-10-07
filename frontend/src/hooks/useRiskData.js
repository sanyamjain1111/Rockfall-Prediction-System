import { useCallback, useEffect, useState } from 'react';
import api from '../services/api';

export default function useRiskData() {
  const [riskData, setRiskData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  const fetchRiskData = useCallback(async () => {
    setIsLoading(true); setError(null);
    try {
      const data = await api.getRiskMap();
      setRiskData(data);
      setLastUpdate(new Date());
    } catch (e) { setError(e.message); }
    finally { setIsLoading(false); }
  }, []);

  useEffect(() => { fetchRiskData(); }, [fetchRiskData]);

  return { riskData, isLoading, error, fetchRiskData, lastUpdate };
}
