import { useCallback, useEffect, useState } from 'react';
import api from '../services/api';

export default function useStats() {
  const [stats, setStats] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchStats = useCallback( async () => {
    setIsLoading(true); setError(null);
    try {
      const data = await api.getStats();
      setStats(data);
    } catch (e) { setError(e.message); }
    finally { setIsLoading(false); }
  }, []);

  useEffect(() => {
    fetchStats();
    const id = setInterval(fetchStats, 10000);
    return () => clearInterval(id);
  }, [fetchStats]);

  return { stats: stats || {}, isLoading, error, fetchStats };
}
