import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' }
});

api.interceptors.response.use((r) => r, (error) => {
  const msg = error?.response?.data?.detail || error.message || 'Network error';
  return Promise.reject(new Error(msg));
});

const healthCheck = async () => (await api.get('/health')).data;
const runPrediction = async (conditions) => (await api.post('/predict', conditions)).data;
const getRiskMap = async () => (await api.get('/risk-map')).data;
const getStats = async () => (await api.get('/stats')).data;
const getScenarios = async () => (await api.get('/scenarios')).data;
const loadScenario = async (id) => (await api.post(`/scenarios/${id}`)).data;
const testAlert = async () => (await api.post('/alert/test')).data;

export default { healthCheck, runPrediction, getRiskMap, getStats, getScenarios, loadScenario, testAlert };
export { healthCheck, runPrediction, getRiskMap, getStats, getScenarios, loadScenario, testAlert };
