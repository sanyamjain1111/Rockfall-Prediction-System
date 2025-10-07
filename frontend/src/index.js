import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/App.css';
import './styles/Map.css';
import './styles/Dashboard.css';
import './styles/Controls.css';
import './styles/Alerts.css';

const container = document.getElementById('root');
const root = createRoot(container);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
