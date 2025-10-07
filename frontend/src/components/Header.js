import React, { useEffect, useState } from 'react';
import { Activity, Satellite } from 'lucide-react';

export default function Header({ backendUp }) {
  const [time, setTime] = useState(new Date());
  useEffect(() => { const id = setInterval(()=>setTime(new Date()), 1000); return ()=>clearInterval(id); }, []);
  return (
    <header className="header">
      <div className="logo">
        <Satellite size={24} />
        <div>
          <div className="title">Rockfall Prediction System</div>
          <div className="subtitle">AI-Powered Mining Safety</div>
        </div>
      </div>
      <div className="status">
        <div className={`status-dot ${backendUp ? 'ok' : 'down'}`} />
        <span>{backendUp ? 'Connected' : 'Disconnected'}</span>
      </div>
      <div className="clock">
        <Activity size={18} />
        <span>{time.toLocaleString()}</span>
      </div>
    </header>
  );
}
