import React from 'react';

export default function LoadingSpinner({ size = 'medium', overlay = false }) {
  const spinner = <div className={`spinner ${size}`} aria-label="loading" />;
  if (!overlay) return spinner;
  return (
    <div className="overlay">
      {spinner}
      <div className="loading-text">Loading…</div>
    </div>
  );
}
