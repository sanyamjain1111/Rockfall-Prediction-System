import React, { useState } from 'react';

export default function SliderControl({ label, value, min, max, step, unit, onChange }) {
  const [local, setLocal] = useState(value);
  const handle = (e) => {
    const v = Number(e.target.value);
    setLocal(v);
    onChange(v);
  };
  return (
    <div className="slider-control">
      <div className="slider-label">
        <span>{label}</span>
        <span className="slider-value">{local}{unit ? ` ${unit}` : ''}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={local} onChange={handle} aria-label={label} />
      <div className="slider-minmax">
        <span>{min}{unit ? ` ${unit}` : ''}</span>
        <span>{max}{unit ? ` ${unit}` : ''}</span>
      </div>
    </div>
  );
}
