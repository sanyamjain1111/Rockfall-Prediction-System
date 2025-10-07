from __future__ import annotations
import os, json, time
from typing import Dict, List, Tuple
import numpy as np, pandas as pd, joblib
from datetime import datetime
from backend.utils.logger import setup_logger
from backend.config import settings
from preprocessing.feature_engineering import FeatureEngineering, NUM_FEATURES, CAT_FEATURES, LITH_CATS
from preprocessing.geospatial_processor import create_grid, zone_id
from models.ensemble_stacker import EnsembleStacker

logger = setup_logger("predict", settings.LOG_LEVEL)

class PredictionService:
    def __init__(self):
        self.grid = create_grid(settings.MINE_CENTER_LAT, settings.MINE_CENTER_LON,
                                settings.GRID_ROWS, settings.GRID_COLS, settings.GRID_SIZE_METERS)
        self.zone_ids = [zone_id(r, c) for r in range(settings.GRID_ROWS) for c in range(settings.GRID_COLS)]
        self.fe = FeatureEngineering(scaler_path=os.path.join(settings.MODEL_PATH, settings.SCALER_FILE))
        self.ensemble = EnsembleStacker()
        self.last_prediction: Dict[str, float] = {}
        self.last_levels: Dict[str, str] = {}
        self.last_update = datetime.utcnow()
        self.start_time = time.time()
        # Load scaler if exists
        try:
            self.fe.scaler = joblib.load(os.path.join(settings.MODEL_PATH, settings.SCALER_FILE))
            self.fe.fitted = True
        except Exception:
            logger.warning("Scaler not found; will fit on-the-fly if needed")
        # Load ensemble if exists
        ens_path = os.path.join(settings.MODEL_PATH, settings.ENSEMBLE_MODEL_FILE)
        if os.path.exists(ens_path):
            try:
                self.ensemble.load_ensemble(ens_path)
                logger.info("Ensemble loaded")
            except Exception as e:
                logger.warning(f"Failed to load ensemble: {e}")
        else:
            logger.info("No ensemble model found; safe-mode heuristic will be used")

        # Minimal per-zone static features (randomized for demo)
        rng = np.random.default_rng(123)
        self.zone_static = pd.DataFrame({
            "slope_angle": rng.uniform(15, 65, len(self.zone_ids)),
            "aspect": rng.uniform(0, 360, len(self.zone_ids)),
            "elevation": rng.uniform(150, 300, len(self.zone_ids)),
            "plan_curvature": rng.normal(0, 0.15, len(self.zone_ids)),
            "profile_curvature": rng.normal(0, 0.15, len(self.zone_ids)),
            "TWI": rng.uniform(5, 20, len(self.zone_ids)),
            "SPI": rng.uniform(0, 100, len(self.zone_ids)),
            "lithology": rng.choice(LITH_CATS, len(self.zone_ids), p=[0.3,0.25,0.25,0.2]),
            "distance_to_fault": rng.uniform(0, 1000, len(self.zone_ids)),
            "fracture_density": rng.uniform(0, 10, len(self.zone_ids)),
            "cohesion": rng.uniform(5, 50, len(self.zone_ids)),
            "friction_angle": rng.uniform(25, 45, len(self.zone_ids)),
            "unit_weight": rng.uniform(20, 28, len(self.zone_ids)),
            "pore_pressure_ratio": rng.uniform(0, 0.8, len(self.zone_ids)),
            "temperature_range": rng.uniform(-10, 40, len(self.zone_ids)),
            "insar_velocity": rng.uniform(0, 50, len(self.zone_ids)),
            "days_since_blast": 10.0 # default; will be overwritten
        }, index=self.zone_ids)

    def _classify(self, p: float) -> Tuple[str,str]:
        if p >= settings.ALERT_THRESHOLD_WARNING: return "HIGH", "#ef4444"
        if p >= settings.ALERT_THRESHOLD_ADVISORY: return "MODERATE", "#f59e0b"
        return "LOW", "#10b981"

    def _heuristic_predict(self, X_scaled: np.ndarray) -> np.ndarray:
        # Safe mode: blend slope, rain, twi and insar-like columns if present
        # Use column names from preprocessed DataFrame later; assume averages here.
        base = X_scaled.mean(axis=1)
        return np.clip(base, 0, 1)

    def calculate_contributing_factors(self, zone_id_str: str) -> List[str]:
        row = self.zone_static.loc[zone_id_str].to_dict()
        # Simple top factors heuristic
        factors = []
        if row["slope_angle"] > 50: factors.append("High slope angle")
        if row["TWI"] > 15: factors.append("High wetness index (TWI)")
        if row["insar_velocity"] > 20: factors.append("Elevated deformation (InSAR)")
        if row["days_since_blast"] < 3: factors.append("Recent blast activity")
        if not factors:
            factors.append("No dominant factor; cumulative moderate contributors")
        return factors[:5]

    def generate_recommendation(self, risk_level: str, zone_id: str) -> str:
        if risk_level == "HIGH":
            return "Evacuate zone, install barricades, geotech inspection within 2 hours."
        if risk_level == "MODERATE":
            return "Increase monitoring to hourly, restrict heavy vehicles, check drainage."
        return "Continue routine monitoring; inspect benches during shift change."

    def predict_risk(self, conditions: Dict) -> Dict:
        # Update dynamic conditions
        dyn = {
            "cumulative_rainfall_24h": float(conditions["rainfall_24h"]),
            "cumulative_rainfall_72h": float(conditions["rainfall_72h"]),
            "days_since_blast": float(conditions["days_since_blast"]),
            "temperature_range": float(conditions["temperature"])
        }
        df = self.zone_static.copy()
        for k,v in dyn.items():
            df[k] = v

        # Full transform
        df_tr = self.fe.full_transform(df, fit=not self.fe.fitted)
        X = df_tr.drop(columns=["lithology"], errors="ignore").values

        # Inference
        try:
            proba = self.ensemble.predict(X)
        except Exception:
            proba = self._heuristic_predict(X)

        results = {}
        levels = {}
        for zid, p in zip(self.zone_ids, proba):
            lvl, color = self._classify(float(p))
            results[zid] = float(p)
            levels[zid] = lvl

        self.last_prediction = results
        self.last_levels = levels
        self.last_update = datetime.utcnow()

        high = [z for z,l in levels.items() if l=="HIGH"]
        mod = [z for z,l in levels.items() if l=="MODERATE"]
        low = [z for z,l in levels.items() if l=="LOW"]
        # crude confidence = 1 - variance across probs
        conf = float(1.0 - np.var(list(results.values())))

        return {
            "predictions": results,
            "levels": levels,
            "high": high,
            "moderate": mod,
            "low": low,
            "confidence": max(0.0, min(1.0, conf))
        }

    def generate_risk_heatmap(self) -> Dict:
        # GeoJSON FeatureCollection
        feats = []
        for (bbox), zid in zip(self.grid, self.zone_ids):
            (min_lon, min_lat), (max_lon, max_lat) = bbox
            p = float(self.last_prediction.get(zid, 0.1))
            lvl, color = self._classify(p)
            feats.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [min_lon, min_lat],[max_lon, min_lat],
                        [max_lon, max_lat],[min_lon, max_lat],[min_lon, min_lat]
                    ]]
                },
                "properties": {
                    "zone_id": zid,
                    "risk_probability": p,
                    "risk_level": lvl,
                    "color": color,
                    "contributing_factors": self.calculate_contributing_factors(zid)
                }
            })
        return {"type": "FeatureCollection", "features": feats}

    def get_system_stats(self) -> Dict:
        high = sum(1 for v in self.last_levels.values() if v=="HIGH")
        mod = sum(1 for v in self.last_levels.values() if v=="MODERATE")
        low = sum(1 for v in self.last_levels.values() if v=="LOW")
        conf = float(1.0 - np.var(list(self.last_prediction.values()))) if self.last_prediction else 0.5
        return {
            "total_zones": len(self.zone_ids),
            "high_risk_count": high,
            "moderate_risk_count": mod,
            "low_risk_count": low,
            "active_alerts": 0,
            "model_confidence": max(0.0, min(1.0, conf)),
            "last_update": self.last_update,
            "uptime_seconds": int(time.time() - self.start_time)
        }
