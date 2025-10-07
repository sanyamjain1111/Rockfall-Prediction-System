from __future__ import annotations
import json, math, os, random
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

from backend.config import settings
from backend.utils.logger import setup_logger

logger = setup_logger("PredictionService")

try:
    import joblib
except Exception:
    joblib = None

class PredictionService:
    def __init__(self):
        self.logger = logger  # ✅ Add this line
        self.rows = settings.GRID_ROWS
        self.cols = settings.GRID_COLS
        self.total = settings.TOTAL_ZONES
        self.grid = self._build_grid()
        self.model = self._load_models()
        self.last_prediction = None
        self.last_risk_map = None
        self.last_update = None


    # -------------------- Grid --------------------
    def _build_grid(self) -> List[Tuple[int,int,str]]:
        grid = []
        for r in range(self.rows):
            for c in range(self.cols):
                grid.append((r, c, f"Z{r:02d}-{c:02d}"))
        return grid

    # -------------------- Model -------------------
    def _load_models(self):
        """Load ensemble and RF; choose best selector using TRAINING_REPORT.json."""
        import json, os
        self.ensemble = None
        self.rf = None
        self.selector = "heuristic"

        model_dir = settings.MODEL_PATH
        ens_path = os.path.join(model_dir, settings.ENSEMBLE_MODEL_FILE)
        rf_path  = os.path.join(model_dir, settings.RF_MODEL_FILE)
        report_path = os.path.join(model_dir, "TRAINING_REPORT.json")

        report = {}
        if os.path.exists(report_path):
            try:
                with open(report_path, "r") as f:
                    report = json.load(f)
            except Exception:
                report = {}

        ens_auc = report.get("ensemble_auc")
        rf_auc  = report.get("random_forest_auc")

        # Try to load ensemble
        if joblib and os.path.exists(ens_path):
            try:
                self.ensemble = joblib.load(ens_path)
                self.logger.info(f"Loaded ensemble from {ens_path}")
            except Exception as e:
                self.logger.error(f"Failed to load ensemble: {e}")
                self.ensemble = None

        # Try to load RF
        if joblib and os.path.exists(rf_path):
            try:
                self.rf = joblib.load(rf_path)
                self.logger.info(f"Loaded RF from {rf_path}")
            except Exception as e:
                self.logger.error(f"Failed to load RF: {e}")
                self.rf = None

        # Selection policy:
        # - Prefer ensemble if AUC>=0.55
        # - Else RF if AUC>=0.55
        # - Else fallback to heuristic
        if self.ensemble and (ens_auc is not None) and (ens_auc >= 0.55):
            self.selector = "ensemble"
        elif self.rf and (rf_auc is not None) and (rf_auc >= 0.55):
            self.selector = "rf"
        elif self.ensemble:
            # If no AUC in report, but ensemble exists, still prefer ensemble
            self.selector = "ensemble"
        elif self.rf:
            self.selector = "rf"
        else:
            self.selector = "heuristic"

        self.logger.info(f"Model selector: {self.selector} (ens_auc={ens_auc}, rf_auc={rf_auc})")

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability of class=1 (unstable) for each row in X.
        Uses selector order: ensemble -> rf -> heuristic.
        """
        try:
            if self.selector == "ensemble" and getattr(self, "ensemble", None):
                obj = self.ensemble
                etype = obj.get("type")
                if etype == "calibrated":
                    return obj["model"].predict_proba(X)[:, 1]
                if etype == "stacked":
                    comps = obj.get("components", [])
                    models = obj.get("models", {})
                    zs = []
                    for name in comps:
                        m = models.get(name)
                        zs.append(m.predict_proba(X)[:, 1][:, None])
                    Z = np.hstack(zs)
                    return obj["meta"].predict_proba(Z)[:, 1]
            if self.selector == "rf" and getattr(self, "rf", None):
                return self.rf.predict_proba(X)[:, 1]
        except Exception as e:
            self.logger.error(f"Predict failed with {self.selector}: {e}")
        # last resort
        return self._heuristic(X)

    # -------------------- Feature synthesis --------
    def _zone_base_features(self, r: int, c: int) -> Dict[str, float]:
        random.seed(r * 1000 + c)
        slope_angle = 20 + 45 * random.random()  # 20-65
        twi = 5 + 15 * random.random()
        cohesion = 10 + 40 * random.random()
        friction_angle = 25 + 20 * random.random()
        days_since_blast = random.randint(0, 30)
        return {
            "slope_angle": slope_angle,
            "twi": twi,
            "cohesion": cohesion,
            "friction_angle": friction_angle,
            "days_since_blast": days_since_blast
        }

    def _compose_features(self, base: Dict[str, float], conditions: Dict[str, float]) -> List[float]:
        # Merge static base with dynamic conditions
        rainfall_24h = conditions["rainfall_24h"]
        rainfall_72h = conditions["rainfall_72h"]
        days_since_blast_now = conditions["days_since_blast"]
        temp = conditions["temperature"]
        slope = conditions["slope_angle"]  # user-provided to emphasize what-if

        f = [
            slope,
            base["twi"],
            base["cohesion"],
            base["friction_angle"],
            rainfall_24h,
            rainfall_72h,
            days_since_blast_now,
            temp
        ]
        return f

    # -------------------- Prediction core ----------
    def predict_risk(self, conditions: Dict[str, float]) -> Dict:
        features = []
        zone_ids = []
        probs = []

        for (r, c, zid) in self.grid:
            base = self._zone_base_features(r, c)
            x = self._compose_features(base, conditions)
            features.append(x)
            zone_ids.append(zid)

        X = np.array(features, dtype=float)

        if self.model:
            try:
                p = self.model.predict_proba(X)[:, 1]
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                p = self._heuristic(X)
        else:
            p = self._heuristic(X)

        probs = p.tolist()

        # Classify
        high_ids, mod_ids, low_ids = [], [], []
        for zid, pr in zip(zone_ids, probs):
            if pr >= settings.ALERT_THRESHOLD_WARNING:
                high_ids.append(zid)
            elif pr >= settings.ALERT_THRESHOLD_ADVISORY:
                mod_ids.append(zid)
            else:
                low_ids.append(zid)

        # Build geojson
        risk_map = self.generate_risk_heatmap(zone_ids, probs)

        # Save
        self.last_prediction = {
            "status": "ok",
            "high_risk_zones": high_ids,
            "moderate_risk_zones": mod_ids,
            "low_risk_zones": low_ids,
            "total_zones": self.total,
            "timestamp": datetime.utcnow().isoformat(),
            "model_confidence": float(np.mean([abs(pr-0.5)*2 for pr in p]))
        }
        self.last_risk_map = risk_map
        self.last_update = datetime.utcnow()
        return self.last_prediction

    def _heuristic(self, X: np.ndarray) -> np.ndarray:
        # X columns: slope, twi, cohesion, friction, rain24, rain72, days_since_blast, temp
        slope = X[:,0]; twi = X[:,1]; coh=X[:,2]; fric=X[:,3]; r24=X[:,4]; r72=X[:,5]; dsb=X[:,6]; temp=X[:,7]
        # Simple risk score combining destabilizing effects
        score = (0.6*(slope/70.0) + 0.4*(r24/150.0) + 0.3*(r72/300.0) + 0.2*(twi/20.0) + 0.15*((30-dsb)/30.0))                 - (0.25*(coh/50.0) + 0.25*(fric/45.0)) + 0.1*((25-abs(22-temp))/25.0)
        score = np.clip(score, 0, 1)
        return score

    # -------------------- GeoJSON ------------------
    def generate_risk_heatmap(self, zone_ids: List[str], probs: List[float]) -> Dict:
        lat0 = settings.MINE_CENTER_LAT
        lon0 = settings.MINE_CENTER_LON
        size_m = settings.GRID_SIZE_METERS
        dlat = size_m / 111320.0
        dlon = size_m / (111320.0 * math.cos(math.radians(lat0)))

        features = []
        idx = 0
        for r in range(self.rows):
            for c in range(self.cols):
                zid = zone_ids[idx]
                p = probs[idx]
                idx += 1

                lat_min = lat0 + (r - self.rows/2) * dlat
                lon_min = lon0 + (c - self.cols/2) * dlon
                lat_max = lat_min + dlat
                lon_max = lon_min + dlon

                color = "#10b981" if p < 0.3 else ("#f59e0b" if p < 0.7 else "#ef4444")
                feat = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [lon_min, lat_min],
                            [lon_max, lat_min],
                            [lon_max, lat_max],
                            [lon_min, lat_max],
                            [lon_min, lat_min]
                        ]]
                    },
                    "properties": {
                        "zone_id": zid,
                        "risk_probability": float(p),
                        "risk_level": "LOW" if p < 0.3 else ("MODERATE" if p < 0.7 else "HIGH"),
                        "color": color,
                        "contributing_factors": [],
                    }
                }
                features.append(feat)
        return {"type":"FeatureCollection", "features": features}

    # -------------------- Stats --------------------
    def get_system_stats(self) -> Dict:
        if not self.last_prediction:
            return {
                "total_zones": self.total,
                "high_risk_count": 0,
                "moderate_risk_count": 0,
                "low_risk_count": self.total,
                "active_alerts": 0,
                "model_confidence": 0.5,
                "last_update": datetime.utcnow().isoformat(),
                "uptime_seconds": 0
            }
        return {
            "total_zones": self.total,
            "high_risk_count": len(self.last_prediction["high_risk_zones"]),
            "moderate_risk_count": len(self.last_prediction["moderate_risk_zones"]),
            "low_risk_count": len(self.last_prediction["low_risk_zones"]),
            "active_alerts": 0,  # AlertService tracks this
            "model_confidence": self.last_prediction["model_confidence"],
            "last_update": self.last_prediction["timestamp"],
            "uptime_seconds": 0
        }
