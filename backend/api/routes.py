from __future__ import annotations
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime
from backend.api.schemas import PredictionRequest, PredictionResponse, SystemStats, AlertResponse
from backend.config import settings
from backend.services.prediction_service import PredictionService
from backend.services.alert_service import AlertService
from backend.services.gemini_service import GeminiService

router = APIRouter()
_pred_service: PredictionService = None
_alert_service: AlertService = None
_gemini: GeminiService = None
_last_conditions: Dict[str, Any] = dict(slope_angle=40, rainfall_24h=10, rainfall_72h=30, days_since_blast=10, temperature=22)

def init_services(ps: PredictionService, asvc: AlertService, gsvc: GeminiService):
    global _pred_service, _alert_service, _gemini
    _pred_service, _alert_service, _gemini = ps, asvc, gsvc

@router.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "loaded_models": True  # ensemble loads in service (safe-mode if unavailable)
    }

@router.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if req.rainfall_72h < req.rainfall_24h:
        raise HTTPException(status_code=400, detail="rainfall_72h must be >= rainfall_24h")
    res = _pred_service.predict_risk({
        "rainfall_24h": req.rainfall_24h,
        "rainfall_72h": req.rainfall_72h,
        "days_since_blast": req.days_since_blast,
        "temperature": req.temperature
    })
    return {
        "status": "success",
        "high_risk_zones": res["high"],
        "moderate_risk_zones": res["moderate"],
        "low_risk_zones": res["low"],
        "total_zones": settings.TOTAL_ZONES,
        "timestamp": datetime.utcnow(),
        "model_confidence": res["confidence"]
    }

@router.get("/risk-map")
def risk_map():
    return _pred_service.generate_risk_heatmap()

@router.get("/stats", response_model=SystemStats)
def stats():
    s = _pred_service.get_system_stats()
    return s

@router.post("/update-conditions", response_model=SystemStats)
def update_conditions(req: PredictionRequest):
    global _last_conditions
    _last_conditions = req.model_dump()
    _pred_service.predict_risk({
        "rainfall_24h": req.rainfall_24h,
        "rainfall_72h": req.rainfall_72h,
        "days_since_blast": req.days_since_blast,
        "temperature": req.temperature
    })
    return _pred_service.get_system_stats()

@router.get("/scenarios")
def scenarios():
    import json, os
    p = os.path.join("backend", "demo_scenarios.json")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@router.post("/scenarios/{scenario_id}", response_model=PredictionResponse)
def run_scenario(scenario_id: int):
    import json, os
    p = os.path.join("backend", "demo_scenarios.json")
    with open(p, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    if not (1 <= scenario_id <= len(scenarios)):
        raise HTTPException(status_code=404, detail="Scenario not found")
    sc = scenarios[scenario_id-1]["conditions"]
    res = _pred_service.predict_risk({
        "rainfall_24h": sc["rainfall_24h"],
        "rainfall_72h": sc["rainfall_72h"],
        "days_since_blast": sc["days_since_blast"],
        "temperature": sc["temperature"]
    })
    return {
        "status": "success",
        "high_risk_zones": res["high"],
        "moderate_risk_zones": res["moderate"],
        "low_risk_zones": res["low"],
        "total_zones": settings.TOTAL_ZONES,
        "timestamp": datetime.utcnow(),
        "model_confidence": res["confidence"]
    }

@router.post("/alert/test", response_model=AlertResponse)
async def alert_test():
    # Level 3 demo
    msg = await _alert_service.send_alert(3, "Z010-010", 0.88, ["Steep slope", "High TWI"])
    return msg
