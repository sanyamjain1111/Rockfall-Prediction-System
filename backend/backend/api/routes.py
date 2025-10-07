from fastapi import APIRouter, HTTPException
from datetime import datetime
from backend.api.schemas import PredictionRequest, PredictionResponse, SystemStats, AlertResponse
from backend.services.prediction_service import PredictionService
from backend.services.alert_service import AlertService
from backend.config import settings

router = APIRouter()

# Singletons (module-level for simplicity)
prediction_service = PredictionService()
alert_service = AlertService()

@router.get("/health")
def health():
    from datetime import datetime
    loaded = getattr(prediction_service, "selector", "heuristic")
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "loaded_models": [loaded]
    }

@router.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        pred = prediction_service.predict_risk(req.model_dump())
        # Trigger alerts
        risk_map = prediction_service.last_risk_map
        alerts = alert_service.check_thresholds(risk_map)
        return pred
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/risk-map")
def risk_map():
    if prediction_service.last_risk_map is None:
        # Generate default using baseline conditions
        prediction_service.predict_risk({
            "slope_angle": 40, "rainfall_24h": 5, "rainfall_72h": 15, "days_since_blast": 10, "temperature": 22
        })
    return prediction_service.last_risk_map

@router.get("/stats", response_model=SystemStats)
def stats():
    s = prediction_service.get_system_stats()
    s["active_alerts"] = len(alert_service.get_alert_history())
    return s

@router.post("/update-conditions", response_model=SystemStats)
def update_conditions(req: PredictionRequest):
    prediction_service.predict_risk(req.model_dump())
    s = prediction_service.get_system_stats()
    s["active_alerts"] = len(alert_service.get_alert_history())
    return s

@router.post("/alert/test", response_model=AlertResponse)
def alert_test():
    # Simulate level 3 WARNING for demo
    alert = alert_service._create_alert(3, "Z00-00", 0.92, ["Heavy rainfall", "Steep slope", "Low cohesion"])
    alert_service.history.append(alert)
    return alert.__dict__

@router.get("/scenarios")
def list_scenarios():
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
        data = json.load(f)
    match = next((s for s in data if s["id"] == scenario_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Scenario not found")
    req = PredictionRequest(
        slope_angle=40,
        rainfall_24h=match["rainfall_24h"],
        rainfall_72h=match["rainfall_72h"],
        days_since_blast=match["days_since_blast"],
        temperature=match["temperature"]
    )
    return predict(req)
