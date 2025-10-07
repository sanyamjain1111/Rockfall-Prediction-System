from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"

def test_predict_and_map():
    body = {"slope_angle": 40, "rainfall_24h": 10, "rainfall_72h": 25, "days_since_blast": 5, "temperature": 22}
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    r = client.get("/risk-map")
    assert r.status_code == 200
    data = r.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1200
