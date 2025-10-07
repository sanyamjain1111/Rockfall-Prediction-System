# Rockfall Prediction System — Backend (Part 1)

## Quick start
```bash
pip install -r requirements.txt
python scripts/train_models.py   # optional: trains and saves RF
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs
```
Endpoints: /health, /predict, /risk-map, /stats, /update-conditions, /alert/test, /scenarios, /scenarios/{id}
