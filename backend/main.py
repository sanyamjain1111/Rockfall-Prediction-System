from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.utils.logger import setup_logger
from backend.config import settings
from backend.api.routes import router, init_services
from backend.services.prediction_service import PredictionService
from backend.services.alert_service import AlertService
from backend.services.gemini_service import GeminiService

logger = setup_logger("app", settings.LOG_LEVEL)

app = FastAPI(title="Rockfall Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    logger.info("Starting up...")
    ps = PredictionService()
    gs = GeminiService(settings.GEMINI_API_KEY)
    al = AlertService(gs)
    init_services(ps, al, gs)
    logger.info("Startup complete")

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down...")

app.include_router(router)
