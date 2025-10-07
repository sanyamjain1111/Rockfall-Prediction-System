from __future__ import annotations
from pydantic import BaseSettings, Field, ValidationError
from typing import Optional

class Settings(BaseSettings):
    BACKEND_HOST: str = Field(default="0.0.0.0")
    BACKEND_PORT: int = Field(default=8000)
    GEMINI_API_KEY: Optional[str] = None
    ALERT_THRESHOLD_WATCH: float = Field(default=0.3, ge=0, le=1)
    ALERT_THRESHOLD_ADVISORY: float = Field(default=0.5, ge=0, le=1)
    ALERT_THRESHOLD_WARNING: float = Field(default=0.7, ge=0, le=1)

    MODEL_PATH: str = "models/saved_models"
    RF_MODEL_FILE: str = "random_forest.pkl"
    XGB_MODEL_FILE: str = "xgboost.pkl"
    CNN_MODEL_FILE: str = "cnn_model.h5"
    LSTM_MODEL_FILE: str = "lstm_model.h5"
    ENSEMBLE_MODEL_FILE: str = "ensemble_stacker.pkl"
    SCALER_FILE: str = "feature_scaler.pkl"

    MINE_CENTER_LAT: float = 23.5
    MINE_CENTER_LON: float = 85.3
    GRID_SIZE_METERS: int = 50
    GRID_ROWS: int = 30
    GRID_COLS: int = 40
    TOTAL_ZONES: int = 1200

    LOG_LEVEL: str = "INFO"
    FAST_MODE: int = 0  # if 1, use lighter ops for tests

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
