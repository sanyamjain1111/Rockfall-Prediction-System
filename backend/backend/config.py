from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    BACKEND_HOST: str = Field(default="0.0.0.0")
    BACKEND_PORT: int = Field(default=8000)
    GEMINI_API_KEY: str | None = None

    ALERT_THRESHOLD_WATCH: float = 0.3
    ALERT_THRESHOLD_ADVISORY: float = 0.5
    ALERT_THRESHOLD_WARNING: float = 0.7

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

    class Config:
        env_file = ".env"

settings = Settings()
