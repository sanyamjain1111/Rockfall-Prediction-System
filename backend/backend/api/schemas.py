from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime
import uuid

class PredictionRequest(BaseModel):
    slope_angle: float = Field(ge=0, le=90)
    rainfall_24h: float = Field(ge=0, le=500)
    rainfall_72h: float = Field(ge=0, le=1000)
    days_since_blast: int = Field(ge=0, le=60)
    temperature: float = Field(ge=-20, le=50)

    @field_validator("rainfall_72h")
    @classmethod
    def rainfall_order(cls, v, info):
        rainfall_24h = info.data.get("rainfall_24h", 0)
        if v < rainfall_24h:
            raise ValueError("rainfall_72h must be >= rainfall_24h")
        return v

class ZoneRisk(BaseModel):
    zone_id: str
    risk_probability: float = Field(ge=0, le=1)
    risk_level: str
    color: str
    contributing_factors: List[str] = []
    recommendation: Optional[str] = None

class PredictionResponse(BaseModel):
    status: str
    high_risk_zones: List[str]
    moderate_risk_zones: List[str]
    low_risk_zones: List[str]
    total_zones: int
    timestamp: datetime
    model_confidence: float

class AlertResponse(BaseModel):
    alert_id: str
    level: int
    level_name: str
    zone_id: str
    message: str
    timestamp: datetime
    acknowledged: bool = False

class SystemStats(BaseModel):
    total_zones: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int
    active_alerts: int
    model_confidence: float
    last_update: datetime
    uptime_seconds: int
