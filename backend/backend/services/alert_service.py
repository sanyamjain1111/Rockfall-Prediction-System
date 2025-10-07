from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List
import uuid
from backend.config import settings
from backend.services.gemini_service import GeminiService
from backend.utils.logger import setup_logger

logger = setup_logger("AlertService")

@dataclass
class Alert:
    alert_id: str
    level: int
    level_name: str
    zone_id: str
    message: str
    timestamp: datetime
    acknowledged: bool = False

class AlertService:
    def __init__(self):
        self.gemini = GeminiService()
        self.history: List[Alert] = []
        self.thresholds = {
            3: settings.ALERT_THRESHOLD_WARNING,
            2: settings.ALERT_THRESHOLD_ADVISORY,
            1: settings.ALERT_THRESHOLD_WATCH,
        }

    def check_thresholds(self, risk_map: dict) -> List[Alert]:
        new_alerts = []
        for f in risk_map.get("features", []):
            p = f["properties"]["risk_probability"]
            zone_id = f["properties"]["zone_id"]
            level = 0
            if p >= self.thresholds[3]: level = 3
            elif p >= self.thresholds[2]: level = 2
            elif p >= self.thresholds[1]: level = 1
            if level:
                alert = self._create_alert(level, zone_id, p, f["properties"].get("contributing_factors", []))
                self.history.append(alert)
                new_alerts.append(alert)
        return new_alerts

    def _create_alert(self, level: int, zone_id: str, probability: float, factors: List[str]) -> Alert:
        ctx = {
            "risk_level": level,
            "zone_id": zone_id,
            "probability": probability,
            "factors": factors,
        }
        message = self.gemini.generate_alert_message(ctx)
        level_name = {1:"WATCH", 2:"ADVISORY", 3:"WARNING"}[level]
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            level=level, level_name=level_name,
            zone_id=zone_id, message=message,
            timestamp=datetime.utcnow()
        )
        logger.info(f"Alert created: {asdict(alert)}")
        return alert

    def get_alert_history(self, limit: int = 50) -> List[Alert]:
        return sorted(self.history, key=lambda a: a.timestamp, reverse=True)[:limit]

    def acknowledge_alert(self, alert_id: str):
        for a in self.history:
            if a.alert_id == alert_id:
                a.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                break
