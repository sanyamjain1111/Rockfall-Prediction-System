from __future__ import annotations
from typing import List, Dict
from datetime import datetime
import uuid
from backend.utils.logger import setup_logger
from backend.config import settings
from backend.services.gemini_service import GeminiService

logger = setup_logger("alert")

class AlertService:
    def __init__(self, gemini: GeminiService):
        self.gemini = gemini
        self.history: List[Dict] = []

    def _level_from_prob(self, p: float) -> int:
        if p >= settings.ALERT_THRESHOLD_WARNING:
            return 3
        if p >= settings.ALERT_THRESHOLD_ADVISORY:
            return 2
        if p >= settings.ALERT_THRESHOLD_WATCH:
            return 1
        return 0

    def check_thresholds(self, risk_map: Dict[str,float]) -> List[Dict]:
        alerts = []
        for zid, p in risk_map.items():
            level = self._level_from_prob(p)
            if level >= 1:
                alerts.append({"zone_id": zid, "probability": p, "level": level})
        return alerts

    async def send_alert(self, alert_level: int, zone: str, probability: float, factors: List[str]):
        level_name = {1:"WATCH", 2:"ADVISORY", 3:"WARNING"}.get(alert_level, "INFO")
        ctx = {
            "risk_level": level_name, "zone_id": zone, "probability": probability,
            "weather_conditions": "n/a", "factors": factors
        }
        message = self.gemini.generate_alert_message(ctx)
        item = {
            "alert_id": str(uuid.uuid4()),
            "level": alert_level,
            "level_name": level_name,
            "zone_id": zone,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "acknowledged": False
        }
        self.history.append(item)
        logger.warning(f"ALERT [{level_name}] {zone} p={probability:.2f}: {message}")
        return item

    def get_alert_history(self, limit: int = 50):
        return sorted(self.history, key=lambda x: x["timestamp"], reverse=True)[:limit]

    def acknowledge_alert(self, alert_id: str):
        for itm in self.history:
            if itm["alert_id"] == alert_id:
                itm["acknowledged"] = True
                return itm
        return None
