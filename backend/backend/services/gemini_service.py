from __future__ import annotations
from backend.config import settings
from backend.utils.logger import setup_logger
import time

logger = setup_logger("GeminiService")

class GeminiService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.enabled = bool(self.api_key)
        if self.enabled:
            try:
                import google.generativeai as genai  # noqa
                # Initialize if installed – otherwise we'll fallback gracefully.
                logger.info("Gemini API key detected. Generative alerts enabled.")
            except Exception as e:
                logger.warning(f"Gemini SDK not available; using fallback messaging. {e}")
                self.enabled = False
        else:
            logger.info("No GEMINI_API_KEY provided; using fallback messaging.")

    def _fallback(self, text: str) -> str:
        return text

    def generate_alert_message(self, context: dict) -> str:
        base = (f"Level {context.get('risk_level')} alert for {context.get('zone_id')} — "
                f"Probability {context.get('probability'):.0%}. "
                f"Factors: {', '.join(context.get('factors', [])) or 'N/A'}. ")
        if self.enabled:
            try:
                # Placeholder for real generation; returning base message for deterministic output
                return base + "Action: Restrict access, increase monitoring, and inform operations."
            except Exception as e:
                logger.error(f"Gemini error: {e}")
                return self._fallback(base + "Immediate action recommended.")
        return self._fallback(base + "Immediate action recommended.")

    def analyze_risk_factors(self, features: dict) -> str:
        keys = ', '.join(f"{k}={v}" for k, v in list(features.items())[:6])
        text = f"Risk drivers identified from features: {keys}. Increased shear vs. reduced FOS likely."
        return self._fallback(text)

    def generate_recommendations(self, zone_id: str, risk_level: str) -> str:
        if risk_level == "HIGH":
            return "Evacuate non-essential staff; deploy spotters; inspect benches; suspend blasting."
        if risk_level == "MODERATE":
            return "Increase monitoring; restrict access to benches; schedule inspection within 24h."
        return "Continue routine monitoring; maintain drainage; review after next rainfall."
