from __future__ import annotations
import os, time, random
from typing import Dict, Optional
from backend.utils.logger import setup_logger

logger = setup_logger("gemini")

try:
    import google.generativeai as genai
except Exception:
    genai = None

class GeminiService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.ready = False
        if genai and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-pro")
                self.ready = True
                logger.info("Gemini initialized")
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")
                self.ready = False

    def _fallback(self, text: str) -> str:
        return text

    def generate_alert_message(self, context: Dict) -> str:
        prompt = (
            "You are a mine-safety assistant. Create a concise, actionable alert. "
            f"Context: {context}. Use ≤80 words, include risk level, zone, "
            "probability (as %), and immediate actions.
"
            "If risk is WARNING, include evacuation and cordon advice."
        )
        if self.ready:
            try:
                resp = self.model.generate_content(prompt)
                return resp.text.strip()
            except Exception as e:
                logger.error(f"Gemini error: {e}")
        # fallback
        risk = context.get("risk_level","UNKNOWN")
        zone = context.get("zone_id","Z?")
        prob = context.get("probability",0.0)
        return self._fallback(f"[{risk}] Zone {zone}: Estimated risk {prob*100:.1f}%. "
                              "Increase monitoring, restrict access, and prepare mitigation.")

    def analyze_risk_factors(self, features: Dict) -> str:
        prompt = (
            "Explain briefly why rockfall risk is elevated given these features: "
            f"{features}. Use technical but clear language."
        )
        if self.ready:
            try:
                resp = self.model.generate_content(prompt)
                return resp.text.strip()
            except Exception as e:
                logger.error(f"Gemini error: {e}")
        return self._fallback("Key contributors likely include steep slopes, high wetness, "
                              "recent blasting, and elevated deformation rates.")

    def generate_recommendations(self, zone_id: str, risk_level: str) -> str:
        prompt = f"Provide safety recommendations for {zone_id} at {risk_level} risk."
        if self.ready:
            try:
                resp = self.model.generate_content(prompt)
                return resp.text.strip()
            except Exception as e:
                logger.error(f"Gemini error: {e}")
        if risk_level == "HIGH":
            return "Evacuate personnel, install temporary barriers, conduct immediate geotechnical inspection."
        if risk_level == "MODERATE":
            return "Increase monitoring frequency, restrict heavy vehicle access, inspect drainage."
        return "Routine monitoring and good housekeeping of benches and drains."
