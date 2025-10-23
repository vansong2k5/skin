"""Interface with the Gemini API for multimodal dermatology analysis."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional
import requests

from backend.config.config import GEMINI_API_KEY

# ==============================================================
# CONFIGURATION
# ==============================================================

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
HEADERS = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}


# ==============================================================
# CUSTOM ERROR
# ==============================================================

class ModelHTTPError(RuntimeError):
    """Raised when the Gemini API returns an HTTP error status code."""


# ==============================================================
# REQUEST WRAPPER
# ==============================================================

def _post(payload: Dict[str, Any], timeout: int = 40, retries: int = 2) -> Dict[str, Any]:
    """Perform a POST request with a retry mechanism."""
    for attempt in range(retries + 1):
        response = requests.post(GEMINI_URL, headers=HEADERS, json=payload, timeout=timeout)
        if response.status_code in (429, 503) and attempt < retries:
            time.sleep(1.5 * (attempt + 1))
            continue
        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise ModelHTTPError(f"{response.status_code} {detail}")
        return response.json()
    raise ModelHTTPError("Gemini request failed after retries")


# ==============================================================
# RESPONSE SCHEMAS (fixed)
# ==============================================================

def _schema_upper() -> Dict[str, Any]:
    """Uppercase variant of schema (for legacy compatibility)."""
    return {
        "type": "OBJECT",
        "properties": {
            "diagnosis": {"type": "STRING"},
            "confidence": {"type": "NUMBER", "minimum": 0, "maximum": 1},
            "next_steps": {"type": "ARRAY", "items": {"type": "STRING"}},
            "disclaimer": {"type": "STRING"},
            "differential_diagnoses": {"type": "ARRAY", "items": {"type": "STRING"}},
            "severity": {"type": "STRING", "enum": ["mild", "moderate", "severe"]},
            "red_flags": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["diagnosis", "confidence", "next_steps", "disclaimer"],
    }


def _schema_lower() -> Dict[str, Any]:
    """Lowercase variant of schema (used when retrying)."""
    return {
        "type": "object",
        "properties": {
            "diagnosis": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "next_steps": {"type": "array", "items": {"type": "string"}},
            "disclaimer": {"type": "string"},
            "differential_diagnoses": {"type": "array", "items": {"type": "string"}},
            "severity": {"type": "string", "enum": ["mild", "moderate", "severe"]},
            "red_flags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["diagnosis", "confidence", "next_steps", "disclaimer"],
    }


# ==============================================================
# HELPERS
# ==============================================================

def _extract_text(api_json: Dict[str, Any]) -> str:
    """Extract the model text output from Gemini JSON response."""
    try:
        parts = api_json.get("candidates", [])[0].get("content", {}).get("parts", [])
        texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        return "\n".join(text for text in texts if text).strip()
    except Exception:
        return ""


# ==============================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================

def analyze_skin_json(
    description: str,
    image_base64: str,
    mime: str = "image/jpeg",
    image_features: Optional[Dict[str, Any]] = None,
    timeout: int = 40,
) -> Dict[str, Any]:
    """Call Gemini with both the raw image and structured feature summary."""

    parts: list[Dict[str, Any]] = [
        {
            "text": (
                "Bạn là bác sĩ da liễu. Trả lời bằng TIẾNG VIỆT và luôn trả về "
                "JSON hợp lệ đúng với schema yêu cầu."
            )
        },
        {"inlineData": {"mimeType": mime, "data": image_base64}},
        {"text": f"Mô tả của bệnh nhân: {description}"},
    ]

    if image_features:
        feature_payload = json.dumps(image_features, ensure_ascii=False)
        parts.append({
            "text": (
                "Các đặc trưng hình ảnh do mô hình thị giác cung cấp (dạng JSON): "
                f"{feature_payload}. Hãy sử dụng thông tin này để đánh giá mức độ."
            )
        })

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _schema_upper(),  # clean schema
        },
    }

    # Try primary schema; fallback to lowercase if API rejects uppercase
    try:
        response_json = _post(payload, timeout=timeout)
    except ModelHTTPError:
        payload["generationConfig"]["responseSchema"] = _schema_lower()
        response_json = _post(payload, timeout=timeout)

    # Extract model output
    text = _extract_text(response_json)
    output: Dict[str, Any] = {}

    if text:
        try:
            output = json.loads(text)
        except Exception:
            output = {"raw_text": text}

    # Metadata enrichment
    output.setdefault("_meta", {})
    output["_meta"]["responseId"] = response_json.get("responseId")
    output["_meta"]["modelVersion"] = response_json.get("modelVersion")
    output["_meta"]["usageMetadata"] = response_json.get("usageMetadata", {})
    if image_features:
        output["_meta"]["image_features"] = image_features

    return output


# ==============================================================
# PING HELPER
# ==============================================================

def analyze_text_ping(prompt: str = "ping", timeout: int = 20) -> Dict[str, Any]:
    """Small helper used by the health-check endpoint to ping Gemini."""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(GEMINI_URL, headers=HEADERS, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return {
        "responseId": data.get("responseId"),
        "modelVersion": data.get("modelVersion"),
        "usageMetadata": data.get("usageMetadata", {}),
    }
