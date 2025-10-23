# backend/models/ai_model.py
import os, json, time, requests
from backend.config.config import GEMINI_API_KEY

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL   = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
HEADERS      = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

class ModelHTTPError(RuntimeError): ...
def _post(payload, timeout=40, retries=2):
    for i in range(retries + 1):
        r = requests.post(GEMINI_URL, headers=HEADERS, json=payload, timeout=timeout)
        # retry nhẹ cho 429/503
        if r.status_code in (429, 503) and i < retries:
            time.sleep(1.5 * (i + 1)); continue
        if r.status_code >= 400:
            try: detail = r.json()
            except Exception: detail = r.text
            raise ModelHTTPError(f"{r.status_code} {detail}")
        return r.json()

def extract_json_from_gemini(api_json: dict) -> dict:
    """Parse JSON trả về trong parts.text; fallback cắt { ... } lớn nhất; nếu fail -> raw_text."""
    try:
        parts = api_json.get("candidates", [])[0].get("content", {}).get("parts", [])
        text  = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    except Exception:
        return {"error": "Empty or unexpected response structure", "raw_text": ""}

    if not text:  # rỗng/safety block
        return {"error": "Empty text from model", "raw_text": ""}

    # 1) Parse trực tiếp
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) Cắt block JSON lớn nhất
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e+1])
        except Exception:
            return {"raw_text": text, "error": "JSON parse failed on sliced payload"}
    # 3) Bó tay -> raw
    return {"raw_text": text, "error": "JSON parse failed"}

def _schema_lower():
    return {
        "type": "object",
        "properties": {
            "diagnosis":  {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "next_steps": {"type": "array", "items": {"type": "string"}},
            "disclaimer": {"type": "string"},
            # tùy chọn
            "differential_diagnoses": {"type": "array", "items": {"type": "string"}},
            "severity": {"type": "string", "enum": ["mild","moderate","severe"]},
            "red_flags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["diagnosis", "confidence", "next_steps", "disclaimer"],
        "additionalProperties": False
    }

def _schema_upper():
    return {
        "type": "OBJECT",
        "properties": {
            "diagnosis":  {"type": "STRING"},
            "confidence": {"type": "NUMBER", "minimum": 0, "maximum": 1},
            "next_steps": {"type": "ARRAY", "items": {"type": "STRING"}},
            "disclaimer": {"type": "STRING"},
            "differential_diagnoses": {"type": "ARRAY", "items": {"type": "STRING"}},
            "severity": {"type": "STRING", "enum": ["mild","moderate","severe"]},
            "red_flags": {"type": "ARRAY", "items": {"type": "STRING"}}
        },
        "required": ["diagnosis", "confidence", "next_steps", "disclaimer"],
        "additionalProperties": False
    }

# backend/models/ai_model.py
import json, requests
from backend.config.config import GEMINI_API_KEY

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

class ModelHTTPError(Exception):
    pass

def _post(payload, timeout=300, retries=0):
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    r = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=timeout)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise ModelHTTPError(f"{r.status_code} {detail}")
    return r.json()

def _schema_uc():
    # KHÔNG dùng additionalProperties
    return {
        "type": "OBJECT",
        "properties": {
            "diagnosis": {"type": "STRING"},
            "confidence": {"type": "NUMBER"},
            "next_steps": {"type": "ARRAY", "items": {"type": "STRING"}},
            "disclaimer": {"type": "STRING"},
            "differential_diagnoses": {"type": "ARRAY", "items": {"type": "STRING"}},
            "severity": {"type": "STRING", "enum": ["mild", "moderate", "severe"]},
            "red_flags": {"type": "ARRAY", "items": {"type": "STRING"}}
        },
        "required": ["diagnosis", "confidence", "next_steps", "disclaimer"]
    }

def _schema_lc():
    # fallback: phiên bản lowercase
    return {
        "type": "object",
        "properties": {
            "diagnosis": {"type": "string"},
            "confidence": {"type": "number"},
            "next_steps": {"type": "array", "items": {"type": "string"}},
            "disclaimer": {"type": "string"},
            "differential_diagnoses": {"type": "array", "items": {"type": "string"}},
            "severity": {"type": "string", "enum": ["mild", "moderate", "severe"]},
            "red_flags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["diagnosis", "confidence", "next_steps", "disclaimer"]
    }

def _extract_text(api_json: dict) -> str:
    try:
        parts = api_json.get("candidates", [])[0].get("content", {}).get("parts", [])
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "\n".join(t for t in texts if t).strip()
    except Exception:
        return ""

def analyze_skin_json(description: str, image_base64: str, mime: str = "image/jpeg", timeout: int = 40) -> dict:
    payload = {
        "contents": [{
            "parts": [
                {"text": "Bạn là bác sĩ da liễu. Trả lời TIẾNG VIỆT. Chỉ trả JSON đúng schema, không thêm chữ ngoài JSON."},
                {"inlineData": {"mimeType": mime, "data": image_base64}},
                {"text": f"Mô tả của bệnh nhân: {description}"}
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _schema_uc()  # ↑ không có additionalProperties
        }
    }
    try:
        js = _post(payload, timeout=timeout)
    except ModelHTTPError:
        # Fallback: thử schema lowercase rồi GỌI LẠI
        payload["generationConfig"]["responseSchema"] = _schema_lc()
        js = _post(payload, timeout=timeout)

    text = _extract_text(js)
    out = {}
    if text:
        try:
            out = json.loads(text)
        except Exception:
            out = {"raw_text": text}

    # Đính kèm meta để debug/quan sát
    out.setdefault("_meta", {})
    out["_meta"]["responseId"] = js.get("responseId")
    out["_meta"]["modelVersion"] = js.get("modelVersion")
    out["_meta"]["usageMetadata"] = js.get("usageMetadata", {})
    return out

# === ADD: text-only ping để kiểm tra kết nối Gemini ===
def analyze_text_ping(prompt: str = "ping", timeout: int = 20) -> dict:
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    j = resp.json()
    return {
        "responseId": j.get("responseId"),
        "modelVersion": j.get("modelVersion"),
        "usageMetadata": j.get("usageMetadata", {})
    }
