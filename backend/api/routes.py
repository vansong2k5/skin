from flask import Blueprint, request, jsonify


from backend.utils.image_processing import encode_image_to_base64
from backend.models.ai_model import analyze_skin_json
from backend.models.image_feature_extractor import get_feature_extractor

# Không đặt url_prefix để URL là /healthz, /debug/ping_gemini, /analyze
routes = Blueprint("routes", __name__)

@routes.get("/healthz")
def healthz():
    return jsonify({"ok": True})

@routes.get("/debug/ping_gemini")
def ping_gemini():
    # gọi text-only để xác nhận kết nối
    from backend.models.ai_model import analyze_text_ping
    try:
        out = analyze_text_ping("ping")
        return jsonify({"ok": True, **out}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502

@routes.post("/analyze")
def analyze():
    desc = request.form.get("description")
    img  = request.files.get("image")
    if not desc or not img:
        return jsonify({"error": "Thiếu mô tả hoặc ảnh"}), 400
    b64, mime, size_bytes, pil_image = encode_image_to_base64(img, return_image=True)  # JPEG + base64 sạch
    if size_bytes > 12 * 1024 * 1024:
        return jsonify({"error": "Ảnh quá lớn sau khi nén"}), 413
    features = None
    try:
        extractor = get_feature_extractor()
        features = extractor.extract_features(pil_image)
    except Exception as exc:
        features = {"error": f"Image feature extraction failed: {exc}"}
    out = analyze_skin_json(desc, b64, mime, image_features=features)
    out.setdefault("_meta", {})
    out["_meta"].setdefault("image", {})
    out["_meta"]["image"].update({"mime": mime, "size_bytes": size_bytes})
    if features:
        out["_meta"]["image_features"] = features
    return jsonify(out), 200
# @app.route("/")
# def serve_index():
#     from pathlib import Path
#     frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
#     return send_from_directory(frontend_dir, "index.html")