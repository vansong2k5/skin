from flask import Blueprint, request, jsonify, send_from_directory


from backend.utils.image_processing import encode_image_to_base64
from backend.models.ai_model import analyze_skin_json

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
    b64, mime, size_bytes = encode_image_to_base64(img)  # JPEG + base64 sạch
    if size_bytes > 12 * 1024 * 1024:
        return jsonify({"error": "Ảnh quá lớn sau khi nén"}), 413
    out = analyze_skin_json(desc, b64, mime)
    return jsonify(out), 200
# @app.route("/")
# def serve_index():
#     from pathlib import Path
#     frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
#     return send_from_directory(frontend_dir, "index.html")