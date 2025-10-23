import os
from pathlib import Path

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from backend.api.routes import routes

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
CORS(app)

# ĐĂNG KÝ BLUEPRINT
app.register_blueprint(routes)

# Route liệt kê toàn bộ URL để kiểm nhanh
@app.get("/__routes")
def list_routes():
    urls = []
    for rule in app.url_map.iter_rules():
        methods = ",".join(sorted(m for m in rule.methods if m in {"GET","POST","PUT","DELETE","PATCH"}))
        urls.append({"rule": str(rule), "endpoint": rule.endpoint, "methods": methods})
    return jsonify(urls)

@app.route("/")
def serve_index():
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    return send_from_directory(frontend_dir, "index.html")

if __name__ == "__main__":
    # Chạy cổng 5001 để tránh nhầm server khác đang chiếm 5000
    app.run(debug=True, port=5001)
