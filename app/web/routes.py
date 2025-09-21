# app/web/routes.py
from flask import Blueprint, request, send_file, render_template, current_app, jsonify, make_response
from werkzeug.utils import secure_filename
from editing.image_io import load_image, save_image, allowed, to_bytes
from editing.filters import adjust_light_color
from editing.geometry import apply_crop_like_phone
import io, os, time

bp = Blueprint("web", __name__)
CACHE = {"src_path": None, "orig_w": None, "orig_h": None}

@bp.get("/")
def index():
    return render_template("index.html")

@bp.post("/api/upload")
def api_upload():
    f = request.files.get("image")
    if not f or not allowed(f.filename):
        return jsonify({"error": "Invalid or missing image"}), 400
    fn = f"{int(time.time())}_{secure_filename(f.filename)}"
    dst = os.path.join(current_app.config["UPLOAD_FOLDER"], fn)
    f.save(dst)
    img = load_image(dst)
    CACHE.update({"src_path": dst, "orig_w": img.width, "orig_h": img.height})
    return jsonify({"ok": True, "path": fn, "w": img.width, "h": img.height})

@bp.post("/api/preview")
def api_preview():
    """Return a downscaled processed preview (fast)."""
    if not CACHE.get("src_path"):
        return jsonify({"error": "No image uploaded"}), 400

    params = request.get_json(force=True) or {}
    img = load_image(CACHE["src_path"])
    
    # Use image's natural aspect ratio for preview, but limit size
    max_w, max_h = 800, 600
    scale = min(max_w / img.width, max_h / img.height, 1.0)
    sw = int(img.width * scale)
    sh = int(img.height * scale)
    
    # apply crop-like transform
    crop = params.get("crop", {}) or {}
    img = apply_crop_like_phone(img, sw, sh,
                                zoom=float(crop.get("zoom", 1.0)),
                                pan_x_pct=float(crop.get("x", 0.0)),
                                pan_y_pct=float(crop.get("y", 0.0)),
                                rotation_deg=float(crop.get("rot", 0.0)))
    
    # edits
    edits = params.get("edits", {}) or {}
    img = adjust_light_color(img, edits)

    # Send PNG bytes
    data = to_bytes(img, "PNG")
    return make_response(data, 200, {"Content-Type": "image/png"})

@bp.post("/api/export")
def api_export():
    """Full-quality export using original image dimensions."""
    if not CACHE.get("src_path"):
        return jsonify({"error": "No image uploaded"}), 400

    params = request.get_json(force=True) or {}
    fmt = (params.get("format") or "png").lower()
    quality = int(params.get("quality", 95))

    img = load_image(CACHE["src_path"])

    # Use original image size for export (full quality)
    sw, sh = img.width, img.height

    crop = params.get("crop", {}) or {}
    img = apply_crop_like_phone(img, sw, sh,
                                zoom=float(crop.get("zoom", 1.0)),
                                pan_x_pct=float(crop.get("x", 0.0)),
                                pan_y_pct=float(crop.get("y", 0.0)),
                                rotation_deg=float(crop.get("rot", 0.0)))

    edits = params.get("edits", {}) or {}
    img = adjust_light_color(img, edits)

    buf = io.BytesIO()
    # Ensure background for JPEG
    if fmt in ("jpg", "jpeg"):
        img = img.convert("RGB")
    img.save(buf, {"png":"PNG","jpg":"JPEG","jpeg":"JPEG","webp":"WEBP"}[fmt], quality=quality)
    buf.seek(0)
    return send_file(buf, mimetype=f"image/{'jpeg' if fmt in ('jpg','jpeg') else fmt}",
                     as_attachment=True, download_name=f"edited.{fmt}")

@bp.post("/api/original")
def api_original():
    """Return the original image without any effects (for before/after comparison)."""
    if not CACHE.get("src_path"):
        return jsonify({"error": "No image uploaded"}), 400

    params = request.get_json(force=True) or {}
    img = load_image(CACHE["src_path"])
    
    # Apply ONLY crop/rotation, NO filters
    max_w, max_h = 800, 600
    scale = min(max_w / img.width, max_h / img.height, 1.0)
    sw = int(img.width * scale)
    sh = int(img.height * scale)
    
    # apply crop-like transform (but no filters)
    crop = params.get("crop", {}) or {}
    img = apply_crop_like_phone(img, sw, sh,
                                zoom=float(crop.get("zoom", 1.0)),
                                pan_x_pct=float(crop.get("x", 0.0)),
                                pan_y_pct=float(crop.get("y", 0.0)),
                                rotation_deg=float(crop.get("rot", 0.0)))
    
    # NO FILTERS APPLIED - just return the cropped original
    data = to_bytes(img, "PNG")
    return make_response(data, 200, {"Content-Type": "image/png"})

@bp.post("/api/export-preview")
def api_export_preview():
    """Return a high-quality processed preview for NST (preserves original aspect ratio)."""
    if not CACHE.get("src_path"):
        return jsonify({"error": "No image uploaded"}), 400

    params = request.get_json(force=True) or {}
    img = load_image(CACHE["src_path"])
    
    # Check if we should use original size (for NST) or scaled size
    use_original_size = params.get("use_original_size", False)
    max_size = params.get("max_size", 1024)
    
    if use_original_size:
        # Use original dimensions - no scaling
        sw, sh = img.width, img.height
        print(f"Using original dimensions: {sw}x{sh}")
    else:
        # Scale down to max_size while preserving aspect ratio
        scale = min(max_size / max(img.width, img.height), 1.0)
        sw = int(img.width * scale)
        sh = int(img.height * scale)
        print(f"Scaling to {sw}x{sh} (scale: {scale:.3f}) for NST input")
    
    # apply crop-like transform
    crop = params.get("crop", {}) or {}
    img = apply_crop_like_phone(img, sw, sh,
                                zoom=float(crop.get("zoom", 1.0)),
                                pan_x_pct=float(crop.get("x", 0.0)),
                                pan_y_pct=float(crop.get("y", 0.0)),
                                rotation_deg=float(crop.get("rot", 0.0)))
    
    # edits
    edits = params.get("edits", {}) or {}
    img = adjust_light_color(img, edits)

    # Send PNG bytes
    data = to_bytes(img, "PNG")
    return make_response(data, 200, {"Content-Type": "image/png"})
