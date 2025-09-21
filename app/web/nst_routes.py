from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.nst.pipeline.gatys_pipeline import GatysStyleTransfer, StyliseParams
from app.nst.pipeline.adain import AdaINStyleTransfer, AdaINParams
from app.nst.utils.nst_image_utils import load_pil_image, allowed_file
import os
import time
import threading

nst_bp = Blueprint("nst", __name__)

# Keep track of running jobs and NST engines
nst_engines = {}
active_jobs = {}

def get_nst_engine(method="gatys"):
    """Get the right NST engine or create it if needed"""
    if method not in nst_engines:
        if method == "gatys":
            nst_engines[method] = GatysStyleTransfer()
        elif method == "adain":
            nst_engines[method] = AdaINStyleTransfer()
        else:
            raise ValueError(f"Don't know method: {method}")
    return nst_engines[method]

@nst_bp.post("/api/nst/upload-style")
def upload_style():
    """Upload a style image"""
    f = request.files.get("style")
    if not f or not allowed_file(f.filename):
        return jsonify({"error": "Bad style image"}), 400
    
    # Save with timestamp to avoid conflicts
    fn = f"style_{int(time.time())}_{secure_filename(f.filename)}"
    dst = os.path.join(current_app.config["UPLOAD_FOLDER"], fn)
    f.save(dst)
    
    return jsonify({"ok": True, "path": fn})

@nst_bp.post("/api/nst/start")
def start_nst():
    """Start the neural style transfer"""
    data = request.get_json(force=True) or {}
    method = data.get("method", "gatys")
    
    # Check if method is valid
    if method not in ["gatys", "adain"]:
        return jsonify({"error": f"Unknown method: {method}"}), 400
    
    # Need a style image
    style_path = data.get("style_path")
    if not style_path:
        return jsonify({"error": "No style image"}), 400
    
    style_full_path = os.path.join(current_app.config["UPLOAD_FOLDER"], style_path)
    if not os.path.exists(style_full_path):
        return jsonify({"error": "Style image not found"}), 400
    
    # Create unique job ID
    job_id = f"nst_{method}_{int(time.time())}"
    
    # Set up parameters based on method
    if method == "gatys":
        params = StyliseParams(steps=int(data.get("steps", 300)))
    elif method == "adain":
        params = AdaINParams(alpha=float(data.get("strength", 1.0)))

    active_jobs[job_id] = {"status": "running", "method": method}
    app_ctx = current_app._get_current_object()
    
    def run_nst():
        """This runs in background thread"""
        with app_ctx.app_context():
            try:
                print(f"Starting {method.upper()} job {job_id}")
                
                # Get current image with edits applied
                preview_data = {
                    "edits": data.get("edits", {}),
                    "crop": data.get("crop", {}),
                    "max_size": 1024
                }
                
                # Get processed content image from preview API
                with app_ctx.test_client() as client:
                    response = client.post('/api/preview', json=preview_data, 
                                         headers={'Content-Type': 'application/json'})
                    
                    if response.status_code != 200:
                        raise Exception(f"Preview failed: {response.status_code}")
                    
                    # Save temporary content image
                    content_path = os.path.join(app_ctx.config["UPLOAD_FOLDER"], f"temp_{job_id}.png")
                    with open(content_path, 'wb') as f:
                        f.write(response.data)
                    
                    print(f"Saved temp content: {content_path}")
                
                # Check if file was created properly
                if not os.path.exists(content_path) or os.path.getsize(content_path) == 0:
                    raise Exception("Failed to create content file")
                
                # Load content image
                content_img = load_pil_image(content_path)
                print(f"Content loaded: {content_img.size}")
                
                # Run NST
                engine = get_nst_engine(method)
                start_time = time.time()
                
                if method == "gatys":
                    style_img = load_pil_image(style_full_path)
                    result = engine.run(content_img, style_img, params)
                elif method == "adain":
                    result = engine.run(content_img, style_full_path, params)
                
                end_time = time.time()
                print(f"{method.upper()} done in {end_time - start_time:.2f} seconds")
                
                # Save result and update cache
                result_path = os.path.join(app_ctx.config["UPLOAD_FOLDER"], f"nst_{job_id}.png")
                result.save(result_path)
                
                # Update main image cache so preview shows the result
                from app.web.routes import CACHE
                CACHE["src_path"] = result_path
                CACHE["orig_w"] = result.size[0]
                CACHE["orig_h"] = result.size[1]
                
                # Clean up temp file
                try:
                    os.remove(content_path)
                except:
                    pass  # Don't care if cleanup fails
                
                active_jobs[job_id] = {"status": "complete", "method": method}
                
            except Exception as e:
                print(f"{method.upper()} job {job_id} failed: {e}")
                active_jobs[job_id] = {"status": "error", "error": str(e), "method": method}
                
                # Clean up on error
                content_path = os.path.join(app_ctx.config["UPLOAD_FOLDER"], f"temp_{job_id}.png")
                try:
                    os.remove(content_path)
                except:
                    pass
    
    # Start background job
    threading.Thread(target=run_nst, daemon=True).start()
    return jsonify({"job_id": job_id})

@nst_bp.get("/api/nst/status/<job_id>")
def get_nst_status(job_id):
    """Check if NST job is done"""
    job = active_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify(job)