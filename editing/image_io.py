# editing/image_io.py
from PIL import Image, ImageOps
import io, os

ALLOWED = {'.png', '.jpg', '.jpeg', '.webp'}

def allowed(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED

def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img

def save_image(img: Image.Image, path: str, fmt: str = None, quality: int = 95):
    fmt = (fmt or os.path.splitext(path)[1][1:] or "PNG").upper()
    params = {}
    if fmt in ("JPG", "JPEG"):
        params.update({"quality": int(quality), "subsampling": 2, "optimize": True})
        fmt = "JPEG"
    elif fmt == "WEBP":
        params.update({"quality": int(quality)})
    img.save(path, fmt, **params)

def to_bytes(img: Image.Image, fmt: str = "PNG", quality: int = 95) -> bytes:
    buf = io.BytesIO()
    params = {}
    if fmt.upper() in ("JPG", "JPEG"):
        params.update({"quality": int(quality), "subsampling": 2, "optimize": True})
        fmt = "JPEG"
    elif fmt.upper() == "WEBP":
        params.update({"quality": int(quality)})
    img.save(buf, fmt.upper(), **params)
    buf.seek(0)
    return buf.read()

def fit_to_box(img: Image.Image, w: int, h: int) -> Image.Image:
    # contains the whole image inside (like phone editor)
    return ImageOps.contain(img, (w, h))
