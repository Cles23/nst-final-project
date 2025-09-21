# editing/geometry.py
from PIL import Image

def apply_crop_like_phone(img: Image.Image, stage_w: int, stage_h: int,
                          zoom: float = 1.0, pan_x_pct: float = 0.0,
                          pan_y_pct: float = 0.0, rotation_deg: float = 0.0) -> Image.Image:
    """
    Simulate iPhone-style crop:
    - start by fitting the image to stage (contain),
    - then apply zoom, pan (% of stage), and rotation around center,
    - finally crop to stage box.
    """
    # 1) Fit to stage (contain - preserve aspect ratio)
    base = img.copy()
    scale = min(stage_w / base.width, stage_h / base.height)
    draw_w = int(base.width * scale)
    draw_h = int(base.height * scale)
    base = base.resize((draw_w, draw_h), Image.LANCZOS)

    # 2) Build a canvas with WHITE background (instead of transparent)
    canvas = Image.new("RGB", (stage_w, stage_h), (255, 255, 255))

    # 3) Compute transform
    # center on canvas
    cx, cy = stage_w // 2, stage_h // 2
    # pan in px
    px = int((pan_x_pct / 100.0) * stage_w)
    py = int((pan_y_pct / 100.0) * stage_h)

    # 4) scale (zoom)
    zw = max(1.0, min(5.0, float(zoom)))
    scaled = base.resize((int(draw_w * zw), int(draw_h * zw)), Image.LANCZOS)

    # 5) rotate around center
    rot = (rotation_deg or 0.0) % 360
    if abs(rot) > 0.1:
        rotated = scaled.rotate(rot, resample=Image.BICUBIC, expand=True)
    else:
        rotated = scaled

    # 6) paste centered + pan
    x = cx - rotated.width // 2 + px
    y = cy - rotated.height // 2 + py
    
    # Paste with bounds checking
    if x >= 0 and y >= 0 and x + rotated.width <= stage_w and y + rotated.height <= stage_h:
        # Simple case: image fits entirely
        canvas.paste(rotated, (x, y))
    else:
        # Complex case: crop the rotated image to fit canvas bounds
        crop_x = max(0, -x)
        crop_y = max(0, -y)
        paste_x = max(0, x)
        paste_y = max(0, y)
        
        crop_w = min(rotated.width - crop_x, stage_w - paste_x)
        crop_h = min(rotated.height - crop_y, stage_h - paste_y)
        
        if crop_w > 0 and crop_h > 0:
            cropped = rotated.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
            canvas.paste(cropped, (paste_x, paste_y))

    return canvas
