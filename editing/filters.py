# editing/filters.py
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

def _clip01(arr):
    np.clip(arr, 0.0, 1.0, out=arr)
    return arr

def adjust_light_color(img: Image.Image, params: dict) -> Image.Image:
    """
    params:
      exposure, contrast, whites, blacks  (-100..100)
      saturation, vibrance, temperature   (-100..100)
      sharpness (0..100), dehaze (0..100), grain (0..100)
    """
    E = lambda k, d=0: float(params.get(k, d))

    # Work in RGB float
    im = img.convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0

    # EXPOSURE: Global brightness adjustment
    exposure = E("exposure") / 100.0
    if abs(exposure) > 1e-6:
        arr *= max(0.1, min(3.0, 1.0 + exposure))
        _clip01(arr)

    # WHITES: Affect highlights (bright areas) - Lightroom style
    whites = E("whites") / 100.0
    if abs(whites) > 1e-6:
        # Create highlight mask (pixels > 0.7 are considered highlights)
        highlight_mask = np.maximum(0, (arr - 0.7) / 0.3)  # 0 to 1 for pixels 0.7-1.0
        highlight_mask = np.power(highlight_mask, 0.5)  # Softer falloff
        whites_adjustment = 1.0 + (whites * 0.8)  # Scale factor
        # Apply whites adjustment only to highlights
        for c in range(3):  # RGB channels
            arr[..., c] += highlight_mask[..., c] * whites * 0.3

    # BLACKS: Affect shadows (dark areas) - Lightroom style  
    blacks = E("blacks") / 100.0
    if abs(blacks) > 1e-6:
        # Create shadow mask (pixels < 0.3 are considered shadows)
        shadow_mask = np.maximum(0, (0.3 - arr) / 0.3)  # 1 to 0 for pixels 0-0.3
        shadow_mask = np.power(shadow_mask, 0.5)  # Softer falloff
        # Apply blacks adjustment only to shadows
        for c in range(3):  # RGB channels
            arr[..., c] += shadow_mask[..., c] * blacks * 0.3

    _clip01(arr)

    # CONTRAST: Linear contrast around midpoint
    contrast = E("contrast") / 100.0
    if abs(contrast) > 1e-6:
        c = max(0.1, min(4.0, 1.0 + contrast))
        arr = ((arr - 0.5) * c) + 0.5
        _clip01(arr)

    # TEMPERATURE: Color temperature shift
    temp = E("temperature") / 100.0
    if abs(temp) > 1e-6:
        # Positive = warmer (more red, less blue)
        arr[..., 0] *= (1.0 + 0.15 * temp)  # Red
        arr[..., 2] *= (1.0 - 0.15 * temp)  # Blue
        _clip01(arr)

    # VIBRANCE & SATURATION
    vib = E("vibrance") / 100.0
    sat = E("saturation") / 100.0
    if abs(vib) > 1e-6 or abs(sat) > 1e-6:
        # Convert to HSL-ish via max/min
        mx = arr.max(axis=-1, keepdims=True)
        mn = arr.min(axis=-1, keepdims=True)
        l = (mx + mn) / 2.0
        s_curr = (mx - mn)
        
        # Vibrance: stronger effect on less saturated pixels
        vib_factor = 1.0 + vib * (1.0 - s_curr * 2.0)
        sat_factor = 1.0 + sat
        
        factor = np.clip(vib_factor * sat_factor, 0.0, 5.0)
        arr = l + (arr - l) * factor
        _clip01(arr)

    # Convert back to PIL Image
    out = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), "RGB")

    # DETAIL adjustments (PIL filters)
    sharp = E("sharpness") / 100.0
    if sharp > 0:
        out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=int(150*sharp), threshold=2))

    dehaze = E("dehaze") / 100.0
    if dehaze > 0:
        out = ImageEnhance.Contrast(out).enhance(1.0 + 0.6*dehaze)
        out = ImageEnhance.Brightness(out).enhance(1.0 - 0.1*max(0.0, dehaze-0.5))

    # GRAIN: Add luminance noise
    grain = E("grain") / 100.0
    if grain > 0:
        a = np.asarray(out).astype(np.float32)
        noise = (np.random.randn(*a.shape[:2], 1) * 255.0 * 0.12 * grain).astype(np.float32)
        a = np.clip(a + noise, 0, 255).astype(np.uint8)
        out = Image.fromarray(a, "RGB")

    return out
