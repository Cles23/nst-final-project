#!/usr/bin/env python3
import argparse, csv, itertools, os, time, sys, platform, traceback
from datetime import datetime
from PIL import Image

import torch
from app.nst.pipeline.gatys_pipeline import GatysStyleTransfer, StyliseParams
from app.nst.pipeline.adain import AdaINStyleTransfer, AdaINParams

# Optional metric (SSIM). If not installed, it’ll skip gracefully.
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except Exception:
    HAS_SSIM = False

def list_images(folder, exts=(".png",".jpg",".jpeg",".webp",".bmp")):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(exts)]

def load_pil(path, max_side):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = max(w, h) / float(max_side)
    if s > 1:
        img = img.resize((int(round(w/s)), int(round(h/s))), Image.LANCZOS)
    return img

def ensure_dirs(*paths):
    for p in paths: os.makedirs(p, exist_ok=True)

def time_it(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    ms = (time.perf_counter() - t0) * 1000.0
    return out, ms

def compute_ssim(imgA, imgB):
    if not HAS_SSIM: return None
    # SSIM expects grayscale or same-sized RGB; convert to Y
    import numpy as np
    a = np.array(imgA.convert("L"))
    b = np.array(imgB.convert("L"))
    a, b = a[:min(a.shape[0], b.shape[0]), :min(a.shape[1], b.shape[1])], b[:min(a.shape[0], b.shape[0]), :min(a.shape[1], b.shape[1])]
    return float(ssim(a, b, data_range=255))

def mosaic_4(content, style, out_gatys, out_adain, save_path, pad=12, bg=(20,20,20)):
    # Build a 4-column mosaic: Content | Style | Gatys | AdaIN
    W = max(im.width for im in (content, style, out_gatys, out_adain))
    H = max(im.height for im in (content, style, out_gatys, out_adain))
    cols, rows = 4, 1
    mw = cols*W + (cols+1)*pad
    mh = rows*H + (rows+1)*pad
    canvas = Image.new("RGB", (mw, mh), bg)
    x = pad
    for im in (content, style, out_gatys, out_adain):
        # center each
        ox = x + (W - im.width)//2
        oy = pad + (H - im.height)//2
        canvas.paste(im, (ox, oy))
        x += W + pad
    canvas.save(save_path)

def main():
    ap = argparse.ArgumentParser("Benchmark Gatys vs AdaIN (optimisation version)")
    ap.add_argument("--content_dir", default="content")
    ap.add_argument("--style_dir",   default="style")
    ap.add_argument("--out_dir",     default="results/outputs")
    ap.add_argument("--mosaic_dir",  default="results/mosaics")
    ap.add_argument("--log_csv",     default="results/logs/bench.csv")
    ap.add_argument("--max_side", type=int, default=720, help="Resize longest side for input PILs")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--gatys_steps", type=int, default=400)
    ap.add_argument("--adain_steps", type=int, default=400)
    ap.add_argument("--adain_alpha", type=float, default=3)
    ap.add_argument("--tv_weight", type=float, default=1e10)
    ap.add_argument("--opt", choices=["adam","lbfgs"], default="adam")
    ap.add_argument("--limit_pairs", type=int, default=0, help="0 means all combinations")
    args = ap.parse_args()

    ensure_dirs(args.out_dir, args.mosaic_dir, os.path.dirname(args.log_csv))

    dev = ("cuda" if (args.device=="auto" and torch.cuda.is_available()) else
           "cpu"  if (args.device=="auto") else args.device)
    device = torch.device(dev)

    # Log environment
    print(f"Device: {device}, torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
    print(f"Max side: {args.max_side}; Gatys steps: {args.gatys_steps}; AdaIN steps: {args.adain_steps}; alpha={args.adain_alpha}")

    contents = list_images(args.content_dir)
    styles   = list_images(args.style_dir)
    if not contents or not styles:
        print("No images found. Check test_images/content/ and test_images/style/ folders.")
        sys.exit(1)

    pairs = list(itertools.product(contents, styles))
    if args.limit_pairs > 0:
        pairs = pairs[:args.limit_pairs]

    # Initialise engines once
    gatys_engine = GatysStyleTransfer(device=str(device))
    adain_engine = AdaINStyleTransfer(device=str(device))

    # CSV header
    fieldnames = [
        "timestamp","algo","device","max_side","content","style",
        "params","time_ms","ssim_content","ssim_style"
    ]
    new_file = not os.path.exists(args.log_csv)
    with open(args.log_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file: writer.writeheader()

        for i, (c_path, s_path) in enumerate(pairs, 1):
            c_name = os.path.splitext(os.path.basename(c_path))[0]
            s_name = os.path.splitext(os.path.basename(s_path))[0]
            tag = f"{c_name}__{s_name}"
            print(f"[{i}/{len(pairs)}] {tag}")

            try:
                c_pil = load_pil(c_path, args.max_side)
                s_pil = load_pil(s_path, args.max_side)

                # --- Gatys ---
                g_params = StyliseParams(steps=args.gatys_steps)
                g_out, g_ms = time_it(gatys_engine.run, c_pil, s_pil, g_params)
                g_out_path = os.path.join(args.out_dir, f"{tag}__gatys.png")
                g_out.save(g_out_path)

                writer.writerow({
                    "timestamp": datetime.utcnow().isoformat(),
                    "algo": "gatys",
                    "device": device.type,
                    "max_side": args.max_side,
                    "content": c_path,
                    "style": s_path,
                    "params": f"steps={args.gatys_steps}",
                    "time_ms": f"{g_ms:.2f}",
                    "ssim_content": f"{compute_ssim(c_pil, g_out):.4f}" if HAS_SSIM else "",
                    "ssim_style":   f"{compute_ssim(s_pil, g_out):.4f}" if HAS_SSIM else "",
                }); f.flush()

                # --- AdaIN (optimisation) ---
                a_params = AdaINParams(alpha=args.adain_alpha, steps=args.adain_steps,
                                       tv_weight=args.tv_weight)
                a_out, a_ms = time_it(adain_engine.stylise, c_pil, s_pil, a_params)
                a_out_path = os.path.join(args.out_dir, f"{tag}__adain.png")
                a_out.save(a_out_path)

                writer.writerow({
                    "timestamp": datetime.utcnow().isoformat(),
                    "algo": "adain_opt",
                    "device": device.type,
                    "max_side": args.max_side,
                    "content": c_path,
                    "style": s_path,
                    "params": f"alpha={args.adain_alpha},steps={args.adain_steps},opt={args.opt},tv={args.tv_weight}",
                    "time_ms": f"{a_ms:.2f}",
                    "ssim_content": f"{compute_ssim(c_pil, a_out):.4f}" if HAS_SSIM else "",
                    "ssim_style":   f"{compute_ssim(s_pil, a_out):.4f}" if HAS_SSIM else "",
                }); f.flush()

                # --- Mosaic for this pair ---
                mosaic_path = os.path.join(args.mosaic_dir, f"{tag}.png")
                mosaic_4(c_pil, s_pil, g_out, a_out, mosaic_path)

            except Exception as e:
                print("ERROR on pair:", tag)
                traceback.print_exc()
                # Continue to next pair

    print("\nDone. CSV:", args.log_csv)
    print("Outputs in:", args.out_dir)
    print("Mosaics in:", args.mosaic_dir)

if __name__ == "__main__":
    main()
