from __future__ import annotations
from PIL import Image
import torch
from torchvision import transforms
import os

ALLOWED = {"png","jpg","jpeg","webp"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit('.',1)[1].lower() in ALLOWED

def load_pil_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def pil_to_tensor(img: Image.Image, max_size: int = 512) -> torch.Tensor:
    w, h = img.size
    scale = min(max_size / max(w,h), 1.0)
    new = (int(w*scale), int(h*scale))
    tfm = transforms.Compose([
        transforms.Resize(new),
        transforms.ToTensor()
    ])
    return tfm(img).unsqueeze(0)

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().squeeze(0).clamp(0,1)
    to_pil = transforms.ToPILImage()
    return to_pil(t)

def save_pil_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
