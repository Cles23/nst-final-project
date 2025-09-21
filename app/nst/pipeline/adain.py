# app/nst/pipeline/adain.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ---------- Configuration ----------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

@dataclass
class AdaINParams:
    alpha: float = 2.5  # style strength [0, 1]
    steps: int = 200          
    tv_weight: float = 1e-6  # total variation regularization weight   

# ---------- Utilities ----------
def tensor_from_pil(pil_image: Image.Image, device: torch.device) -> torch.Tensor:
    """Convert PIL image to tensor in [0,1] range."""
    return transforms.ToTensor()(pil_image).unsqueeze(0).to(device)

def pil_from_tensor(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor back to PIL image."""
    tensor = tensor.detach().clamp(0, 1).cpu().squeeze(0)
    return transforms.ToPILImage()(tensor)

def normalize_for_vgg(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor for VGG input."""
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std

def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """Total variation regularizer to reduce noise."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return (dx.abs().mean() + dy.abs().mean())

# ---------- VGG Encoder ----------
class VGGEncoder(nn.Module):
    """VGG19 encoder up to relu4_1 (layer 21)."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.features = nn.Sequential(*list(vgg.children())[:21]).eval()
        
        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

# ---------- AdaIN Operation ----------
def adaptive_instance_normalization(content_features: torch.Tensor, 
                                  style_features: torch.Tensor, 
                                  eps: float = 1e-5) -> torch.Tensor:
    """Apply AdaIN: transfer style statistics to content features."""
    # Calculate statistics
    content_mean = content_features.mean(dim=(2, 3), keepdim=True)
    content_std = content_features.std(dim=(2, 3), keepdim=True) + eps
    style_mean = style_features.mean(dim=(2, 3), keepdim=True)
    style_std = style_features.std(dim=(2, 3), keepdim=True) + eps
    
    # Normalize content and apply style statistics
    normalized = (content_features - content_mean) / content_std
    return normalized * style_std + style_mean

# ---------- Main AdaIN Style Transfer ----------
class AdaINStyleTransfer:
    """AdaIN style transfer using feature target optimization."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"AdaIN Feature Optimization initialized on {self.device}")
        
        # Initialize VGG encoder
        self.encoder = VGGEncoder().to(self.device)
        
        print("AdaIN ready")

    def _prepare_image(self, pil_image: Image.Image, max_size: int = 512) -> Image.Image:
        """Resize image while preserving aspect ratio."""
        if max(pil_image.size) > max_size:
            scale = max_size / max(pil_image.size)
            new_size = tuple(int(dim * scale) for dim in pil_image.size)
            return pil_image.resize(new_size, Image.LANCZOS)
        return pil_image

    def run(self, content_image: Image.Image, style_image_path: str, 
            parameters: Optional[AdaINParams] = None) -> Image.Image:
        """Main entry point for style transfer."""
        from app.nst.utils.nst_image_utils import load_pil_image
        style_image = load_pil_image(style_image_path)
        return self.stylise(content_image, style_image, parameters)

    def stylise(self, content_image: Image.Image, style_image: Image.Image, 
                parameters: Optional[AdaINParams] = None) -> Image.Image:
        """Perform AdaIN style transfer using feature optimization."""
        params = parameters or AdaINParams()
        original_size = content_image.size
        
        print(f"Running AdaIN Feature Optimization (alpha={params.alpha:.2f})...")
        
        # Resize images for processing
        max_size = 512  # Balance between quality and speed
        content_resized = self._prepare_image(content_image, max_size)
        style_resized = self._prepare_image(style_image, max_size)
        
        print(f"Processing at {content_resized.size}")
        
        # Convert to tensors
        content_tensor = tensor_from_pil(content_resized, self.device)
        style_tensor = tensor_from_pil(style_resized, self.device)
        
        # Normalize for VGG
        content_normalized = normalize_for_vgg(content_tensor)
        style_normalized = normalize_for_vgg(style_tensor)
        
        # Extract features and apply AdaIN
        with torch.no_grad():
            content_features = self.encoder(content_normalized)
            style_features = self.encoder(style_normalized)
            target_features = adaptive_instance_normalization(content_features, style_features)
            
            # Blend with original content based on alpha
            target_features = params.alpha * target_features + (1.0 - params.alpha) * content_features
        
        # Optimize pixels to match target features
        print("Optimizing pixels to match target features...")
        result_tensor = self._optimize_pixels(content_tensor, target_features, steps=300)
        
        # Convert back to PIL
        result_image = pil_from_tensor(result_tensor)
        
        # Resize back to original size
        if result_image.size != original_size:
            result_image = result_image.resize(original_size, Image.LANCZOS)
            print(f"Resized result back to {original_size}")
        
        print("AdaIN Feature Optimization completed")
        return result_image

    def _optimize_pixels(self, initial_image: torch.Tensor, target_features: torch.Tensor, 
                        steps: int = 200) -> torch.Tensor:
        """Optimize pixel values to match target features."""
        # Start from content image
        optimized_image = initial_image.clone().detach().requires_grad_(True)
        
        # Use Adam optimizer for stable convergence
        optimizer = torch.optim.Adam([optimized_image], lr=0.02)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Ensure pixels stay in valid range
            current_image = optimized_image.clamp(0, 1)
            
            # Get features from current image
            current_normalized = normalize_for_vgg(current_image)
            current_features = self.encoder(current_normalized)
            
            # Feature matching loss
            feature_loss = F.mse_loss(current_features, target_features)
            
            # Total variation regularization (reduces noise)
            tv_loss = total_variation_loss(current_image) * 1e-6
            
            # Total loss
            total_loss = feature_loss + tv_loss
            
            # Backprop and optimize
            total_loss.backward()
            optimizer.step()
            
            # Progress logging
            if (step + 1) % 50 == 0 or step == 0:
                print(f"  Step {step + 1:3d}/{steps}: loss={total_loss.item():.6f} "
                      f"(feature={feature_loss.item():.6f}, tv={tv_loss.item():.8f})")
        
        return optimized_image.detach().clamp(0, 1)
