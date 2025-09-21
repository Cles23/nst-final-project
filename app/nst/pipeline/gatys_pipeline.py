from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import time

def gram_matrix(tensor):
    """Calculate gram matrix - this captures the style patterns"""
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)

class VGGFeatures(nn.Module):
    def __init__(self, style_layers=None, content_layer='21'):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
        # Don't train the VGG weights
        for param in vgg.parameters():
            param.requires_grad = False
        self.model = vgg
        # These layers capture different levels of style
        self.style_layers = style_layers or ['0', '5', '10', '19', '28']
        # This layer captures content
        self.content_layer = content_layer

    def forward(self, x):
        content_feature = None
        style_features = []
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name == self.content_layer:
                content_feature = x
            if name in self.style_layers:
                style_features.append(x)
        return content_feature, style_features

@dataclass
class StyliseParams:
    steps: int = 1000  # how many optimization steps

class StyleTransfer:
    """The original Gatys method - slow but high quality"""
    def __init__(self, content, style, device='cpu', content_weight=1e5, style_weight=1e8):
        self.device = device
        self.content = content.clone().to(device)
        self.style = style.clone().to(device)
        # Start optimization from content image
        self.generated = content.clone().requires_grad_(True).to(device)

        self.model = VGGFeatures().to(device)
        self.optimizer = optim.Adam([self.generated], lr=0.01)

        # How much to care about content vs style
        self.content_weight = content_weight
        self.style_weight = style_weight

        # Get target features that we want to match
        self.target_content, _ = self.model(self.content)
        _, self.target_styles = self.model(self.style)
        # Convert style features to gram matrices
        self.target_grams = [gram_matrix(s) for s in self.target_styles]

    def run(self, steps=1000, save_intermediate=False):
        for step in range(1, steps + 1):
            self.optimizer.zero_grad()

            # Get features from current generated image
            gen_content, gen_styles = self.model(self.generated)

            # How different is content from what we want?
            content_loss = nn.functional.mse_loss(gen_content, self.target_content)
            
            # How different is style from what we want?
            style_loss = sum(
                nn.functional.mse_loss(gram_matrix(g), a)
                for g, a in zip(gen_styles, self.target_grams)
            )

            # Total loss combines both
            total_loss = self.content_weight * content_loss + self.style_weight * style_loss
            total_loss.backward()
            self.optimizer.step()

            # Keep pixels in valid range
            with torch.no_grad():
                self.generated.clamp_(0, 1)

            # Show progress
            if step % 50 == 0:
                print(f"Step {step}/{steps} | Content: {content_loss.item():.4f} | Style: {style_loss.item():.4f}")

        return self.generated.detach()

class GatysStyleTransfer:
    """Simple wrapper to make Gatys work with the app"""
    def __init__(self, device: str | None = None):
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        # Make images smaller for faster processing
        if max(img.size) > 512:
            scale = 512 / max(img.size)
            new_size = tuple(int(dim * scale) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)
        
        transform = transforms.ToTensor()
        return transform(img).unsqueeze(0)

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.detach().cpu().squeeze(0).clamp(0, 1)
        return transforms.ToPILImage()(tensor)

    def run(self, content_pil: Image.Image, style_pil: Image.Image, params: StyliseParams) -> Image.Image:
        print(f"Starting Gatys NST on {self.device}")
        
        # Remember original size
        original_size = content_pil.size
        
        # Convert to tensors
        content_tensor = self.pil_to_tensor(content_pil).to(self.device)
        style_tensor = self.pil_to_tensor(style_pil).to(self.device)
        
        # Do the style transfer
        style_transfer = StyleTransfer(
            content_tensor, style_tensor, self.device,
            content_weight=1e4, style_weight=1e16
        )
        
        result_tensor = style_transfer.run(steps=params.steps)
        result_pil = self.tensor_to_pil(result_tensor)
        
        # Make it original size again
        if result_pil.size != original_size:
            result_pil = result_pil.resize(original_size, Image.LANCZOS)
        
        return result_pil
