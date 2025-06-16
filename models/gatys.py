import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os 
from torchvision.utils import save_image


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)


class VGGFeatures(nn.Module):
    def __init__(self, style_layers=None, content_layer='21'):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.model = vgg
        self.style_layers = style_layers or ['0', '5', '10', '19', '28']
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


class StyleTransfer:
    def __init__(self, content, style, device='cpu', content_weight=1e5, style_weight=1e8):
        self.device = device
        self.content = content.clone().to(device)
        self.style = style.clone().to(device)
        self.generated = content.clone().requires_grad_(True).to(device)

        self.model = VGGFeatures().to(device)
        self.optimizer = optim.Adam([self.generated], lr=0.01)

        self.content_weight = content_weight
        self.style_weight = style_weight

        self.target_content, _ = self.model(self.content)
        _, self.target_styles = self.model(self.style)
        self.target_grams = [gram_matrix(s) for s in self.target_styles]

    def run(self, steps=300, save_intermediate=False):
        for step in range(1, steps + 1):
            self.optimizer.zero_grad()

            gen_content, gen_styles = self.model(self.generated)

            content_loss = nn.functional.mse_loss(gen_content, self.target_content)
            style_loss = sum(
                nn.functional.mse_loss(gram_matrix(g), a)
                for g, a in zip(gen_styles, self.target_grams)
            )

            total_loss = self.content_weight * content_loss + self.style_weight * style_loss
            total_loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.generated.clamp_(0, 1)

            if step % 50 == 0:
                print(f"Step {step}/{steps} | Content Loss: {content_loss.item():.4f} | Style Loss: {style_loss.item():.4f}")
                if save_intermediate:
                    from torchvision.utils import save_image
                    save_image(self.generated, f"output_step_{step}.png")

        return self.generated.detach()

