
import torch

def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.size()
    f = feat.view(c, h*w)
    G = torch.mm(f, f.t())
    return G / (c*h*w)
