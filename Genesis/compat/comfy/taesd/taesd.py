"""
TAESD (Tiny AutoEncoder for Stable Diffusion) stub
Author: eddy
"""

import torch
import torch.nn as nn

class TAESD:
    """TAESD stub for latent preview"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def decode(self, latent):
        """Decode latent to image (stub implementation)"""
        # Just return a dummy image tensor
        if isinstance(latent, torch.Tensor):
            b = latent.shape[0] if len(latent.shape) > 3 else 1
            return torch.zeros(b, 3, 256, 256, device=self.device)
        return None

    def encode(self, image):
        """Encode image to latent (stub implementation)"""
        # Just return a dummy latent tensor
        if isinstance(image, torch.Tensor):
            b = image.shape[0] if len(image.shape) > 3 else 1
            return torch.zeros(b, 4, 32, 32, device=self.device)
        return None