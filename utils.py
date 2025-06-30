import torch
from torchvision.utils import save_image
import os

def generate_noise(batch_size, latent_dim, device):
    return torch.randn(batch_size, latent_dim, 1, 1, device=device)

def save_generated_images(images, path, nrow=5, normalize=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(images, path, nrow=nrow, normalize=normalize)