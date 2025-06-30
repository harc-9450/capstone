
import torch
from torchvision.utils import save_image
from models.generator import Generator
import os

# Config
latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
generator = Generator(latent_dim).to(device)
generator.load_state_dict(torch.load("outputs/generator_final.pth", map_location=device))
generator.eval()

# Generate one image
z = torch.randn(1, latent_dim, 1, 1, device=device)
gen_img = generator(z)

# Save
os.makedirs("inference", exist_ok=True)
save_image(gen_img, "inference/generated_sample.png", normalize=True)
print("✅ Image saved at: inference/generated_sample.png")
