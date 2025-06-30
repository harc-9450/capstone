import sys
print("🐍 Python Path:", sys.executable)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from models.generator import Generator
from models.discriminator import Discriminator
from utils import generate_noise, save_generated_images

# Hyperparameters
latent_dim = 100
image_size = 32
batch_size = 128
epochs = 20
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms and dataset
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.CIFAR10(root="data/", download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

os.makedirs("outputs", exist_ok=True)

# Training loop
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # Labels
        real = torch.ones(batch_size, device=device)
        fake = torch.zeros(batch_size, device=device)

        # === Train Discriminator ===
        z = generate_noise(batch_size, latent_dim, device)
        gen_imgs = generator(z).detach()
        loss_real = criterion(discriminator(real_imgs), real)
        loss_fake = criterion(discriminator(gen_imgs), fake)
        loss_D = (loss_real + loss_fake) / 2

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # === Train Generator ===
        z = generate_noise(batch_size, latent_dim, device)
        gen_imgs = generator(z)
        loss_G = criterion(discriminator(gen_imgs), real)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Batch {i}/{len(dataloader)} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

    # Save samples per epoch
    save_generated_images(gen_imgs[:25], f"outputs/sample_epoch_{epoch+1}.png", nrow=5, normalize=True)

# Save final generator model
torch.save(generator.state_dict(), "outputs/generator_final.pth")