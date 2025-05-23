"""
Script for training an autoencoder/VAE for reconstructoin and image generation with CelebA dataset.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from torch import nn, optim
import torch.nn.functional as F # <-- IMPORT F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image # For better image saving
from typing import Tuple, List


# Use VariationalAutoencoder for VAE tasks
from model import VariationalAutoencoder 

torch.manual_seed(445)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_celeba_dataset(
    data_dir: str,
    image_size: int = 64
) -> datasets.ImageFolder:
    """
    Returns a torchvision ImageFolder dataset for unlabeled CelebA images.
    Assumes all your .jpg files live under data_dir/<some_folder>/*.jpg.
    """
    transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),                       # [0,1]
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # → [–1,1]
    ])
    # …and in your decoder’s last layer:
    nn.Tanh()   # output in [–1,1]


    # If your images are directly in data_dir/, wrap them in a dummy class folder:
    # e.g. data_dir/all_faces/*.jpg
    return datasets.ImageFolder(root=data_dir, transform=transform)

def make_celeba_loaders(
    data_dir: str,
    batch_size: int = 128,
    image_size: int = 64,
    train_frac: float = 0.8,
    num_workers: int = 4,
    device_is_cuda: bool = False,
    seed: int = 42
):
    """
    Loads CelebA images from data_dir, splits into train/test,
    and returns (train_loader, test_loader).
    """
    # 1) Build the full dataset
    full_dataset = make_celeba_dataset(data_dir, image_size)

    # 2) Compute split sizes
    total = len(full_dataset)
    train_size = int(train_frac * total)
    test_size = total - train_size

    # 3) Split
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # 4) Wrap in DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device_is_cuda
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device_is_cuda
    )

    return train_loader, test_loader

def retrain(path: str) -> bool:
    train_from_scratch = True
    if os.path.exists(path):
        load_model = None
        while load_model not in ["y", "n"]:
            load_model = input(f"Found a saved model in {path}. Do you want to use this model? (y/n) ")
            load_model = load_model.lower().replace(" ", "")
        train_from_scratch = load_model == "n"
    return train_from_scratch

def plot_vae_results(
    original_imgs: torch.Tensor,
    reconstructed_imgs: torch.Tensor,
    mu_latents: torch.Tensor,      # unused here, but could be plotted as a heatmap below
    filename: str,
    n_display: int = 10,
    latent_display_dim: Tuple[int, int] = (8, 8)
) -> None:
    """
    Plot original and reconstructed images side by side for an RGB VAE.
    """
    # Move to CPU / NumPy
    original_imgs      = original_imgs.cpu().detach()
    reconstructed_imgs = reconstructed_imgs.cpu().detach()

    # Create subplots: 2 rows × n_display columns
    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_display,
        figsize=(n_display * 2, 4),    # 2 inches per image, 4 inches tall
        constrained_layout=True
    )

    for i in range(n_display):
        # --------------------
        # 1) Original image
        # --------------------
        img = original_imgs[i].numpy()           # (3, 64, 64)
        img = np.transpose(img, (1, 2, 0))       # (64, 64, 3)
        img = (img + 1) / 2.0                    # Rescale to [0, 1]
        img = np.clip(img, 0, 1)                 # Clip to avoid artifacts
        ax = axes[0, i]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Original", fontsize=12)

        # --------------------
        # 2) Reconstructed image
        # --------------------
        recon = reconstructed_imgs[i].numpy()
        recon = np.transpose(recon, (1, 2, 0))
        recon = (recon + 1) / 2.0
        recon = np.clip(recon, 0, 1)
        ax = axes[1, i]
        ax.imshow(recon)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Reconstructed", fontsize=12)

    # Optionally, you could plot mu_latents as a grid of heatmaps in a third row:
    # if mu_latents is not None:
    #     ...  

    # Save and show
    fig.savefig(f"{filename}.png", dpi=150)
    plt.show()
    plt.close(fig)

def beta_kld(epoch, warmup=20):
    return min(1.0, epoch / warmup) # Linear warmup for beta

# --- VAE Loss Function ---
def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    # recon_x, x: (B, C, H, W)
    # Flatten per-example features:
    recon_flat = recon_x.flatten(start_dim=1)
    x_flat     = x.flatten(start_dim=1)
    recon_loss = F.mse_loss(recon_flat, x_flat, reduction='sum')
    
    # KL divergence (same as before)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * KLD

def train_vae(
    vae_model: nn.Module,
    trainloader: DataLoader,
    optimizer: optim.Optimizer,
    epochs: int = 10,
    beta_kld: float = 2.0
) -> nn.Module:
    vae_model.to(DEVICE)
    vae_model.train()

    for epoch in range(epochs):
        total_loss = 0
        recon_loss_total = 0
        kld_loss_total = 0

        # leave=False so old bars don’t pile up
        tbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for images, _ in tbar:
            images = images.to(DEVICE)

            optimizer.zero_grad()
            recon_batch, mu, logvar = vae_model(images)

            loss = vae_loss_function(recon_batch, images, mu, logvar, beta=beta_kld)
            loss.backward()
            optimizer.step()

            # ----- Logging per batch -----
            with torch.no_grad():
                recon_flat = recon_batch.flatten(start_dim=1)
                img_flat   = images.flatten(start_dim=1)

                MSE_item = F.mse_loss(recon_flat, img_flat, reduction='sum').item()
                KLD_item = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()

            total_loss       += loss.item()
            recon_loss_total += MSE_item
            kld_loss_total   += KLD_item

            tbar.set_postfix(
                loss=loss.item()/len(images),
                recon=MSE_item/len(images),
                KLD=KLD_item/len(images)
            )
            # ----- End per-batch logging -----

        # Epoch summary
        avg_loss = total_loss       / len(trainloader.dataset)
        avg_recon_loss = recon_loss_total / len(trainloader.dataset)
        avg_kld_loss = kld_loss_total   / len(trainloader.dataset)
        print(f'====> Epoch: {epoch+1} '
              f'Average loss: {avg_loss:.4f} | '
              f'Recon Loss: {avg_recon_loss:.4f} | '
              f'KLD: {avg_kld_loss:.4f}')
    return vae_model

def test_vae(
    vae_model: nn.Module,
    testloader: DataLoader,
    filename: str,
    n_display: int = 10,
    latent_dim_for_plot: int = 64 # for reshaping mu
    ) -> None:
    """
    Test the VAE on a batch of images and plot the results.
    Also, generate some new samples.
    """
    vae_model.to(DEVICE)
    vae_model.eval() # Set model to evaluation mode

    with torch.no_grad():
        # 1. Test reconstruction
        x_test_batch, _ = next(iter(testloader))
        x_test_batch = x_test_batch.to(DEVICE)[:n_display] # Get a batch for display

        reconstructed_imgs, mu, log_var = vae_model(x_test_batch)

        plot_vae_results(x_test_batch, reconstructed_imgs, mu, filename + "_reconstruction", n_display=n_display)

        # 2. Generate new samples
        num_samples_to_generate = n_display * 2
        # Sample z from the prior (standard normal distribution)
        z_sample = torch.randn(num_samples_to_generate, vae_model.latent_dim).to(DEVICE)
        generated_samples = vae_model.decode(z_sample).cpu() # decode now reshapes
        
        generated_samples = (generated_samples + 1.0) / 2.0 # Rescale to [0, 1]
        generated_samples = torch.clamp(generated_samples, 0, 1) # Clip to avoid artifacts

        # Save generated samples as a grid
        save_image(generated_samples, filename + "_generated_samples.png", nrow=n_display)
        print(f"Saved generated samples to {filename}_generated_samples.png")

def part_a_vae(trainloader, testloader, latent_dim=64): # Changed to VAE
    """
    Train the Variational Autoencoder.
    """
    print(f"\n--- Training VAE (latent_dim={latent_dim}) ---")
    # Note: VariationalAutoencoder class expects in_shape=(H,W) and img_channels
    # FashionMNIST is 28x28, 1 channel
    vae_model = VariationalAutoencoder()
    model_path = os.path.join("checkpoints", f"vae_celeba_ld{latent_dim}.pth")

    if retrain(model_path):
        print("Starting training for VAE!")
        optimizer = optim.Adam(vae_model.parameters(), lr=0.001) # Adam is good for VAEs
        vae_model = train_vae(vae_model, trainloader, optimizer, epochs=30, beta_kld=1.0) # More epochs for VAE
        torch.save(vae_model.state_dict(), model_path)
        print(f"VAE model saved to {model_path}")
    else:
        vae_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded VAE model from {model_path}")
    test_vae(vae_model, testloader, f'vae_celeba_ld{latent_dim}_results', n_display=10, latent_dim_for_plot=latent_dim)
1

def main():
    os.makedirs("checkpoints", exist_ok=True)
    print(f"Using device: {DEVICE}")

    #trainset = datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
    #trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True if DEVICE.type == 'cuda' else False) # Larger batch for VAE

    #testset = datasets.FashionMNIST('./data', download=True, train=False, transform=transform)
    # For test_vae, n_display is used, so batch_size here can be larger than n_display
    #testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True if DEVICE.type == 'cuda' else False)

    celeba_dir = "./data/celeba"
    trainloader, testloader = make_celeba_loaders(
        data_dir=celeba_dir,
        batch_size=128,
        image_size=64,
        train_frac=0.8,
        num_workers=4,
        device_is_cuda=DEVICE.type == 'cuda',
    )

    part_a_vae(trainloader, testloader, latent_dim=200) # Train VAE firs

if __name__ == "__main__":
    main()