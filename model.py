"""
VAE Autoencoder
"""
import torch
import torch.nn as nn
import numpy as np

from typing import Tuple

class VariationalAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        """
        Initialize the Variational Autoencoder (CNN version)

        Args:
        - in_channels: Number of input channels (1 for grayscale Fashion MNIST)
        - latent_dim: size of the latent representation
        """
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels # 3 for RGB values 

         # === ENCODER ===
        # Input: (B, 3, 64, 64)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels,  32, 3, 2, 1),  # ↓ 64→32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 3, 2, 1),  # ↓ 32→16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, 1, 1),  # ↔ 16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 1, 1),  # ↔ 16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten()                            # → (B, 256*16*16)
        )

        self.flattened_size = 256 * 16 * 16
        self.fc_mu     = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # === DECODER ===
        self.decoder_fc = nn.Linear(latent_dim, self.flattened_size)

        self.decoder_conv = nn.Sequential(
            # start from (B,256,16,16)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),  # ↔ 16
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128,  64, kernel_size=3, stride=2, padding=1, output_padding=1),  # ↑ 16→32
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64,   32, kernel_size=3, stride=2, padding=1, output_padding=1),  # ↑ 32→64
            nn.LeakyReLU(0.2, inplace=True),

            # Final conv to get back to RGB
            nn.ConvTranspose2d(32, self.in_channels, kernel_size=3, stride=1, padding=1),     # ↔ 64
            nn.Tanh()  # outputs in [-1,1], to match your normalized inputs
        )

        self.apply(self._init_weights)

    def _init_weights(self, layer: nn.Module) -> None:
        """
        Initialize the weights of the layer

        Args:
        - layer: layer to initialize
        """
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.fill_(0.01)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input image into mu and logvar for the latent distribution.

        Args:
        - x: input image (B, C, H, W) e.g., (B, 1, 28, 28)

        Returns:
        - mu: mean of the latent Gaussian
        - logvar: log variance of the latent Gaussian
        """
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var).

        Args:
        - mu: mean of the latent Gaussian
        - logvar: log variance of the latent Gaussian

        Returns:
        - z: sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Sample from N(0, I)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent representation z into the reconstructed image.

        Args:
        - z: latent representation (B, latent_dim)

        Returns:
        - reconstructed image (B, C, H, W) e.g., (B, 3, 64, 64)
        """
        h = self.decoder_fc(z)
        h = h.view(-1, 256, 16, 16)
        recon_x = self.decoder_conv(h)
        return recon_x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE.

        Args:
        - x: input image (B, C, H, W)

        Returns:
        - recon_x: reconstructed image
        - mu: mean of the latent Gaussian
        - logvar: log variance of the latent Gaussian
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar