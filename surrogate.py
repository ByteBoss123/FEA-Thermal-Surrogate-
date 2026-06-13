"""
ThermalNet Surrogate Model

Architecture: Hybrid encoder-decoder
  - Condition encoder: MLP encodes 5 physical parameters -> latent vector
  - Spatial decoder: transposed conv upsamples latent -> 32x32 temperature field
  - Uncertainty: Monte Carlo Dropout (enabled at inference for UQ)

Design rationale:
  - Physical parameters are global (scalar) -> MLP encoder
  - Output is a spatial field -> CNN decoder (parameter efficient vs full MLP)
  - MC Dropout gives calibrated uncertainty without ensemble overhead
  - Architecture inspired by conditional neural fields used in surrogate modeling
"""

import torch
import torch.nn as nn
import numpy as np


class ConditionEncoder(nn.Module):
    """Encodes 5 physical parameters into a latent condition vector."""

    def __init__(self, input_dim=5, latent_dim=128, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SpatialDecoder(nn.Module):
    """
    Decodes latent vector -> 32x32 temperature field.
    4x4 base -> 8x8 -> 16x16 -> 32x32 via transposed convolutions.
    """

    def __init__(self, latent_dim=128, dropout_p=0.1):
        super().__init__()
        # Project latent to 4x4 spatial seed
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(dropout_p),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(dropout_p),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # Output head: single channel temperature field
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),   # output in [0,1] (normalized field)
        )

    def forward(self, z):
        x = self.project(z).view(-1, 256, 4, 4)
        return self.decoder(x).squeeze(1)   # (B, 32, 32)


class ThermalSurrogate(nn.Module):
    """
    Full surrogate model: physical params -> predicted temperature field.
    Dropout stays active at inference for MC uncertainty estimation.
    """

    def __init__(self, input_dim=5, latent_dim=128, dropout_p=0.1):
        super().__init__()
        self.encoder = ConditionEncoder(input_dim, latent_dim, dropout_p)
        self.decoder = SpatialDecoder(latent_dim, dropout_p)
        self.dropout_p = dropout_p

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def predict_with_uncertainty(self, x, n_samples=50):
        """
        MC Dropout inference: run n_samples forward passes with dropout active.
        Returns mean prediction and epistemic uncertainty (std across samples).
        """
        self.train()   # keep dropout active
        with torch.no_grad():
            preds = torch.stack([self(x) for _ in range(n_samples)], dim=0)
        self.eval()

        mean = preds.mean(dim=0)
        std  = preds.std(dim=0)
        return mean, std


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ThermalSurrogate()
    x = torch.randn(4, 5)
    out = model(x)
    print(f"Output shape: {out.shape}")   # (4, 32, 32)
    print(f"Parameters: {count_params(model):,}")

    mean, std = model.predict_with_uncertainty(x, n_samples=20)
    print(f"MC mean shape: {mean.shape} | std shape: {std.shape}")
    print(f"Mean uncertainty: {std.mean().item():.4f}")
