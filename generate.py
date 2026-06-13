"""
ThermalNet — Surrogate dataset generator.

Simulates 2D steady-state heat diffusion on a rectangular plate using the
analytical solution to the Laplace equation (separation of variables).

Physical setup:
  - Plate: L x W (normalized to 1x1)
  - Boundary conditions: 3 sides at T=0, top edge at T=T_top(x)
  - Heat source: Gaussian point source at (cx, cy) with intensity Q
  - Material: thermal conductivity k varies per sample (steel, aluminum, titanium range)

Each sample = (boundary conditions + material props + source params) -> temperature field
This mimics what FEA solves in hours; surrogate infers in milliseconds.
"""

import numpy as np
from scipy.special import factorial
import os

GRID = 32          # 32x32 spatial grid
N_TERMS = 40       # Fourier series terms for analytical solution
N_SAMPLES = 10000  # training + val + test


def analytical_heat(T_top, Q, cx, cy, k, grid=GRID, n_terms=N_TERMS):
    """
    Analytical solution: Laplace + Gaussian source superposition.
    T_top: peak temperature on top boundary (BC)
    Q: heat source intensity
    cx, cy: normalized source location [0,1]
    k: thermal conductivity (normalized)
    """
    x = np.linspace(0, 1, grid)
    y = np.linspace(0, 1, grid)
    X, Y = np.meshgrid(x, y)

    # Part 1: homogeneous solution (top BC sinusoidal, others zero)
    T = np.zeros((grid, grid))
    for n in range(1, n_terms + 1):
        bn = (2 * T_top / (n * np.pi)) * (1 - np.cos(n * np.pi))
        sinh_n = np.sinh(n * np.pi)
        if abs(sinh_n) < 1e-10:
            continue
        T += bn * np.sin(n * np.pi * X) * np.sinh(n * np.pi * Y) / sinh_n

    # Part 2: Gaussian heat source contribution (Green's function approximation)
    sigma = 0.08
    source = Q / k * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    # Smooth decay from source center, zero at boundaries
    boundary_decay = np.sin(np.pi * X) * np.sin(np.pi * Y)
    T += source * boundary_decay * 0.15

    # Enforce zero BCs on 3 sides
    T[0, :] = 0   # bottom
    T[:, 0] = 0   # left
    T[:, -1] = 0  # right
    T[-1, :] = T_top * np.sin(np.pi * x)  # top BC

    return T.astype(np.float32)


def generate_dataset(n=N_SAMPLES, seed=42):
    np.random.seed(seed)

    # Input features: [T_top, Q, cx, cy, k]
    T_tops = np.random.uniform(50, 500, n)      # boundary temp (C)
    Qs     = np.random.uniform(0.1, 2.0, n)     # heat source intensity
    cxs    = np.random.uniform(0.15, 0.85, n)   # source x position
    cys    = np.random.uniform(0.15, 0.85, n)   # source y position
    ks     = np.random.uniform(0.5, 2.0, n)     # thermal conductivity (normalized)
    # k=0.5 ~ titanium, k=1.0 ~ steel, k=2.0 ~ aluminum

    inputs = np.stack([T_tops, Qs, cxs, cys, ks], axis=1).astype(np.float32)
    fields = np.zeros((n, GRID, GRID), dtype=np.float32)

    print(f"Generating {n} FEA-equivalent samples...")
    for i in range(n):
        fields[i] = analytical_heat(T_tops[i], Qs[i], cxs[i], cys[i], ks[i])
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n}")

    # Normalize inputs
    input_mean = inputs.mean(axis=0)
    input_std  = inputs.std(axis=0)
    inputs_norm = (inputs - input_mean) / (input_std + 1e-8)

    # Normalize fields (per-sample min-max for physical interpretability)
    field_min = fields.reshape(n, -1).min(axis=1, keepdims=True).reshape(n, 1, 1)
    field_max = fields.reshape(n, -1).max(axis=1, keepdims=True).reshape(n, 1, 1)
    fields_norm = (fields - field_min) / (field_max - field_min + 1e-8)

    # Split 80/10/10
    idx = np.random.permutation(n)
    tr, va = int(0.8*n), int(0.9*n)
    splits = {
        "train": idx[:tr],
        "val":   idx[tr:va],
        "test":  idx[va:]
    }

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/inputs_raw.npy", inputs)
    np.save("data/processed/inputs_norm.npy", inputs_norm)
    np.save("data/processed/fields_raw.npy", fields)
    np.save("data/processed/fields_norm.npy", fields_norm)
    np.save("data/processed/input_mean.npy", input_mean)
    np.save("data/processed/input_std.npy", input_std)
    np.save("data/processed/field_min.npy", field_min)
    np.save("data/processed/field_max.npy", field_max)
    np.save("data/processed/splits.npy", splits)

    print(f"\nDataset saved to data/processed/")
    print(f"  Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    print(f"  Input shape: {inputs_norm.shape} | Field shape: {fields_norm.shape}")
    print(f"  T range: [{fields.min():.1f}, {fields.max():.1f}] C")
    return inputs_norm, fields_norm, splits


if __name__ == "__main__":
    generate_dataset()
