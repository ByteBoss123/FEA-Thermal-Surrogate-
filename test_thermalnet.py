"""
ThermalNet Tests
"""
import numpy as np
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.surrogate import ThermalSurrogate, ConditionEncoder, SpatialDecoder
from data.generate import analytical_heat


# ── Model architecture tests ──────────────────────────────────────────────────

def test_model_output_shape():
    model = ThermalSurrogate()
    x = torch.randn(8, 5)
    out = model(x)
    assert out.shape == (8, 32, 32), f"Expected (8,32,32), got {out.shape}"


def test_model_output_range():
    """Output should be in [0,1] due to Sigmoid."""
    model = ThermalSurrogate()
    model.eval()
    x = torch.randn(16, 5)
    with torch.no_grad():
        out = model(x)
    assert out.min() >= 0.0 - 1e-5
    assert out.max() <= 1.0 + 1e-5


def test_mc_dropout_variance():
    """MC samples should produce non-zero variance."""
    model = ThermalSurrogate(dropout_p=0.2)
    x = torch.randn(2, 5)
    mean, std = model.predict_with_uncertainty(x, n_samples=20)
    assert mean.shape == (2, 32, 32)
    assert std.shape  == (2, 32, 32)
    assert std.mean().item() > 0, "MC Dropout should produce non-zero std"


def test_encoder_output_shape():
    enc = ConditionEncoder(input_dim=5, latent_dim=128)
    x = torch.randn(4, 5)
    z = enc(x)
    assert z.shape == (4, 128)


def test_decoder_output_shape():
    dec = SpatialDecoder(latent_dim=128)
    z = torch.randn(4, 128)
    out = dec(z)
    assert out.shape == (4, 32, 32)


def test_param_count():
    """Model should be under 2M params (lightweight for surrogate use)."""
    model = ThermalSurrogate()
    n = sum(p.numel() for p in model.parameters())
    assert n < 2_000_000, f"Too many params: {n:,}"


# ── Data generation tests ─────────────────────────────────────────────────────

def test_analytical_heat_shape():
    field = analytical_heat(T_top=300, Q=1.0, cx=0.5, cy=0.5, k=1.0)
    assert field.shape == (32, 32)


def test_analytical_heat_boundary_conditions():
    """3 sides should be ~0, top boundary should follow sin profile."""
    field = analytical_heat(T_top=300, Q=1.0, cx=0.5, cy=0.5, k=1.0)
    assert abs(field[0, 16]) < 5.0,  "Bottom BC should be ~0"
    assert abs(field[16, 0]) < 5.0,  "Left BC should be ~0"
    assert abs(field[16, -1]) < 5.0, "Right BC should be ~0"
    assert field[-1, 16] > 0,        "Top BC should be positive"


def test_analytical_heat_monotonic_in_T():
    """Higher T_top should produce higher mean temperature."""
    f1 = analytical_heat(T_top=100, Q=1.0, cx=0.5, cy=0.5, k=1.0)
    f2 = analytical_heat(T_top=400, Q=1.0, cx=0.5, cy=0.5, k=1.0)
    assert f2.mean() > f1.mean()


def test_analytical_heat_conductivity():
    """Higher k should reduce source-driven hotspot intensity."""
    f_low_k  = analytical_heat(T_top=200, Q=1.5, cx=0.5, cy=0.5, k=0.5)
    f_high_k = analytical_heat(T_top=200, Q=1.5, cx=0.5, cy=0.5, k=2.0)
    # Higher conductivity diffuses heat more — lower local max near source
    assert f_low_k.max() >= f_high_k.max() - 0.1


# ── Inference speed test ──────────────────────────────────────────────────────

def test_inference_speed():
    """Single inference should complete under 50ms on CPU."""
    import time
    model = ThermalSurrogate()
    model.eval()
    x = torch.randn(1, 5)
    # Warmup
    with torch.no_grad():
        model(x)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            model(x)
    elapsed_ms = (time.perf_counter() - t0) / 10 * 1000
    assert elapsed_ms < 50, f"Inference too slow: {elapsed_ms:.1f}ms"
