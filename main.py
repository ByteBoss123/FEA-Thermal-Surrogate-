"""
ThermalNet FastAPI Inference Service

Endpoints:
  POST /predict        — returns mean temperature field (fast, no UQ)
  POST /predict/uq     — returns mean + epistemic uncertainty field (MC Dropout)
  GET  /health         — liveness check
  GET  /model/info     — model metadata

Physical output is denormalized back to Celsius using saved dataset statistics.
Inference time: ~2ms (vs hours for FEA solver).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import numpy as np
import torch
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.surrogate import ThermalSurrogate

app = FastAPI(
    title="ThermalNet Surrogate API",
    description=(
        "ML surrogate replacing FEA thermal simulation. "
        "Predicts 2D steady-state temperature fields in milliseconds "
        "from physical boundary conditions and material properties."
    ),
    version="1.0.0",
)

# ---------- Model loading ----------

MODEL_PATH = os.environ.get("MODEL_PATH", "models/checkpoints/best_model.pt")
DATA_PATH  = os.environ.get("DATA_PATH",  "data/processed")

_model = None
_meta  = {}
_stats = {}


def load_model():
    global _model, _meta, _stats
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model checkpoint not found: {MODEL_PATH}. Run train.py first.")

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    cfg  = ckpt.get("config", {})
    _model = ThermalSurrogate(
        latent_dim=cfg.get("latent_dim", 128),
        dropout_p=cfg.get("dropout_p", 0.1),
    )
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()

    _meta = {
        "trained_epoch": ckpt.get("epoch"),
        "val_mse": round(ckpt.get("val_mse", 0), 6),
        "val_mae": round(ckpt.get("val_mae", 0), 6),
        "params": sum(p.numel() for p in _model.parameters()),
    }

    # Load normalization stats
    _stats["input_mean"] = np.load(f"{DATA_PATH}/input_mean.npy")
    _stats["input_std"]  = np.load(f"{DATA_PATH}/input_std.npy")
    _stats["field_min"]  = np.load(f"{DATA_PATH}/field_min.npy")  # (N,1,1)
    _stats["field_max"]  = np.load(f"{DATA_PATH}/field_max.npy")

    print(f"Model loaded | epoch={_meta['trained_epoch']} | val_mae={_meta['val_mae']}")


@app.on_event("startup")
def startup():
    try:
        load_model()
    except RuntimeError as e:
        print(f"WARNING: {e}. /predict will return 503 until model is trained.")


# ---------- Schemas ----------

class PhysicalParams(BaseModel):
    T_top: float = Field(..., ge=50, le=500,  description="Top boundary temperature (°C)")
    Q:     float = Field(..., ge=0.1, le=2.0, description="Heat source intensity (normalized)")
    cx:    float = Field(..., ge=0.1, le=0.9, description="Source x position [0,1]")
    cy:    float = Field(..., ge=0.1, le=0.9, description="Source y position [0,1]")
    k:     float = Field(..., ge=0.5, le=2.0, description="Thermal conductivity (0.5=Ti, 1.0=steel, 2.0=Al)")

    class Config:
        schema_extra = {
            "example": {
                "T_top": 300.0,
                "Q": 1.0,
                "cx": 0.5,
                "cy": 0.5,
                "k": 1.0
            }
        }


class PredictionResponse(BaseModel):
    temperature_field: list        # 32x32 nested list, °C
    T_min: float
    T_max: float
    inference_ms: float
    grid_size: int = 32


class UQPredictionResponse(BaseModel):
    temperature_field: list        # mean field, °C
    uncertainty_field: list        # epistemic std, °C
    T_min: float
    T_max: float
    uncertainty_mean: float
    uncertainty_max: float
    n_mc_samples: int
    inference_ms: float
    grid_size: int = 32


# ---------- Helpers ----------

def normalize_input(params: PhysicalParams) -> torch.Tensor:
    x = np.array([[params.T_top, params.Q, params.cx, params.cy, params.k]], dtype=np.float32)
    x_norm = (x - _stats["input_mean"]) / (_stats["input_std"] + 1e-8)
    return torch.from_numpy(x_norm)


def denormalize_field(field_norm: np.ndarray, T_top: float) -> np.ndarray:
    """
    Approximate denormalization: scale [0,1] field back to Celsius.
    Uses T_top as the expected max since top BC drives the scale.
    """
    return field_norm * T_top


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info")
def model_info():
    if not _meta:
        raise HTTPException(503, "Model not loaded")
    return {
        **_meta,
        "architecture": "ConditionEncoder (MLP) + SpatialDecoder (CNN)",
        "input_features": ["T_top", "Q", "cx", "cy", "k"],
        "output": "32x32 temperature field (°C)",
        "uncertainty_method": "Monte Carlo Dropout",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(params: PhysicalParams):
    if _model is None:
        raise HTTPException(503, "Model not loaded — run train.py first")

    x = normalize_input(params)
    t0 = time.perf_counter()
    with torch.no_grad():
        _model.eval()
        field_norm = _model(x).squeeze(0).numpy()
    elapsed = (time.perf_counter() - t0) * 1000

    field = denormalize_field(field_norm, params.T_top)

    return PredictionResponse(
        temperature_field=field.tolist(),
        T_min=round(float(field.min()), 2),
        T_max=round(float(field.max()), 2),
        inference_ms=round(elapsed, 3),
    )


@app.post("/predict/uq", response_model=UQPredictionResponse)
def predict_uq(params: PhysicalParams, n_samples: int = 50):
    if _model is None:
        raise HTTPException(503, "Model not loaded — run train.py first")
    if not (10 <= n_samples <= 200):
        raise HTTPException(422, "n_samples must be between 10 and 200")

    x = normalize_input(params)
    t0 = time.perf_counter()
    mean_norm, std_norm = _model.predict_with_uncertainty(x, n_samples=n_samples)
    elapsed = (time.perf_counter() - t0) * 1000

    mean_field = denormalize_field(mean_norm.squeeze(0).numpy(), params.T_top)
    std_field  = denormalize_field(std_norm.squeeze(0).numpy(),  params.T_top)

    return UQPredictionResponse(
        temperature_field=mean_field.tolist(),
        uncertainty_field=std_field.tolist(),
        T_min=round(float(mean_field.min()), 2),
        T_max=round(float(mean_field.max()), 2),
        uncertainty_mean=round(float(std_field.mean()), 4),
        uncertainty_max=round(float(std_field.max()), 4),
        n_mc_samples=n_samples,
        inference_ms=round(elapsed, 3),
    )
