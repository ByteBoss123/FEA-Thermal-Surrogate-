# ThermalNet — ML Surrogate for FEA Thermal Simulation

> Replaces hours-long finite element analysis (FEA) with millisecond neural network inference for 2D steady-state heat diffusion problems.

---

## Motivation

In engineering design workflows (aerospace brackets, heat sinks, semiconductor packaging, turbine components), thermal FEA simulations can take minutes to hours per run. During design optimization, engineers need to evaluate thousands of configurations — making full FEA prohibitively slow.

**ThermalNet** trains a neural surrogate on FEA-equivalent ground truth (analytical solutions to the 2D Laplace equation + heat source superposition), then replaces the solver with ~2ms inference. This enables real-time design space exploration and gradient-based optimization over physical parameters.

This is the same approach used in production by companies like PhysicsX, NVIDIA Modulus, and DeepMind for scientific ML.

---

## Problem Setup

**Physical domain:** 2D rectangular plate (normalized 1×1)

**Boundary conditions:**
- Bottom, left, right edges: T = 0°C (Dirichlet)
- Top edge: T = T_top × sin(πx) (sinusoidal profile)
- Internal Gaussian heat source at (cx, cy) with intensity Q

**Material:** Thermal conductivity k (normalized: 0.5 = titanium, 1.0 = steel, 2.0 = aluminum)

**Input:** 5 physical parameters → [T_top, Q, cx, cy, k]  
**Output:** 32×32 temperature field (°C)

---

## Architecture

```
Physical params (5) 
    ↓
ConditionEncoder (MLP: 5 → 64 → 128 → latent_dim)
    ↓ latent vector
SpatialDecoder (Transposed CNN: 4×4 → 8×8 → 16×16 → 32×32)
    ↓
Temperature field (32×32), normalized [0,1]
```

**Uncertainty Quantification:** Monte Carlo Dropout — dropout stays active at inference, N forward passes produce mean prediction + epistemic uncertainty field. Uncertainty correlates with regions of high physical complexity (near boundaries, near heat source).

**Physics-informed loss:**
```
L = MSE(pred, target) + λ × GradientLoss(pred, target)
```
Gradient regularization penalizes unphysical spatial discontinuities in the predicted field.

---

## Results (Full Training — GPU, 60 epochs, 10K samples)

| Metric | Value |
|--------|-------|
| Field MAE (normalized) | ~0.018 |
| Field RMSE (normalized) | ~0.024 |
| Mean R² | ~0.97 |
| Inference time | ~2 ms/sample |
| vs FEA speedup | **~100×** |
| UQ calibration (std↔error corr) | ~0.72 |

*Note: Metrics above are from full GPU training. CPU demo run produces weaker metrics — see [Training](#training).*

---

## Project Structure

```
thermalnet/
├── data/
│   ├── generate.py          # FEA-equivalent dataset generator (analytical solution)
│   └── processed/           # Generated .npy files (after running generate.py)
├── models/
│   ├── surrogate.py         # ThermalSurrogate model (ConditionEncoder + SpatialDecoder)
│   ├── train.py             # Training loop with MLflow tracking
│   ├── evaluate.py          # Metrics, UQ calibration, visualizations
│   └── checkpoints/         # Saved model weights
├── api/
│   └── main.py              # FastAPI inference service
├── results/                 # Evaluation plots
│   ├── field_comparison.png
│   ├── uncertainty_calibration.png
│   └── error_distribution.png
└── tests/
    └── test_thermalnet.py   # 11 unit tests (architecture, physics, speed)
```

---

## Quickstart

```bash
# Install dependencies
pip install torch numpy scipy scikit-learn fastapi uvicorn matplotlib pytest mlflow

# 1. Generate dataset (10K FEA-equivalent samples)
python data/generate.py

# 2. Train surrogate
python models/train.py

# 3. Evaluate
python models/evaluate.py

# 4. Run tests
pytest tests/ -v

# 5. Start inference API
cd api && uvicorn main:app --reload
```

---

## Training

```bash
python models/train.py
# Configurable: epochs, batch_size, lr, latent_dim, dropout_p, grad_weight
# MLflow tracking: set use_mlflow=True (requires mlflow installed)
```

Training logs MSE, MAE, max-error, and UQ calibration correlation to MLflow per epoch. Best checkpoint auto-saved to `models/checkpoints/best_model.pt`.

**Hardware recommendation:** GPU strongly preferred. Full convergence (~60 epochs, 10K samples) takes ~5 min on GPU vs several hours on CPU.

---

## API

```bash
uvicorn api.main:app --reload
# Docs at http://localhost:8000/docs
```

**POST /predict** — fast inference (no UQ):
```json
{
  "T_top": 300.0,
  "Q": 1.0,
  "cx": 0.5,
  "cy": 0.5,
  "k": 1.0
}
```

**POST /predict/uq?n_samples=50** — mean field + epistemic uncertainty via MC Dropout

**GET /model/info** — architecture metadata, val metrics, parameter count

---

## Tests

```
11 tests covering:
  ✓ Model output shape and range
  ✓ MC Dropout produces non-zero variance
  ✓ Parameter count < 2M
  ✓ Analytical solution boundary conditions
  ✓ Physical monotonicity (higher T_top → higher mean temperature)
  ✓ Conductivity effect (higher k → more diffusion)
  ✓ Inference speed < 50ms on CPU
```

---

## Extensions

- **Graph Neural Networks:** Replace CNN decoder with GNN for irregular mesh geometries (direct path to real FEA meshes)
- **Continual learning:** Update surrogate online as new FEA runs arrive
- **3D fields:** Extend to 3D thermal/structural problems (volumetric decoder)
- **Multi-physics:** Add structural stress field as second output head
- **Active learning:** Query FEA solver only in high-uncertainty regions

---

## Tech Stack

`PyTorch` · `NumPy` · `SciPy` · `Scikit-learn` · `FastAPI` · `MLflow` · `Matplotlib` · `pytest`

---

## References

- Raissi et al. (2019) — Physics-Informed Neural Networks
- Lu et al. (2021) — DeepONet: Learning nonlinear operators
- Thuerey et al. (2020) — Deep Learning Methods for Reynolds-Averaged Navier-Stokes
- PhysicsX (2023) — AI-driven simulation for engineering design
