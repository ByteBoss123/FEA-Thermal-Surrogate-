# ThermalNet: Neural Surrogate for 2D Steady-State Heat Fields

A network that maps five physical parameters to a 32×32 temperature field, with Monte Carlo dropout uncertainty
and a FastAPI endpoint. Trained and scored on CPU, October 2026.

**Problem.** A square plate held at 0 °C on three edges, with a sinusoidal temperature profile on the top edge
and a Gaussian heat source inside. Inputs: top-edge temperature (50–500 °C), source strength, source position
(x, y) and conductivity. Output: the temperature at 32×32 points.

**Reference data, stated plainly.** Targets come from `data/generate.py`: a 40-term Fourier series that solves
Laplace's equation for the boundary conditions exactly, plus a hand-shaped Gaussian bump for the heat source.
That source term is an approximation, not a solution of the heat equation, and no finite-element solver is
involved. This is a surrogate for that reference function, which makes it a clean testbed for the method rather
than a stand-in for FEA.

## Results (`results/metrics.json`, 1,000 held-out cases, 40 epochs)

| Metric | Result |
|---|---|
| Mean absolute error | **2.3 °C** (fields range 0–499 °C); R² 0.991 per field on average |
| Worst point in a field | 13.4 °C median, 21.5 °C at the 95th percentile, mostly near the heat source |
| Uncertainty quality | Correlation between MC-dropout std and actual error: **0.32** (weak but positive) |
| Speed | Surrogate 1.3 ms per case in a batch, 1.9 ms one at a time; the reference solver takes **1.35 ms**, so there is **no speedup on this problem** |

The speed result is the important one. The surrogate only pays off when the reference solver is expensive, as a
real FEA mesh solve would be. Here the reference is a cheap closed-form series, so the network is about as fast
as what it imitates.

## What changed in the October 2026 rebuild

- **Removed claims that were never measured.** The previous README reported R² ≈ 0.97, MAE ≈ 0.018 and a ~100×
  speedup "from full GPU training". The metrics file committed alongside it showed R² −0.83, and the speedup was a
  hard-coded 200 ms solver time, not a measurement.
- **Fixed a normalization flaw.** Each field was min-max scaled by its own minimum and maximum, which the model
  cannot know when predicting, so its output could not be turned back into °C (the API approximated it with the
  top-edge temperature). Fields are now scaled by one constant from the training set, and errors are reported in °C.
- **Measured the reference solver** on the same cases instead of assuming its runtime.
- Restored the `data/`, `models/`, `tests/` layout that the imports expected; 11 tests pass.

## Architecture

Five parameters → MLP encoder (5 → 64 → 128 → latent) → transposed-CNN decoder (4×4 → 32×32) → sigmoid.
Loss is MSE plus a gradient-matching term that penalizes jagged fields. Dropout stays on at inference; 50 passes
give a mean field and a per-point standard deviation.

## Run

```
pip install numpy scipy torch matplotlib fastapi uvicorn pytest
python data/generate.py              # 10,000 cases, 80/10/10 split, ~10 s
EPOCHS=40 python models/train.py      # ~40 min on 2 CPU cores
python models/evaluate.py             # writes results/
python -m pytest -q tests
uvicorn main:app                      # POST /predict
```
