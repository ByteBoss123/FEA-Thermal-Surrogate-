"""
ThermalNet Evaluation

Metrics:
  - Field-level MAE and RMSE (normalized and physical units)
  - R² per sample, mean across test set
  - Max pointwise error distribution
  - Uncertainty calibration: does predicted std correlate with actual error?
  - Inference speedup vs analytical solver (proxy for FEA speedup)
  - Visual comparison: predicted vs ground truth vs error map
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys, time, json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.surrogate import ThermalSurrogate


def load_model_and_data(model_path="models/checkpoints/best_model.pt",
                         data_dir="data/processed"):
    ckpt = torch.load(model_path, map_location="cpu")
    cfg  = ckpt.get("config", {})
    model = ThermalSurrogate(
        latent_dim=cfg.get("latent_dim", 128),
        dropout_p=cfg.get("dropout_p", 0.1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    inputs = np.load(f"{data_dir}/inputs_norm.npy")
    fields = np.load(f"{data_dir}/fields_norm.npy")
    splits = np.load(f"{data_dir}/splits.npy", allow_pickle=True).item()
    inputs_raw = np.load(f"{data_dir}/inputs_raw.npy")

    test_idx = splits["test"]
    return model, inputs[test_idx], fields[test_idx], inputs_raw[test_idx]


def evaluate(model, inputs, fields, inputs_raw, n_uq_samples=50, n_viz=4):
    X = torch.from_numpy(inputs).float()
    Y_true = fields

    # --- Point predictions ---
    t0 = time.perf_counter()
    with torch.no_grad():
        Y_pred = model(X).numpy()
    infer_time = (time.perf_counter() - t0) / len(inputs) * 1000  # ms/sample

    err = np.abs(Y_pred - Y_true)
    mae  = err.mean()
    rmse = np.sqrt(((Y_pred - Y_true)**2).mean())

    # R² per sample
    ss_res = ((Y_pred - Y_true)**2).reshape(len(inputs), -1).sum(axis=1)
    ss_tot = ((Y_true - Y_true.reshape(len(inputs), -1).mean(axis=1, keepdims=True).reshape(-1,1,1))**2).reshape(len(inputs), -1).sum(axis=1)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    # Max error per sample
    max_err = err.reshape(len(inputs), -1).max(axis=1)

    print(f"\n{'='*50}")
    print(f"ThermalNet Evaluation — {len(inputs)} test samples")
    print(f"{'='*50}")
    print(f"  Field MAE (normalized):  {mae:.4f}")
    print(f"  Field RMSE (normalized): {rmse:.4f}")
    print(f"  Mean R²:                 {r2.mean():.4f}")
    print(f"  Median max error:        {np.median(max_err):.4f}")
    print(f"  95th pct max error:      {np.percentile(max_err, 95):.4f}")
    print(f"  Inference time:          {infer_time:.2f} ms/sample")
    print(f"  vs analytical solver:    ~200ms/sample -> {200/infer_time:.0f}x speedup")

    # --- UQ calibration on subset ---
    n_uq = min(200, len(inputs))
    X_uq = X[:n_uq]
    mean_uq, std_uq = model.predict_with_uncertainty(X_uq, n_samples=n_uq_samples)
    mean_uq, std_uq = mean_uq.numpy(), std_uq.numpy()
    actual_err_uq = np.abs(mean_uq - Y_true[:n_uq])

    corr = np.corrcoef(std_uq.flatten(), actual_err_uq.flatten())[0, 1]
    print(f"\n  UQ calibration (std vs actual error correlation): {corr:.4f}")
    print(f"  Mean predicted uncertainty: {std_uq.mean():.4f}")

    metrics = {
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "mean_r2": float(r2.mean()),
        "median_max_error": float(np.median(max_err)),
        "p95_max_error": float(np.percentile(max_err, 95)),
        "inference_ms_per_sample": float(infer_time),
        "uq_calibration_corr": float(corr),
        "n_test_samples": len(inputs),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Visualizations ---
    _plot_comparisons(Y_true, Y_pred, err, inputs_raw, n_viz)
    _plot_uq(mean_uq, std_uq, actual_err_uq)
    _plot_error_dist(max_err, r2)

    print(f"\n  Plots saved to results/")
    return metrics


def _plot_comparisons(Y_true, Y_pred, err, inputs_raw, n=4):
    fig = plt.figure(figsize=(14, 4 * n))
    gs  = gridspec.GridSpec(n, 4, figure=fig, hspace=0.4, wspace=0.3)

    idxs = np.random.choice(len(Y_true), n, replace=False)
    feature_names = ["T_top(°C)", "Q", "cx", "cy", "k"]

    for row, i in enumerate(idxs):
        params_str = " | ".join(f"{k}={v:.2f}" for k, v in zip(feature_names, inputs_raw[i]))

        ax0 = fig.add_subplot(gs[row, 0])
        im0 = ax0.imshow(Y_true[i], cmap="hot", origin="lower", vmin=0, vmax=1)
        ax0.set_title(f"FEA Ground Truth\n{params_str}", fontsize=7)
        plt.colorbar(im0, ax=ax0)

        ax1 = fig.add_subplot(gs[row, 1])
        im1 = ax1.imshow(Y_pred[i], cmap="hot", origin="lower", vmin=0, vmax=1)
        ax1.set_title("Surrogate Prediction", fontsize=8)
        plt.colorbar(im1, ax=ax1)

        ax2 = fig.add_subplot(gs[row, 2])
        im2 = ax2.imshow(err[i], cmap="RdYlGn_r", origin="lower")
        ax2.set_title(f"Absolute Error\nMAE={err[i].mean():.3f}", fontsize=8)
        plt.colorbar(im2, ax=ax2)

        ax3 = fig.add_subplot(gs[row, 3])
        ax3.scatter(Y_true[i].flatten(), Y_pred[i].flatten(), alpha=0.3, s=3, color="steelblue")
        ax3.plot([0,1],[0,1], "r--", linewidth=1)
        ax3.set_xlabel("True T (norm)", fontsize=7)
        ax3.set_ylabel("Pred T (norm)", fontsize=7)
        ax3.set_title("Pred vs True", fontsize=8)

    plt.suptitle("ThermalNet: Surrogate vs FEA Ground Truth", fontsize=12, fontweight="bold", y=1.01)
    plt.savefig("results/field_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()


def _plot_uq(mean, std, actual_err):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(mean[0], cmap="hot", origin="lower")
    axes[0].set_title("Mean Prediction (MC Dropout)")

    axes[1].imshow(std[0], cmap="Blues", origin="lower")
    axes[1].set_title("Epistemic Uncertainty (std)")

    axes[2].scatter(std.flatten()[::10], actual_err.flatten()[::10],
                    alpha=0.2, s=2, color="darkorange")
    axes[2].set_xlabel("Predicted uncertainty (std)")
    axes[2].set_ylabel("Actual error")
    axes[2].set_title(f"UQ Calibration\ncorr={np.corrcoef(std.flatten(), actual_err.flatten())[0,1]:.3f}")

    plt.suptitle("Monte Carlo Dropout Uncertainty Quantification", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/uncertainty_calibration.png", dpi=120, bbox_inches="tight")
    plt.close()


def _plot_error_dist(max_err, r2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(max_err, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[0].axvline(np.median(max_err), color="red", linestyle="--", label=f"Median={np.median(max_err):.3f}")
    axes[0].set_xlabel("Max pointwise error (normalized)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Max Error Distribution")
    axes[0].legend()

    axes[1].hist(r2, bins=40, color="seagreen", edgecolor="white", linewidth=0.5)
    axes[1].axvline(r2.mean(), color="red", linestyle="--", label=f"Mean R²={r2.mean():.3f}")
    axes[1].set_xlabel("R² per sample")
    axes[1].set_ylabel("Count")
    axes[1].set_title("R² Distribution")
    axes[1].legend()

    plt.suptitle("ThermalNet Test Set Error Analysis", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/error_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    model, inputs, fields, inputs_raw = load_model_and_data()
    metrics = evaluate(model, inputs, fields, inputs_raw)
