"""
ThermalNet Training Script

Loss: MSE on normalized field + gradient regularization (penalizes unphysical
      discontinuities in the predicted temperature field)
Optimizer: AdamW with cosine annealing LR schedule
Tracking: MLflow — logs loss curves, field MAE, max-error, uncertainty calibration
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.surrogate import ThermalSurrogate


class ThermalDataset(Dataset):
    def __init__(self, inputs, fields):
        self.inputs = torch.from_numpy(inputs).float()
        self.fields = torch.from_numpy(fields).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.fields[i]


def gradient_loss(pred, target):
    """
    Physics-informed regularization: penalize unphysical discontinuities.
    Computes spatial gradient difference between prediction and target field.
    """
    def grad(f):
        dx = f[:, :, 1:] - f[:, :, :-1]
        dy = f[:, 1:, :] - f[:, :-1, :]
        return dx, dy

    pred_dx, pred_dy   = grad(pred)
    target_dx, target_dy = grad(target)
    return (nn.functional.mse_loss(pred_dx, target_dx) +
            nn.functional.mse_loss(pred_dy, target_dy))


def train(
    data_dir="data/processed",
    epochs=60,
    batch_size=128,
    lr=3e-4,
    latent_dim=128,
    dropout_p=0.1,
    grad_weight=0.1,
    device=None,
    use_mlflow=True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load data
    inputs  = np.load(f"{data_dir}/inputs_norm.npy")
    fields  = np.load(f"{data_dir}/fields_norm.npy")
    splits  = np.load(f"{data_dir}/splits.npy", allow_pickle=True).item()

    tr_ds = ThermalDataset(inputs[splits["train"]], fields[splits["train"]])
    va_ds = ThermalDataset(inputs[splits["val"]],   fields[splits["val"]])

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ThermalSurrogate(latent_dim=latent_dim, dropout_p=dropout_p).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # MLflow setup
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("ThermalNet")
            run = mlflow.start_run(run_name="surrogate_v1")
            mlflow.log_params({
                "epochs": epochs, "batch_size": batch_size,
                "lr": lr, "latent_dim": latent_dim,
                "dropout_p": dropout_p, "grad_weight": grad_weight,
                "device": device
            })
        except ImportError:
            print("MLflow not installed, skipping tracking.")
            use_mlflow = False

    best_val = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb) + grad_weight * gradient_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr_ds)

        # Validate
        model.eval()
        va_loss, va_mae, va_max = 0, 0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                va_loss += nn.functional.mse_loss(pred, yb).item() * len(xb)
                va_mae  += (pred - yb).abs().mean().item() * len(xb)
                va_max  += (pred - yb).abs().max().item() * len(xb)

        va_loss /= len(va_ds)
        va_mae  /= len(va_ds)
        va_max  /= len(va_ds)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": tr_loss,
                         "val_loss": va_loss, "val_mae": va_mae})

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train MSE: {tr_loss:.4f} | "
                  f"Val MSE: {va_loss:.4f} | "
                  f"Val MAE: {va_mae:.4f} | "
                  f"Val MaxErr: {va_max:.4f}")

        if use_mlflow:
            import mlflow
            mlflow.log_metrics({
                "train_mse": tr_loss, "val_mse": va_loss,
                "val_mae": va_mae, "val_max_error": va_max
            }, step=epoch)

        # Save best
        if va_loss < best_val:
            best_val = va_loss
            os.makedirs("models/checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_mse": va_loss,
                "val_mae": va_mae,
                "config": {
                    "latent_dim": latent_dim,
                    "dropout_p": dropout_p,
                }
            }, "models/checkpoints/best_model.pt")

    # Save history
    with open("models/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    if use_mlflow:
        import mlflow
        mlflow.log_artifact("models/checkpoints/best_model.pt")
        mlflow.end_run()

    print(f"\nTraining complete. Best val MSE: {best_val:.4f}")
    return model, history


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    model, history = train(epochs=60, use_mlflow=False)
