import argparse
import copy
import os
import random

import joblib
import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.data_contracts import ETT_CONTRACT, FEATURE_COLUMNS
from src.logging_config import configure_logging, get_logger
from src.mlflow_utils import build_reproducibility_tags, log_and_register_model
from src.model_architecture import DLinear
from src.monitoring.drift_check import ReferenceProfile

configure_logging()
logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cudnn.benchmark picks kernels by timing on every run, which would defeat the
# fixed seed below.
torch.backends.cudnn.benchmark = False

# Weight init, DataLoader shuffling and the Optuna search are all random. Without
# a seed, two runs with practically the same lr ended ~26% apart in val MSE
# (registry versions 2 and 4), and the gate rejected one by chance. The seed
# does not make a candidate better; it makes the same commit and data give the
# same result every time.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEFAULT_REGISTERED_MODEL_NAME = "dlinear-ett-forecaster"
# Position of the target (OT) among the input features: the persistence
# baseline reads its last value in the window.
TARGET_FEATURE_INDEX = FEATURE_COLUMNS.index(ETT_CONTRACT.target_column)
DVC_FILE_BY_DATASET = {
    "toy": "data/toy/ETTh1_toy.csv.dvc",
    "raw": "data/raw/ETTh1.csv.dvc",
}


class ModelTrainer:
    def __init__(self, artifact_dir: str, batch_size: int = 256):
        self.artifact_dir = artifact_dir
        self.batch_size = batch_size

        logger.info(f"Loading tensors from {artifact_dir}...")
        X_train, y_train = torch.load(
            os.path.join(artifact_dir, "train_tensors.pt"), weights_only=True
        )
        X_val, y_val = torch.load(os.path.join(artifact_dir, "val_tensors.pt"), weights_only=True)
        # The real holdout: val already picks the lr and the checkpoint, so only
        # test gives an unbiased error (see evaluate_test_set).
        X_test, y_test = torch.load(
            os.path.join(artifact_dir, "test_tensors.pt"), weights_only=True
        )

        self.n_features = X_train.shape[2]
        self.seq_len = X_train.shape[1]
        # y_train arrives as (N, pred_len) from data_processing.py ((N, 1) for one step).
        self.pred_len = y_train.shape[1] if y_train.dim() > 1 else 1

        self.train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
        )

    def optimize_hyperparameters(self, n_trials=3) -> float:
        """Search the learning rate with Optuna (2-epoch proxy per trial, scored on val)."""
        logger.info("Starting the Optuna search...")

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            model = DLinear(
                seq_len=self.seq_len, n_features=self.n_features, pred_len=self.pred_len
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            for _epoch in range(2):
                model.train()
                for x_b, y_b in self.train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(x_b.to(device)), y_b.to(device))
                    loss.backward()
                    optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_b, y_b in self.val_loader:
                    loss = criterion(model(x_b.to(device)), y_b.to(device))
                    val_loss += loss.item()

            return val_loss / len(self.val_loader)

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED)
        )
        study.optimize(objective, n_trials=n_trials)
        best_lr = float(study.best_params["lr"])
        logger.info(f"Best learning rate found: {best_lr:.5f}")
        return best_lr

    def train(
        self, best_lr: float, epochs: int = 25, patience: int = 5
    ) -> tuple[str, float, nn.Module]:
        """Main training loop with early stopping on val loss.

        Logs per-epoch metrics to the active MLflow run and returns the exported
        state_dict path (a transient file; the product is the registered MLflow
        version), the best val loss, and the model restored to that checkpoint
        (so evaluate_test_set does not reload it from disk).
        """
        logger.info(f"Training the final DLinear model on {device}...")
        model = DLinear(
            seq_len=self.seq_len, n_features=self.n_features, pred_len=self.pred_len
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=best_lr)

        best_val_loss = float("inf")
        no_improve = 0
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for x_b, y_b in self.train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                preds = model(x_b)
                loss = criterion(preds, y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_mean = train_loss / len(self.train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_b, y_b in self.val_loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    preds = model(x_b)
                    loss = criterion(preds, y_b)
                    val_loss += loss.item()

            val_mean = val_loss / len(self.val_loader)
            mlflow.log_metrics({"train_mse": train_mean, "val_mse": val_mean}, step=epoch)

            if val_mean < best_val_loss:
                best_val_loss, no_improve = val_mean, 0
                # deepcopy: state_dict() returns REFERENCES to the live tensors.
                # Without a copy, best_state would follow every optimizer.step()
                # and the restored "best" checkpoint would be the last epoch.
                best_state = copy.deepcopy(model.state_dict())
                status = "[saved]"
            else:
                no_improve += 1
                status = ""

            logger.info(
                f"Epoch {epoch:2d} | Train MSE: {train_mean:.4f} | Val MSE: {val_mean:.4f} {status}"
            )

            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore the best weights and export them; the file is then bundled and
        # registered in MLflow.
        if best_state is None:
            raise RuntimeError(
                "Training never produced an epoch that improved on validation; "
                "there are no weights to save."
            )
        model.load_state_dict(best_state)
        model_export_path = os.path.join(self.artifact_dir, "dlinear_model.pth")
        torch.save(model.state_dict(), model_export_path)
        logger.info(f"Training finished. State dict exported to {model_export_path}")
        return model_export_path, best_val_loss, model

    def evaluate_test_set(
        self,
        model: nn.Module,
        scaler_x_path: str,
        scaler_y_path: str,
        target_feature_index: int = TARGET_FEATURE_INDEX,
    ) -> dict[str, float]:
        """Evaluate the final model on the test holdout, next to a naive baseline
        on exactly the same windows.

        The only evaluation that influenced nothing: train fits the weights, val
        picks the learning rate (Optuna) and the checkpoint (early stopping), and
        test is untouched until here, so these numbers are an honest error
        estimate.

        The baseline is persistence: repeat the last observed target value over
        the whole horizon. With a series as inert as oil temperature it is hard
        to beat a few hours ahead, so a low absolute error means little on its
        own; what justifies serving a model is how much it improves on
        persistence (`test_mae_skill`), and quality_gate.py requires it.

        Metrics are in the target's real unit (°C), not the scaled one.
        """
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)

        model.eval()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_last_observed: list[np.ndarray] = []
        with torch.no_grad():
            for x_b, y_b in self.test_loader:
                all_preds.append(model(x_b.to(device)).cpu().numpy())
                all_targets.append(y_b.numpy())
                # Last target value in the input window (scaled with scaler_X,
                # not scaler_y).
                all_last_observed.append(x_b[:, -1, target_feature_index].numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        last_observed = np.concatenate(all_last_observed, axis=0)

        # scaler_y was fitted on one column (the target) and expects (N, 1), so
        # each horizon step is unscaled separately and reassembled into (N, pred_len).
        original_shape = preds.shape
        preds_original = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(original_shape)
        targets_original = scaler_y.inverse_transform(targets.reshape(-1, 1)).reshape(
            original_shape
        )

        # Persistence: the last value, unscaled with its own column's scaler_X
        # parameters, repeated over the pred_len steps.
        last_observed_original = (
            last_observed * scaler_x.scale_[target_feature_index]
            + scaler_x.mean_[target_feature_index]
        )
        persistence_original = np.repeat(
            last_observed_original[:, np.newaxis], original_shape[1], axis=1
        )

        model_metrics = _forecast_metrics(preds_original, targets_original)
        persistence_metrics = _forecast_metrics(persistence_original, targets_original)
        return {
            "test_mse": model_metrics["mse"],
            "test_mae": model_metrics["mae"],
            "test_rmse": model_metrics["rmse"],
            "test_wape": model_metrics["wape"],
            "test_mse_persistence": persistence_metrics["mse"],
            "test_mae_persistence": persistence_metrics["mae"],
            # Share of the persistence error the model removes:
            # 0 = no better than repeating the last value, >0 better, <0 worse.
            "test_mae_skill": 1.0 - model_metrics["mae"] / persistence_metrics["mae"],
        }


def _forecast_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """MSE, MAE, RMSE and WAPE in the target's units."""
    errors = predicted - actual
    mse = float(np.mean(errors**2))
    # WAPE (sum|error| / sum|actual|), not MAPE: the oil temperature in ETTh1's
    # test split crosses zero, and MAPE divides point by point, so a few readings
    # near 0 °C would dominate it. The epsilon only covers a holdout whose
    # targets sum to exactly zero.
    epsilon = 1e-8
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(mse)),
        "wape": float(np.sum(np.abs(errors)) / max(np.sum(np.abs(actual)), epsilon) * 100),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLinear model training")
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    # 5 trials, not 3: each trial is a 2-epoch proxy, while the final training
    # runs up to `epochs` with early stopping, so an lr that looks fine after two
    # epochs may not converge in 25. With 3 trials the gate rejected a real
    # candidate whose lr was too low for that budget. The seed makes the search
    # reproducible, not better; more trials lower the odds of a bad lr.
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument(
        "--dataset",
        choices=["toy", "raw"],
        default="raw",
        help="Must match the dataset used in data_processing.py, so the recorded "
        "DVC data hash is the right one.",
    )
    parser.add_argument("--registered_model_name", type=str, default=DEFAULT_REGISTERED_MODEL_NAME)
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "dlinear-ett-forecasting"))

    trainer = ModelTrainer(args.artifact_dir)

    with mlflow.start_run() as run:
        # Reproducibility tuple: Git commit + DVC data hash + hyperparameters +
        # MLflow run ID (this run) + container image tag.
        repro_tags = build_reproducibility_tags(DVC_FILE_BY_DATASET[args.dataset])
        mlflow.set_tags(repro_tags)
        mlflow.log_params(
            {
                "seq_len": trainer.seq_len,
                "pred_len": trainer.pred_len,
                "n_features": trainer.n_features,
                "batch_size": trainer.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "n_trials": args.n_trials,
                "dataset": args.dataset,
            }
        )

        best_lr = trainer.optimize_hyperparameters(n_trials=args.n_trials)
        mlflow.log_param("best_lr", best_lr)

        model_export_path, best_val_loss, best_model = trainer.train(
            best_lr=best_lr, epochs=args.epochs, patience=args.patience
        )
        mlflow.log_metric("final_val_mse", best_val_loss)

        test_metrics = trainer.evaluate_test_set(
            best_model,
            scaler_x_path=os.path.join(args.artifact_dir, "scaler_X.pkl"),
            scaler_y_path=os.path.join(args.artifact_dir, "scaler_y.pkl"),
        )
        # final_test_mse_persistence is the bar quality_gate.py requires every
        # candidate to beat.
        mlflow.log_metrics({f"final_{key}": value for key, value in test_metrics.items()})
        logger.info("test_set_evaluated", **test_metrics)

        # Complete the reference profile data_processing.py started (raw
        # per-sensor stats and weekly-mean band, no performance figures yet - it trained no
        # model) with the held-out metrics just computed, and log it as an
        # artifact of THIS run. batch_inference.py downloads it via the
        # served model's run_id and uses it for the post-inference drift
        # check (core_ml/src/monitoring/drift_check.py).
        profile_path = os.path.join(args.artifact_dir, "reference_profile.json")
        reference_profile = ReferenceProfile.read(profile_path)
        reference_profile.baseline_test_mse = test_metrics["test_mse"]
        reference_profile.baseline_test_mae = test_metrics["test_mae"]
        reference_profile.write(profile_path)
        mlflow.log_artifact(profile_path, artifact_path="monitoring")

        model_version = log_and_register_model(
            model_state_dict_path=model_export_path,
            scaler_x_path=os.path.join(args.artifact_dir, "scaler_X.pkl"),
            scaler_y_path=os.path.join(args.artifact_dir, "scaler_y.pkl"),
            seq_len=trainer.seq_len,
            n_features=trainer.n_features,
            pred_len=trainer.pred_len,
            registered_model_name=args.registered_model_name,
        )

        logger.info(
            "training_run_completed",
            mlflow_run_id=run.info.run_id,
            model_name=args.registered_model_name,
            model_version=model_version.version,
            git_commit_hash=repro_tags["git_commit_hash"],
            dvc_data_hash=repro_tags["dvc_data_hash"],
            container_image_tag=repro_tags["container_image_tag"],
        )
        # quality_gate.py evaluates exactly this run (see RUN_ID_FILE there).
        with open("run_id.txt", "w", encoding="utf-8") as handle:
            handle.write(run.info.run_id)
