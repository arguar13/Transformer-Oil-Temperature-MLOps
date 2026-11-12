import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.data_contracts import (
    ETT_CONTRACT,
    FEATURE_COLUMNS,
    TOY_ETT_CONTRACT,
    DataContractError,
    ETTDatasetContract,
    validate_ett_csv,
)
from src.logging_config import configure_logging, get_logger
from src.monitoring.drift_check import build_feature_baselines

configure_logging()
logger = get_logger(__name__)

# DVC-tracked datasets (core_ml/data/*.dvc). `dvc pull` materialises them before
# a run, so every run starts from a known data hash.
TOY_DATA_PATH = "data/toy/ETTh1_toy.csv"
RAW_DATA_PATH = "data/raw/ETTh1.csv"


class DataProcessor:
    def __init__(
        self,
        data_path: str,
        seq_length: int = 48,
        output_dir: str = "artifacts",
        contract: ETTDatasetContract = ETT_CONTRACT,
        pred_length: int = 1,
    ):
        self.data_path = data_path
        self.seq_length = seq_length
        self.output_dir = output_dir
        self.contract = contract
        # Forecast horizon: how many future steps each window predicts. 1 = one
        # step ahead; >1 yields `pred_length` future values per window.
        self.pred_length = pred_length

        os.makedirs(self.output_dir, exist_ok=True)

    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        """Sliding-window sequences.

        Each input window X[i:i+seq_length] is paired with the next
        `pred_length` values y[i+seq_length : i+seq_length+pred_length]. `y`
        arrives scaled and single-column (the target), so `.squeeze(-1)` leaves
        each output window as (pred_length,), not (pred_length, 1).
        """
        Xs, ys = [], []
        last_start = len(X) - self.seq_length - self.pred_length + 1
        for i in range(last_start):
            Xs.append(X[i : i + self.seq_length])
            ys.append(y[i + self.seq_length : i + self.seq_length + self.pred_length].squeeze(-1))
        return np.array(Xs), np.array(ys)

    def process_and_save(self):
        """Validate, split, scale and window the data, and export what train.py needs."""
        # Validate the contract before spending compute on scaling or windowing.
        logger.info(f"Validating the data contract for {self.data_path}...")
        data = validate_ett_csv(self.data_path, contract=self.contract)
        logger.info("Data contract OK.")

        data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d %H:%M:%S")
        # Sliding windows assume chronological order.
        data.sort_values("date", inplace=True)
        data.set_index("date", inplace=True)
        date_index = pd.DatetimeIndex(data.index)

        data["month"] = date_index.month
        data["day"] = date_index.day
        data["hour"] = date_index.hour

        target_col = self.contract.target_column
        # The contract's explicit columns (the order the API and batch expect),
        # not `data.columns`: an extra CSV column would silently change
        # n_features and the model's input order.
        feature_cols = list(FEATURE_COLUMNS)

        # Chronological 70/10/20 split (no shuffling: that would leak the future)
        n = len(data)
        train_size = int(n * 0.7)
        val_size = int(n * 0.1)

        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size : train_size + val_size]
        test_data = data.iloc[train_size + val_size :]

        # Every split needs at least seq_length + pred_length rows for a single
        # window. Otherwise a short split yields empty arrays and fails much
        # later in train.py as an unexplained ZeroDivisionError.
        min_split_rows = self.seq_length + self.pred_length
        for split_name, split_df in (("train", train_data), ("val", val_data), ("test", test_data)):
            if len(split_df) < min_split_rows:
                raise DataContractError(
                    f"Split '{split_name}' has {len(split_df)} rows; it needs at "
                    f"least {min_split_rows} (seq_length={self.seq_length} + "
                    f"pred_length={self.pred_length}) to build one window."
                )

        # Reference profile for the drift check (monitoring/drift_check.py): the
        # per-sensor band of weekly means on train, BEFORE scaling, because the
        # inference batch is compared in raw units (°C, MW). Features only:
        # train.py adds baseline_test_mse/mae once a model exists.
        feature_baselines = build_feature_baselines(train_data)
        feature_baselines.write(os.path.join(self.output_dir, "reference_profile.json"))

        # Scalers fitted on train only
        logger.info("Scaling features and saving the scalers...")
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        train_X = scaler_X.fit_transform(train_data[feature_cols])
        val_X = scaler_X.transform(val_data[feature_cols])
        test_X = scaler_X.transform(test_data[feature_cols])

        train_y = scaler_y.fit_transform(train_data[[target_col]])
        val_y = scaler_y.transform(val_data[[target_col]])
        test_y = scaler_y.transform(test_data[[target_col]])

        # Persisted so train.py can bundle them with the model
        joblib.dump(scaler_X, os.path.join(self.output_dir, "scaler_X.pkl"))
        joblib.dump(scaler_y, os.path.join(self.output_dir, "scaler_y.pkl"))

        logger.info("Building sliding-window sequences...")
        X_train_seq, y_train_seq = self._create_sequences(train_X, train_y)
        X_val_seq, y_val_seq = self._create_sequences(val_X, val_y)
        X_test_seq, y_test_seq = self._create_sequences(test_X, test_y)

        logger.info("Writing tensors to disk...")
        torch.save(
            (
                torch.tensor(X_train_seq, dtype=torch.float32),
                torch.tensor(y_train_seq, dtype=torch.float32),
            ),
            os.path.join(self.output_dir, "train_tensors.pt"),
        )
        torch.save(
            (
                torch.tensor(X_val_seq, dtype=torch.float32),
                torch.tensor(y_val_seq, dtype=torch.float32),
            ),
            os.path.join(self.output_dir, "val_tensors.pt"),
        )
        torch.save(
            (
                torch.tensor(X_test_seq, dtype=torch.float32),
                torch.tensor(y_test_seq, dtype=torch.float32),
            ),
            os.path.join(self.output_dir, "test_tensors.pt"),
        )

        logger.info(f"Processing finished. Artifacts in {self.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETTh1 data processing")
    parser.add_argument(
        "--dataset",
        choices=["toy", "raw"],
        default="raw",
        help="'toy' uses the ~1,000-row dataset for quick end-to-end runs; 'raw' the full dataset.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Overrides the input CSV path (derived from --dataset by default).",
    )
    parser.add_argument("--seq_length", type=int, default=48)
    parser.add_argument(
        "--pred_length",
        type=int,
        default=48,
        help="Forecast horizon (future steps per window). 48 = the next 48 hours from the "
        "previous 48 (same order as seq_length, one of the DLinear paper's standard "
        "ETTh1 horizons).",
    )
    parser.add_argument("--output_dir", type=str, default="artifacts")
    args = parser.parse_args()

    if args.dataset == "toy":
        data_path = args.data_path or TOY_DATA_PATH
        contract = TOY_ETT_CONTRACT
    else:
        data_path = args.data_path or RAW_DATA_PATH
        contract = ETT_CONTRACT

    processor = DataProcessor(
        data_path,
        args.seq_length,
        args.output_dir,
        contract=contract,
        pred_length=args.pred_length,
    )
    processor.process_and_save()
