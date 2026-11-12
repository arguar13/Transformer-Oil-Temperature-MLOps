import numpy as np
import pandas as pd
import pytest
import torch

from src.data_contracts import FEATURE_COLUMNS, DataContractError, ETTDatasetContract
from src.data_processing import DataProcessor


def test_create_sequences_generates_expected_sliding_windows(tmp_path):
    processor = DataProcessor(
        data_path=str(tmp_path / "unused.csv"),
        seq_length=3,
        output_dir=str(tmp_path),
    )
    x = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).reshape(10, 1).astype(float)

    xs, ys = processor._create_sequences(x, y)

    assert xs.shape == (7, 3, 2)  # 10 rows - seq_length(3) = 7 windows
    assert ys.shape == (7, 1)  # pred_length=1 (the default)
    np.testing.assert_array_equal(xs[0], x[0:3])
    np.testing.assert_array_equal(ys[0], y[3])
    np.testing.assert_array_equal(xs[-1], x[6:9])
    np.testing.assert_array_equal(ys[-1], y[9])


def test_create_sequences_supports_multi_step_horizon(tmp_path):
    processor = DataProcessor(
        data_path=str(tmp_path / "unused.csv"),
        seq_length=3,
        output_dir=str(tmp_path),
        pred_length=2,
    )
    x = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).reshape(10, 1).astype(float)

    xs, ys = processor._create_sequences(x, y)

    # 10 rows - seq_length(3) - pred_length(2) + 1 = 6 usable windows
    assert xs.shape == (6, 3, 2)
    assert ys.shape == (6, 2)
    np.testing.assert_array_equal(xs[0], x[0:3])
    np.testing.assert_array_equal(ys[0], y[3:5].squeeze(-1))
    np.testing.assert_array_equal(xs[-1], x[5:8])
    np.testing.assert_array_equal(ys[-1], y[8:10].squeeze(-1))


def _write_raw_csv(path, n_rows: int, extra_column: bool = False) -> None:
    dates = pd.date_range("2016-07-01", periods=n_rows, freq="h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "HUFL": rng.uniform(0, 10, n_rows),
            "HULL": rng.uniform(0, 5, n_rows),
            "MUFL": rng.uniform(0, 5, n_rows),
            "MULL": rng.uniform(0, 2, n_rows),
            "LUFL": rng.uniform(0, 5, n_rows),
            "LULL": rng.uniform(0, 2, n_rows),
            "OT": rng.uniform(10, 40, n_rows),
        }
    )
    if extra_column:
        df["unexpected_sensor"] = 1.0
    df.to_csv(path, index=False)


def test_process_and_save_fails_fast_when_a_split_is_too_short(tmp_path):
    csv_path = tmp_path / "ett.csv"
    _write_raw_csv(csv_path, n_rows=300)  # val = 30 rows < seq(24) + pred(24)
    processor = DataProcessor(
        data_path=str(csv_path),
        seq_length=24,
        output_dir=str(tmp_path / "out"),
        contract=ETTDatasetContract(min_rows=200),
        pred_length=24,
    )

    with pytest.raises(DataContractError, match="Split 'val'"):
        processor.process_and_save()


def test_process_and_save_uses_only_contract_feature_columns(tmp_path):
    csv_path = tmp_path / "ett.csv"
    _write_raw_csv(csv_path, n_rows=400, extra_column=True)
    out_dir = tmp_path / "out"
    processor = DataProcessor(
        data_path=str(csv_path),
        seq_length=8,
        output_dir=str(out_dir),
        contract=ETTDatasetContract(min_rows=200),
        pred_length=4,
    )

    processor.process_and_save()

    x_train, y_train = torch.load(out_dir / "train_tensors.pt", weights_only=True)
    assert x_train.shape[1:] == (8, len(FEATURE_COLUMNS))  # the extra column is ignored
    assert y_train.shape[1] == 4
