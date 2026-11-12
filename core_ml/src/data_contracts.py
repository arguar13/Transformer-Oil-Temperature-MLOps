"""Data contracts for the ETT (Electricity Transformer Temperature) pipeline.

Runs BEFORE any expensive step (preprocessing, training, batch inference). A
dataset that breaks the contract raises `DataContractError` right away instead
of letting corrupt data flow silently into the model, or into production.

Per-column ranges are calibrated on the full ETTh1.csv (17,420 hourly rows, no
nulls, Jul 2016 to Jun 2018) with margin for natural sensor variation, while
still rejecting absurd values.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field


class DataContractError(Exception):
    """The dataset breaks its data contract (raised before training or inference)."""


class ColumnContract(BaseModel):
    """Contract of one numeric ETT column."""

    name: str
    min_value: float
    max_value: float
    allow_null: bool = False


class ETTDatasetContract(BaseModel):
    """Declarative data contract for the ETT dataset."""

    required_columns: tuple[str, ...] = (
        "date",
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",
        "OT",
    )
    target_column: str = "OT"
    # An integrity floor for the raw dataset, NOT a guarantee of how many
    # windows it yields: that depends on seq_length + pred_length
    # (data_processing.py), which this contract does not know. The real dataset
    # (~17k rows) is far above it either way.
    min_rows: int = Field(default=600)
    columns: tuple[ColumnContract, ...] = (
        ColumnContract(name="HUFL", min_value=-40.0, max_value=40.0),
        ColumnContract(name="HULL", min_value=-15.0, max_value=20.0),
        ColumnContract(name="MUFL", min_value=-40.0, max_value=30.0),
        ColumnContract(name="MULL", min_value=-15.0, max_value=15.0),
        ColumnContract(name="LUFL", min_value=-10.0, max_value=15.0),
        ColumnContract(name="LULL", min_value=-5.0, max_value=8.0),
        ColumnContract(name="OT", min_value=-15.0, max_value=60.0),
    )


ETT_CONTRACT = ETTDatasetContract()

# Relaxed contract for smoke-test data (the toy dataset, ~1,000 rows).
TOY_ETT_CONTRACT = ETTDatasetContract(min_rows=200)

# Model input columns at inference: the 7 sensor readings plus the derived
# calendar features ('date' is gone by then, see data_processing.py).
FEATURE_COLUMNS: tuple[str, ...] = (
    "HUFL",
    "HULL",
    "MUFL",
    "MULL",
    "LUFL",
    "LULL",
    "OT",
    "month",
    "day",
    "hour",
)


def _check_column_ranges(df: pd.DataFrame, columns: tuple[ColumnContract, ...]) -> list[str]:
    violations: list[str] = []
    for col_contract in columns:
        col = col_contract.name
        series = df[col]

        n_nulls = int(series.isnull().sum())
        if n_nulls and not col_contract.allow_null:
            violations.append(f"Column '{col}' has {n_nulls} null values (not allowed).")

        if not pd.api.types.is_numeric_dtype(series):
            violations.append(f"Column '{col}' is not numeric (dtype={series.dtype}).")
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        out_of_range = (non_null < col_contract.min_value) | (non_null > col_contract.max_value)
        n_out = int(out_of_range.sum())
        if n_out:
            violations.append(
                f"Column '{col}': {n_out} value(s) outside the range "
                f"[{col_contract.min_value}, {col_contract.max_value}] "
                f"(observed min={non_null.min():.3f}, max={non_null.max():.3f})."
            )
    return violations


def validate_ett_dataframe(df: pd.DataFrame, contract: ETTDatasetContract = ETT_CONTRACT) -> None:
    """Validate a raw ETT dataset (with its 'date' column) against `contract`.

    Raises `DataContractError` listing ALL violations, not just the first, so
    one run is enough to fix them.
    """
    missing = [c for c in contract.required_columns if c not in df.columns]
    if missing:
        raise DataContractError(
            f"Missing columns in the dataset: {missing}. "
            f"Required: {list(contract.required_columns)}."
        )

    violations: list[str] = []

    if len(df) < contract.min_rows:
        violations.append(
            f"The dataset has {len(df)} rows; at least {contract.min_rows} are required."
        )

    try:
        # Explicit format: otherwise pandas warns "Could not infer format...
        # falling back to dateutil". ETT always uses this exact format.
        pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError) as exc:
        violations.append(f"Column 'date' has values that cannot be parsed as dates: {exc}")
    else:
        duplicated_dates = int(df["date"].duplicated().sum())
        if duplicated_dates:
            violations.append(f"There are {duplicated_dates} duplicated timestamps in 'date'.")

    violations.extend(_check_column_ranges(df, contract.columns))

    if violations:
        report = "\n  - ".join(violations)
        raise DataContractError(
            f"The dataset breaks the data contract ({len(violations)} violation(s)):\n  - {report}"
        )


def validate_feature_dataframe(
    df: pd.DataFrame, contract: ETTDatasetContract = ETT_CONTRACT
) -> None:
    """Validate a feature-engineered frame (no 'date', with month/day/hour), as
    both the online API and batch scoring feed it to the model.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError(
            f"Missing columns for inference: {missing}. Required: {list(FEATURE_COLUMNS)}."
        )

    violations = _check_column_ranges(df, contract.columns)
    if violations:
        report = "\n  - ".join(violations)
        raise DataContractError(
            f"The inference batch breaks the data contract "
            f"({len(violations)} violation(s)):\n  - {report}"
        )


def validate_ett_csv(path: str, contract: ETTDatasetContract = ETT_CONTRACT) -> pd.DataFrame:
    """Load `path` and validate it against `contract`; raises if the file is
    missing or breaks the contract.
    """
    import os

    if not os.path.exists(path):
        raise DataContractError(
            f"Data file '{path}' does not exist. Did you run `dvc pull` or `make data-toy`?"
        )
    df = pd.read_csv(path)
    validate_ett_dataframe(df, contract=contract)
    return df
