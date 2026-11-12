import pandas as pd
import pytest

from src.data_contracts import (
    DataContractError,
    ETTDatasetContract,
    validate_ett_csv,
    validate_ett_dataframe,
    validate_feature_dataframe,
)


def _make_raw_df(n_rows: int = 700) -> pd.DataFrame:
    dates = pd.date_range("2016-07-01", periods=n_rows, freq="h")
    return pd.DataFrame(
        {
            "date": dates.astype(str),
            "HUFL": 5.0,
            "HULL": 2.0,
            "MUFL": 1.5,
            "MULL": 0.5,
            "LUFL": 4.0,
            "LULL": 1.0,
            "OT": 30.0,
        }
    )


TEST_CONTRACT = ETTDatasetContract(min_rows=600)


def test_valid_dataframe_passes():
    validate_ett_dataframe(_make_raw_df(), contract=TEST_CONTRACT)  # must not raise


def test_missing_columns_fail_fast():
    df = _make_raw_df().drop(columns=["OT"])

    with pytest.raises(DataContractError, match="Missing columns"):
        validate_ett_dataframe(df, contract=TEST_CONTRACT)


def test_too_few_rows_rejected():
    df = _make_raw_df(n_rows=10)

    with pytest.raises(DataContractError, match="are required"):
        validate_ett_dataframe(df, contract=TEST_CONTRACT)


def test_out_of_range_value_rejected():
    df = _make_raw_df()
    df.loc[0, "OT"] = 9999.0

    with pytest.raises(DataContractError, match="OT"):
        validate_ett_dataframe(df, contract=TEST_CONTRACT)


def test_null_value_rejected():
    df = _make_raw_df()
    df.loc[0, "HUFL"] = None

    with pytest.raises(DataContractError, match="null values"):
        validate_ett_dataframe(df, contract=TEST_CONTRACT)


def test_duplicated_timestamps_rejected():
    df = _make_raw_df()
    df.loc[1, "date"] = df.loc[0, "date"]

    with pytest.raises(DataContractError, match="duplicated"):
        validate_ett_dataframe(df, contract=TEST_CONTRACT)


def test_unparseable_date_rejected():
    df = _make_raw_df()
    df.loc[0, "date"] = "not-a-date"

    with pytest.raises(DataContractError, match="cannot be parsed"):
        validate_ett_dataframe(df, contract=TEST_CONTRACT)


def test_reports_multiple_violations_at_once():
    df = _make_raw_df(n_rows=10)
    df.loc[0, "OT"] = 9999.0

    with pytest.raises(DataContractError) as exc_info:
        validate_ett_dataframe(df, contract=TEST_CONTRACT)

    message = str(exc_info.value)
    assert "are required" in message
    assert "OT" in message


def test_validate_ett_csv_missing_file_fails_fast(tmp_path):
    with pytest.raises(DataContractError, match="does not exist"):
        validate_ett_csv(str(tmp_path / "no_existe.csv"), contract=TEST_CONTRACT)


def test_validate_ett_csv_loads_and_validates(tmp_path):
    csv_path = tmp_path / "ett.csv"
    _make_raw_df().to_csv(csv_path, index=False)

    df = validate_ett_csv(str(csv_path), contract=TEST_CONTRACT)

    assert len(df) == 700


def test_validate_feature_dataframe_requires_no_date_column():
    df = _make_raw_df().drop(columns=["date"])
    df["month"], df["day"], df["hour"] = 7, 1, 0

    validate_feature_dataframe(df, contract=TEST_CONTRACT)  # must not raise


def test_validate_feature_dataframe_missing_time_columns_fails():
    df = _make_raw_df().drop(columns=["date"])

    with pytest.raises(DataContractError, match="month"):
        validate_feature_dataframe(df, contract=TEST_CONTRACT)
