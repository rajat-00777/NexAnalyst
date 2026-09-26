"""
Phase 2 tests — pure pandas logic, no LLM/API key required.
Run: pytest tests/test_cleaning_agent.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from tools.cleaning_tools import (
    remove_duplicate_rows,
    standardize_text_columns,
    handle_missing_values,
    fix_dtypes,
    flag_outliers,
    clean_dataset,
)


def test_remove_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    cleaned, log = remove_duplicate_rows(df)
    assert len(cleaned) == 2
    assert len(log) == 1
    assert log[0]["operation"] == "remove_duplicates"


def test_remove_duplicate_rows_no_dupes():
    df = pd.DataFrame({"a": [1, 2, 3]})
    cleaned, log = remove_duplicate_rows(df)
    assert len(cleaned) == 3
    assert len(log) == 0


def test_standardize_text_strips_whitespace():
    df = pd.DataFrame({"city": [" New York", "Chicago ", "Boston"]})
    cleaned, log = standardize_text_columns(df)
    assert cleaned["city"].tolist() == ["New York", "Chicago", "Boston"]
    assert any(l["operation"] == "strip_whitespace" for l in log)


def test_standardize_text_merges_case_variants():
    df = pd.DataFrame({"status": ["Active", "active", "Active", "ACTIVE", "Inactive"]})
    cleaned, log = standardize_text_columns(df)
    assert cleaned["status"].nunique() == 2  # Active, Inactive
    assert any(l["operation"] == "normalize_text_case" for l in log)


def test_standardize_text_skips_high_cardinality():
    # Each name is unique -> high cardinality -> should NOT case-merge
    df = pd.DataFrame({"name": ["Alice", "alice", "Bob", "Charlie", "Dave"]})
    cleaned, log = standardize_text_columns(df)
    # "Alice"/"alice" stay distinct since column is treated as identifier-like
    assert cleaned["name"].nunique() == 5


def test_handle_missing_values_numeric_median():
    df = pd.DataFrame({"val": [10.0, None, 30.0, 40.0]})
    cleaned, log = handle_missing_values(df)
    assert cleaned["val"].isnull().sum() == 0
    assert log[0]["operation"] == "fill_missing_numeric"
    assert log[0]["rows_affected"] == 1


def test_handle_missing_values_categorical_mode_low_cardinality():
    df = pd.DataFrame({"cat": ["a", "a", None, "b"]})
    cleaned, log = handle_missing_values(df)
    assert cleaned["cat"].isnull().sum() == 0
    assert cleaned["cat"].iloc[2] == "a"  # mode
    assert log[0]["operation"] == "fill_missing_categorical"
    assert log[0]["strategy"] == "mode"


def test_handle_missing_values_skips_mode_fill_high_cardinality():
    # 5 unique names out of 6 rows -> high cardinality -> "Unknown" not mode
    df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie", "Dave", "Eve", None]})
    cleaned, log = handle_missing_values(df)
    assert cleaned["name"].iloc[5] == "Unknown"
    assert log[0]["strategy"] == "unknown_placeholder"


def test_handle_missing_values_skips_high_missing_column():
    df = pd.DataFrame({"val": [1.0, None, None, None]})  # 75% missing
    cleaned, log = handle_missing_values(df)
    assert cleaned["val"].isnull().sum() == 3  # left as-is
    assert log[0]["operation"] == "high_missing_warning"


def test_handle_missing_values_no_missing():
    df = pd.DataFrame({"a": [1, 2, 3]})
    cleaned, log = handle_missing_values(df)
    assert len(log) == 0


def test_fix_dtypes_converts_date_column():
    df = pd.DataFrame({"order_date": ["2024-01-01", "2024-01-02", "2024-01-03"]})
    cleaned, log = fix_dtypes(df, datetime_like_cols=["order_date"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["order_date"])
    assert len(log) == 1


def test_fix_dtypes_skips_non_date_text():
    df = pd.DataFrame({"order_date": ["apple", "banana", "cherry"]})
    cleaned, log = fix_dtypes(df, datetime_like_cols=["order_date"])
    assert not pd.api.types.is_datetime64_any_dtype(cleaned["order_date"])
    assert len(log) == 0


def test_fix_dtypes_creates_nat_for_bad_value():
    # 10 rows, 1 bad value = 10% failure rate, under the 20% threshold -> converts
    dates = ["2024-01-0" + str(i) for i in range(1, 10)] + ["invalid_date"]
    df = pd.DataFrame({"order_date": dates})
    cleaned, log = fix_dtypes(df, datetime_like_cols=["order_date"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["order_date"])
    assert cleaned["order_date"].isnull().sum() == 1  # invalid_date -> NaT


def test_flag_outliers_detects_extreme_value():
    df = pd.DataFrame({"age": [28, 29, 31, 33, 35, 120]})  # 120 is an outlier
    log, report = flag_outliers(df)
    assert len(report) == 1
    assert report[0]["column"] == "age"
    assert 120 in report[0]["outlier_values"]


def test_flag_outliers_does_not_modify_data():
    df = pd.DataFrame({"age": [28, 29, 31, 33, 35, 120]})
    log, report = flag_outliers(df)
    assert df["age"].tolist() == [28, 29, 31, 33, 35, 120]  # unchanged


def test_flag_outliers_no_outliers():
    df = pd.DataFrame({"age": [28, 29, 30, 31, 32]})
    log, report = flag_outliers(df)
    assert report == []


def test_clean_dataset_on_sample_csv():
    path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_sales.csv")
    df = pd.read_csv(path)

    result = clean_dataset(df, datetime_like_cols=["order_date"])

    cleaned_df = result["cleaned_df"]
    assert cleaned_df.isnull().sum().sum() == 0  # no missing values left
    assert result["rows_before"] == 20
    assert result["total_operations"] > 0
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["order_date"])


def test_clean_dataset_full_pipeline_on_messy_data():
    """End-to-end test replicating the messy dataset issues we found earlier."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 5],
        "name": ["Alice", "Bob", "Charlie", "Dave", "Eve", "Eve"],
        "age": [28, None, 35, 41, 29, 29],
        "city": ["New York", "New York", "Los Angeles", "los angeles", "Chicago", "Chicago"],
        "salary": [55000, 62000, None, 71000, 58000, 58000],
        "join_date": ["2021-03-15", "2021-04-01", "2021-05-20", "invalid_date", "2021-06-10", "2021-06-10"],
        "status": ["Active", "active", "Active", "Inactive", "Active", "Active"],
    })

    result = clean_dataset(df, datetime_like_cols=["join_date"])
    cleaned_df = result["cleaned_df"]

    # Duplicate row removed
    assert result["rows_after"] == 5

    # Case variants merged
    assert cleaned_df["status"].nunique() == 2  # Active, Inactive
    assert cleaned_df["city"].nunique() == 3    # New York, Los Angeles, Chicago

    # No missing values remain (including the NaT from invalid_date)
    assert cleaned_df.isnull().sum().sum() == 0

    # date column properly converted
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["join_date"])
