"""
Phase 3 tests — pure pandas/numpy logic, no LLM/API key required.
Run: pytest tests/test_eda_agent.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from tools.stats_tools import (
    descriptive_stats,
    correlation_matrix,
    categorical_summary,
    distribution_shape,
    group_comparisons,
    outlier_summary,
    run_eda,
)


def test_descriptive_stats_basic():
    df = pd.DataFrame({"val": [10, 20, 30, 40, 50]})
    stats = descriptive_stats(df)
    assert stats["val"]["mean"] == 30.0
    assert stats["val"]["median"] == 30.0
    assert stats["val"]["min"] == 10.0
    assert stats["val"]["max"] == 50.0


def test_descriptive_stats_ignores_non_numeric():
    df = pd.DataFrame({"val": [1, 2, 3], "name": ["a", "b", "c"]})
    stats = descriptive_stats(df)
    assert "val" in stats
    assert "name" not in stats


def test_correlation_matrix_perfect_correlation():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
    corr = correlation_matrix(df)
    assert corr["a"]["b"] == 1.0


def test_correlation_matrix_single_numeric_column_returns_empty():
    df = pd.DataFrame({"a": [1, 2, 3], "name": ["x", "y", "z"]})
    corr = correlation_matrix(df)
    assert corr == {}


def test_categorical_summary_basic():
    df = pd.DataFrame({"status": ["Active", "Active", "Inactive"]})
    summary = categorical_summary(df)
    assert summary["status"]["unique_values"] == 2
    assert summary["status"]["most_frequent"] == "Active"
    assert summary["status"]["top_values"]["Active"] == 2


def test_distribution_shape_symmetric():
    df = pd.DataFrame({"val": [10, 20, 30, 40, 50]})
    shape = distribution_shape(df)
    assert shape["val"]["shape"] == "roughly symmetric"


def test_distribution_shape_right_skewed():
    df = pd.DataFrame({"val": [1, 2, 2, 3, 3, 3, 100]})  # long tail to the right
    shape = distribution_shape(df)
    assert shape["val"]["skewness"] > 0.5


def test_group_comparisons_basic():
    df = pd.DataFrame({
        "region": ["North", "North", "South"],
        "revenue": [100, 200, 300],
    })
    result = group_comparisons(df)
    assert result["region"]["revenue"]["North"] == 150.0
    assert result["region"]["revenue"]["South"] == 300.0


def test_group_comparisons_skips_high_cardinality():
    # 5 unique names in 5 rows -> identifier-like -> should be skipped
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
        "score": [1, 2, 3, 4, 5],
    })
    result = group_comparisons(df)
    assert "name" not in result


def test_outlier_summary_detects_outlier():
    df = pd.DataFrame({"age": [28, 29, 31, 33, 35, 120]})
    report = outlier_summary(df)
    assert len(report) == 1
    assert report[0]["column"] == "age"


def test_descriptive_stats_excludes_id_like_column():
    df = pd.DataFrame({
        "order_id": [1001, 1002, 1003, 1004, 1005],  # all unique -> ID-like
        "revenue": [50, 60, 70, 80, 90],
    })
    stats = descriptive_stats(df)
    assert "revenue" in stats
    assert "order_id" not in stats


def test_correlation_matrix_excludes_id_like_column():
    df = pd.DataFrame({
        "order_id": [1001, 1002, 1003, 1004, 1005],
        "revenue": [50, 60, 70, 80, 90],
        "quantity": [1, 2, 3, 4, 5],
    })
    corr = correlation_matrix(df)
    assert "order_id" not in corr
    assert "revenue" in corr


def test_run_eda_on_sample_csv():
    """Full EDA pipeline against the cleaned version of the sample sales data."""
    from tools.cleaning_tools import clean_dataset

    path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_sales.csv")
    df = pd.read_csv(path)
    cleaning_result = clean_dataset(df, datetime_like_cols=["order_date"])
    cleaned_df = cleaning_result["cleaned_df"]

    eda = run_eda(cleaned_df)

    assert "revenue" in eda["descriptive_stats"]
    assert "quantity" in eda["descriptive_stats"]
    assert "order_id" not in eda["descriptive_stats"]  # identifier-like, excluded
    assert eda["correlation_matrix"] != {}
    assert "product" in eda["categorical_summary"]
    assert "region" in eda["group_comparisons"]  # low cardinality -> included
    assert isinstance(eda["outliers"], list)