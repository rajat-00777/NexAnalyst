"""
Phase 1 test — no LLM/API key required, pure pandas logic.
Run: pytest tests/test_data_agent.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from agents.data_agent import profile_dataset
from utils.data_loader import load_dataset, DataLoadError


def test_profile_dataset_basic():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["a", "b", "c", "c"],
        "value": [10, None, 30, 40],
    })
    profile = profile_dataset(df)

    assert profile["n_rows"] == 4
    assert profile["n_cols"] == 3
    assert profile["missing_values"]["value"] == 1
    assert "id" in profile["possible_id_columns"]


def test_load_sample_csv():
    path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_sales.csv")
    df = load_dataset(path)
    assert df.shape[0] == 20
    assert "revenue" in df.columns


def test_load_missing_file_raises():
    try:
        load_dataset("does_not_exist.csv")
        assert False, "Should have raised DataLoadError"
    except DataLoadError:
        pass