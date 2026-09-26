"""
Loads CSV/Excel files into a pandas DataFrame with basic validation.
No LLM calls here — pure I/O and sanity checks.
"""

import os
import pandas as pd


class DataLoadError(Exception):
    pass


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise DataLoadError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            raise DataLoadError(f"Unsupported file type: {ext}")
    except Exception as e:
        raise DataLoadError(f"Failed to read file: {e}")

    if df.empty:
        raise DataLoadError("Dataset is empty.")
    if df.shape[1] == 0:
        raise DataLoadError("Dataset has no columns.")

    return df