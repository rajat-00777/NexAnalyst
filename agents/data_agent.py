"""
Data Agent — Step 1 of the pipeline.

Looks at the uploaded dataset and builds a profile of it: how many rows
and columns, what type each column is, how many values are missing, etc.

This is pure Pandas — no AI involved. The numbers here are exact and
repeatable, which is important since every later agent relies on them.
"""



import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """Pure pandas profiling. No LLM involved — fully deterministic."""
    n_rows, n_cols = df.shape

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing = df.isnull().sum().to_dict()
    missing_pct = (df.isnull().mean() * 100).round(2).to_dict()
    duplicates = int(df.duplicated().sum())

    unique_counts = df.nunique().to_dict()

    # naive ID-column guess: object/int columns with (near-)unique values
    possible_id_columns = [
        col for col in df.columns
        if unique_counts.get(col, 0) >= n_rows * 0.95 and n_rows > 1
    ]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    datetime_like_cols = [
        col for col in df.columns
        if "date" in col.lower() or "time" in col.lower()
    ]

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_values": missing,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicates,
        "unique_counts": unique_counts,
        "possible_id_columns": possible_id_columns,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_like_columns": datetime_like_cols,
    }


def data_agent_node(state: dict) -> dict:
    """LangGraph node wrapper."""
    from utils.data_loader import load_dataset

    try:
        df = load_dataset(state["dataset_path"])
        profile = profile_dataset(df)
        state["data_profile"] = profile
        state["current_agent"] = "data_agent"
    except Exception as e:
        state.setdefault("errors", []).append(f"DataAgent error: {e}")

    return state