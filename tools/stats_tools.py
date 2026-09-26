"""
Pure pandas/numpy EDA functions. No LLM calls here.
Every function returns a plain dict/JSON-serializable structure so the
EDA Agent and downstream agents (Insight, Validation) can rely on exact
computed numbers rather than anything the LLM might guess.
"""

import pandas as pd
import numpy as np

# When grouping a numeric column by a categorical column, skip categorical
# columns with more distinct values than this — grouping by something like
# a "name" or "id" column produces a huge, meaningless table.
MAX_GROUPBY_CATEGORIES = 15


# A numeric column where almost every row has a unique value (e.g.
# order_id) is an identifier, not a measurable quantity. Averaging or
# correlating an ID number produces meaningless noise, so we exclude such
# columns from every statistical analysis below.
#
# Detection uses two signals combined, to avoid false positives on small
# datasets where a genuine metric (e.g. 5 different revenue values) can
# also happen to be 100% unique:
#   1. Column name looks like an ID ("id", "_id", "order_id", etc.)
#      combined with high uniqueness -> flagged even on small datasets.
#   2. No name signal, but near-total uniqueness AND enough rows that the
#      pattern is statistically meaningful -> flagged as a fallback.
import re

ID_LIKE_RATIO_THRESHOLD = 0.95
ID_NAME_PATTERN = re.compile(r"(^id$|_id$|^id_|id$)", re.IGNORECASE)
MIN_ROWS_FOR_RATIO_ONLY_DETECTION = 20


def identify_id_like_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Returns numeric columns that look like identifiers (near-unique values)."""
    numeric_cols = df.select_dtypes(include="number").columns
    n_rows = len(df)
    if n_rows == 0:
        return []

    id_like = []
    for col in numeric_cols:
        n_unique = df[col].nunique(dropna=True)
        ratio = n_unique / n_rows
        name_suggests_id = bool(ID_NAME_PATTERN.search(col.lower()))

        if name_suggests_id and ratio >= 0.8:
            id_like.append(col)
        elif n_rows >= MIN_ROWS_FOR_RATIO_ONLY_DETECTION and ratio >= ID_LIKE_RATIO_THRESHOLD:
            id_like.append(col)

    return id_like


def descriptive_stats(df: pd.DataFrame) -> dict:
    """Mean, median, std, min, max, quartiles for every numeric column
    (excluding identifier-like columns such as order_id)."""
    id_like_cols = identify_id_like_numeric_columns(df)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in id_like_cols]
    result = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        result[col] = {
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "q1": round(float(series.quantile(0.25)), 4),
            "q3": round(float(series.quantile(0.75)), 4),
        }

    return result


def correlation_matrix(df: pd.DataFrame) -> dict:
    """
    Pearson correlation between all numeric columns (excluding
    identifier-like columns such as order_id).
    Returns a nested dict: {col_a: {col_b: correlation, ...}, ...}
    Only meaningful with 2+ numeric columns and 2+ rows.
    """
    id_like_cols = identify_id_like_numeric_columns(df)
    numeric_df = df.select_dtypes(include="number").drop(columns=id_like_cols, errors="ignore")
    if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 2:
        return {}

    corr = numeric_df.corr(numeric_only=True).round(4)
    return corr.where(pd.notnull(corr), None).to_dict()


def categorical_summary(df: pd.DataFrame, top_n: int = 5) -> dict:
    """
    For every categorical column: number of unique values, the most
    frequent value, and the top N value counts.
    """
    categorical_cols = df.select_dtypes(include=["object", "str"]).columns
    result = {}

    for col in categorical_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        value_counts = series.value_counts().head(top_n)
        result[col] = {
            "unique_values": int(series.nunique()),
            "most_frequent": str(value_counts.index[0]),
            "top_values": {str(k): int(v) for k, v in value_counts.items()},
        }

    return result


def distribution_shape(df: pd.DataFrame) -> dict:
    """
    Skewness for every numeric column (excluding identifier-like columns),
    plus a plain-language label. Skewness needs at least 3 data points.
    """
    id_like_cols = identify_id_like_numeric_columns(df)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in id_like_cols]
    result = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 3 or series.std() == 0:
            continue

        skew = float(series.skew())

        if skew > 1:
            label = "highly right-skewed"
        elif skew > 0.5:
            label = "moderately right-skewed"
        elif skew < -1:
            label = "highly left-skewed"
        elif skew < -0.5:
            label = "moderately left-skewed"
        else:
            label = "roughly symmetric"

        result[col] = {
            "skewness": round(skew, 4),
            "shape": label,
        }

    return result


def group_comparisons(df: pd.DataFrame) -> dict:
    """
    For each (categorical column, numeric column) pair, computes the mean
    of the numeric column grouped by each category. Skips categorical
    columns with too many distinct values (identifier-like columns), since
    grouping by those produces a huge, meaningless table instead of an
    actual comparison.
    """
    categorical_cols = df.select_dtypes(include=["object", "str"]).columns
    id_like_cols = identify_id_like_numeric_columns(df)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in id_like_cols]
    result = {}

    for cat_col in categorical_cols:
        n_unique = df[cat_col].nunique(dropna=True)
        n_rows = len(df)
        if n_unique == 0:
            continue
        is_identifier_like = n_rows >= 5 and (n_unique / n_rows) > 0.6
        if n_unique > MAX_GROUPBY_CATEGORIES or is_identifier_like:
            continue

        col_result = {}
        for num_col in numeric_cols:
            grouped = df.groupby(cat_col)[num_col].mean().round(4)
            col_result[num_col] = {str(k): float(v) for k, v in grouped.items()}

        if col_result:
            result[cat_col] = col_result

    return result


def outlier_summary(df: pd.DataFrame) -> list[dict]:
    """
    Reuses the same IQR-based outlier detection logic as the Cleaning
    Agent, run on the CLEANED data. Excludes identifier-like columns.
    Detection only — no changes.
    """
    from tools.cleaning_tools import flag_outliers
    id_like_cols = identify_id_like_numeric_columns(df)
    df_for_outliers = df.drop(columns=id_like_cols, errors="ignore")
    _, report = flag_outliers(df_for_outliers)
    return report


def run_eda(df: pd.DataFrame) -> dict:
    """Orchestrates all EDA steps and returns one combined results dict."""
    return {
        "descriptive_stats": descriptive_stats(df),
        "correlation_matrix": correlation_matrix(df),
        "categorical_summary": categorical_summary(df),
        "distribution_shape": distribution_shape(df),
        "group_comparisons": group_comparisons(df),
        "outliers": outlier_summary(df),
    }