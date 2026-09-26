"""
Pure pandas cleaning functions. No LLM calls here.
Every function returns (cleaned_df, list_of_operation_logs) so the
Cleaning Agent can report exactly what was done and why.

Design rule: NEVER mutate the input df in place. Always work on a copy,
so the original uploaded dataset is preserved untouched in the caller.
"""

import pandas as pd
import numpy as np

# If a categorical column has more unique values than this fraction of total
# rows, we treat it as "identifier-like" (e.g. names, free text, IDs) and
# avoid mode-filling it, since fabricating a value that repeats an existing
# name/entry is misleading. We fill with "Unknown" instead.
HIGH_CARDINALITY_RATIO = 0.6

# If a column is missing more than this fraction of its values, filling it
# entirely would mean most of the column is fabricated. We skip auto-fill
# and flag it instead so the person (or Report Agent) knows to look closer.
HIGH_MISSING_THRESHOLD = 0.5


def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Drops fully duplicate rows. Logs how many were removed."""
    log = []
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)

    if removed > 0:
        log.append({
            "operation": "remove_duplicates",
            "detail": f"Removed {removed} fully duplicate row(s)",
            "rows_before": before,
            "rows_after": len(cleaned),
        })
    return cleaned, log


def standardize_text_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Two-step text cleanup on object/str columns:
    1. Strip leading/trailing whitespace on every text column (always safe).
    2. Case-insensitive value merging (e.g. "Active"/"active"/"ACTIVE" -> one
       canonical value) — but ONLY on low/medium-cardinality columns, since
       high-cardinality free-text columns (names, addresses) are more likely
       to have legitimately different values that happen to share a lowercase
       form, and merging those would be guessing.
    """
    cleaned = df.copy()
    log = []
    text_cols = cleaned.select_dtypes(include=["object", "str"]).columns

    for col in text_cols:
        non_null = cleaned[col].dropna()
        if non_null.empty:
            continue

        # Step 1: whitespace trim
        stripped = cleaned[col].where(cleaned[col].isna(), cleaned[col].astype(str).str.strip())
        n_trimmed = int((stripped != cleaned[col]).sum())
        cleaned[col] = stripped
        if n_trimmed > 0:
            log.append({
                "operation": "strip_whitespace",
                "column": col,
                "rows_affected": n_trimmed,
            })

        # Step 2: case-insensitive merge, only for low/medium cardinality.
        # IMPORTANT: cardinality must be measured case-insensitively here —
        # otherwise a column like ["Active","active","ACTIVE","Inactive"]
        # looks like 4 unique values (high cardinality) and gets skipped,
        # when really it's just 2 values with inconsistent casing.
        n_rows = len(cleaned)
        non_null_vals = cleaned[col].dropna()
        if non_null_vals.empty:
            continue
        n_unique_ci = non_null_vals.str.lower().nunique()
        if n_rows == 0 or n_unique_ci / n_rows > HIGH_CARDINALITY_RATIO:
            continue  # likely free text / identifier-like, don't merge

        lower_groups = non_null_vals.groupby(non_null_vals.str.lower())
        canonical_map = {}
        merges_found = 0
        for _, group in lower_groups:
            if group.nunique() <= 1:
                continue  # already consistent
            canonical = group.value_counts().idxmax()  # most frequent casing
            for variant in group.unique():
                if variant != canonical:
                    canonical_map[variant] = canonical
                    merges_found += 1

        if canonical_map:
            cleaned[col] = cleaned[col].replace(canonical_map)
            log.append({
                "operation": "normalize_text_case",
                "column": col,
                "detail": f"Merged {merges_found} case-variant value(s) into canonical form",
                "examples": {k: v for k, v in list(canonical_map.items())[:5]},
            })

    return cleaned, log


def fix_dtypes(df: pd.DataFrame, datetime_like_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Attempts to convert columns that look like dates (by name, from the
    Data Agent's profile) into actual datetime dtype. Skips silently if
    conversion fails on more than 20% of values (likely not really a date).
    Invalid individual values become NaT (handled later by missing-value fill).
    """
    cleaned = df.copy()
    log = []
    datetime_like_cols = datetime_like_cols or []

    for col in datetime_like_cols:
        if col not in cleaned.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(cleaned[col]):
            continue

        original_non_null = cleaned[col].notna().sum()
        if original_non_null == 0:
            continue

        converted = pd.to_datetime(cleaned[col], errors="coerce", format="mixed")
        new_nulls = int(converted.isnull().sum() - cleaned[col].isnull().sum())
        failure_rate = converted.isnull().sum() / max(original_non_null, 1)

        if failure_rate <= 0.2:
            cleaned[col] = converted
            entry = {
                "operation": "convert_dtype",
                "column": col,
                "new_dtype": "datetime64[ns]",
                "conversion_failure_rate": round(float(failure_rate), 4),
            }
            if new_nulls > 0:
                entry["detail"] = f"{new_nulls} value(s) could not be parsed as dates and became missing"
            log.append(entry)

    return cleaned, log


def flag_outliers(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """
    Detects outliers in numeric columns using the IQR method.
    Does NOT modify or remove any data — outliers might be real and
    meaningful (e.g. a genuinely huge sale). We only flag them so the
    person and the Report Agent are aware, rather than silently guessing.
    Returns (log_entries, outlier_report).
    """
    log = []
    report = []
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = int(outlier_mask.sum())

        if n_outliers > 0:
            outlier_values = df.loc[outlier_mask, col].tolist()
            report.append({
                "column": col,
                "count": n_outliers,
                "bounds": {"lower": round(float(lower), 4), "upper": round(float(upper), 4)},
                "outlier_values": outlier_values[:10],  # cap for readability
            })
            log.append({
                "operation": "flag_outliers",
                "column": col,
                "detail": f"{n_outliers} value(s) fall outside the normal IQR range (not modified)",
                "bounds": {"lower": round(float(lower), 4), "upper": round(float(upper), 4)},
            })

    return log, report


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Fills missing values column by column, type-aware:
    - Numeric columns: fill with median
    - Datetime columns: fill with mode (most common date)
    - Categorical/object columns: fill with mode, UNLESS the column is
      high-cardinality (identifier-like), in which case we fill with
      "Unknown" instead of fabricating a repeated name/value.
    - Any column missing more than HIGH_MISSING_THRESHOLD of its values is
      SKIPPED (not filled) and flagged instead — filling it would mean most
      of the column is made up.
    Every action is logged individually.
    """
    cleaned = df.copy()
    log = []
    n_rows = len(cleaned)

    numeric_cols = cleaned.select_dtypes(include="number").columns
    datetime_cols = cleaned.select_dtypes(include="datetime").columns
    categorical_cols = cleaned.select_dtypes(include=["object", "str"]).columns

    def missing_ratio(col):
        return cleaned[col].isnull().sum() / n_rows if n_rows else 0

    # --- Numeric ---
    for col in numeric_cols:
        n_missing = int(cleaned[col].isnull().sum())
        if n_missing == 0:
            continue

        ratio = missing_ratio(col)
        if ratio > HIGH_MISSING_THRESHOLD:
            log.append({
                "operation": "high_missing_warning",
                "column": col,
                "detail": f"{round(ratio * 100, 1)}% missing — left as-is instead of auto-filling",
                "rows_affected": n_missing,
            })
            continue

        fill_value = cleaned[col].median() if numeric_strategy == "median" else cleaned[col].mean()
        cleaned[col] = cleaned[col].fillna(fill_value)
        log.append({
            "operation": "fill_missing_numeric",
            "column": col,
            "strategy": numeric_strategy,
            "fill_value": round(float(fill_value), 4),
            "rows_affected": n_missing,
        })

    # --- Datetime ---
    for col in datetime_cols:
        n_missing = int(cleaned[col].isnull().sum())
        if n_missing == 0:
            continue

        ratio = missing_ratio(col)
        if ratio > HIGH_MISSING_THRESHOLD:
            log.append({
                "operation": "high_missing_warning",
                "column": col,
                "detail": f"{round(ratio * 100, 1)}% missing — left as-is instead of auto-filling",
                "rows_affected": n_missing,
            })
            continue

        mode_series = cleaned[col].mode(dropna=True)
        if mode_series.empty:
            continue
        fill_value = mode_series.iloc[0]
        cleaned[col] = cleaned[col].fillna(fill_value)
        log.append({
            "operation": "fill_missing_datetime",
            "column": col,
            "strategy": "mode",
            "fill_value": str(fill_value),
            "rows_affected": n_missing,
        })

    # --- Categorical ---
    for col in categorical_cols:
        n_missing = int(cleaned[col].isnull().sum())
        if n_missing == 0:
            continue

        ratio = missing_ratio(col)
        if ratio > HIGH_MISSING_THRESHOLD:
            log.append({
                "operation": "high_missing_warning",
                "column": col,
                "detail": f"{round(ratio * 100, 1)}% missing — left as-is instead of auto-filling",
                "rows_affected": n_missing,
            })
            continue

        n_unique = cleaned[col].nunique(dropna=True)
        is_identifier_like = n_rows > 0 and (n_unique / n_rows) > HIGH_CARDINALITY_RATIO

        if is_identifier_like:
            cleaned[col] = cleaned[col].fillna("Unknown")
            log.append({
                "operation": "fill_missing_categorical",
                "column": col,
                "strategy": "unknown_placeholder",
                "detail": "Column looks identifier-like (high cardinality) — filled with 'Unknown' instead of the most common value, to avoid fabricating a duplicate entry",
                "fill_value": "Unknown",
                "rows_affected": n_missing,
            })
        else:
            mode_series = cleaned[col].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)
            log.append({
                "operation": "fill_missing_categorical",
                "column": col,
                "strategy": "mode",
                "fill_value": str(fill_value),
                "rows_affected": n_missing,
            })

    return cleaned, log


def clean_dataset(df: pd.DataFrame, datetime_like_cols: list[str] | None = None) -> dict:
    """
    Orchestrates all cleaning steps in order and returns a full result dict:
    the cleaned dataframe, a combined operation log, outlier report, and
    before/after shape.

    Order matters:
    1. Remove exact duplicate rows
    2. Standardize text (whitespace + case) BEFORE filling, so mode
       calculations aren't split across case variants
    3. Convert date-like columns to real datetime (may introduce new NaT
       for unparseable values)
    4. Fill missing values (now type-aware: numeric/datetime/categorical),
       AFTER dtype conversion so NaTs created in step 3 get handled too
    5. Flag outliers (detection only — never modifies data)
    """
    original_shape = df.shape

    step1_df, log1 = remove_duplicate_rows(df)
    step2_df, log2 = standardize_text_columns(step1_df)
    step3_df, log3 = fix_dtypes(step2_df, datetime_like_cols)
    step4_df, log4 = handle_missing_values(step3_df)
    log5, outlier_report = flag_outliers(step4_df)

    full_log = log1 + log2 + log3 + log4 + log5

    return {
        "cleaned_df": step4_df,
        "operations_log": full_log,
        "outliers_detected": outlier_report,
        "rows_before": original_shape[0],
        "rows_after": step4_df.shape[0],
        "cols_before": original_shape[1],
        "cols_after": step4_df.shape[1],
        "total_operations": len(full_log),
    }
