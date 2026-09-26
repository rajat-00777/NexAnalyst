"""
Pure functions supporting the Validation Agent: extracting numbers from an
insight's supporting_data text, flattening all real computed numbers from
EDA results into one lookup set, and checking each insight's cited numbers
against that set. No LLM call here — this is a programmatic fact-check,
not another opinion.
"""

import re

# Rounding tolerance when comparing an extracted number against a real
# computed value — insights may quote numbers with slightly different
# rounding than what's stored internally.
NUMBER_MATCH_TOLERANCE = 0.02

# If more than this fraction of insights fail validation, the Validation
# Agent signals that the batch should be regenerated rather than accepting
# a mostly-unreliable set of insights.
FAILURE_RATE_FOR_RETRY = 0.4

MAX_RETRIES = 2


def extract_numbers(text: str) -> list[float]:
    """Extracts all numeric values (including negatives and decimals) from a string."""
    matches = re.findall(r"-?\d+\.?\d*", text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            continue
    return numbers


def flatten_eda_numbers(eda_results: dict) -> set[float]:
    """
    Collects every real computed number from EDA results into one flat set,
    so an insight's cited numbers can be checked against "did this number
    actually get computed anywhere", regardless of which specific section
    it came from.
    """
    numbers = set()

    for stats in eda_results.get("descriptive_stats", {}).values():
        for key in ("mean", "median", "std", "min", "max", "q1", "q3"):
            if key in stats and stats[key] is not None:
                numbers.add(round(float(stats[key]), 2))

    for col_a, row in eda_results.get("correlation_matrix", {}).items():
        for col_b, val in row.items():
            if col_a == col_b or val is None:
                continue  # skip trivial self-correlation (always 1.0)
            numbers.add(round(float(val), 2))

    for info in eda_results.get("categorical_summary", {}).values():
        for count in info.get("top_values", {}).values():
            numbers.add(round(float(count), 2))

    for info in eda_results.get("distribution_shape", {}).values():
        if "skewness" in info:
            numbers.add(round(float(info["skewness"]), 2))

    for numeric_dict in eda_results.get("group_comparisons", {}).values():
        for group_vals in numeric_dict.values():
            for val in group_vals.values():
                numbers.add(round(float(val), 2))

    for outlier in eda_results.get("outliers", []):
        numbers.add(float(outlier.get("count", 0)))
        bounds = outlier.get("bounds", {})
        if "lower" in bounds:
            numbers.add(round(float(bounds["lower"]), 2))
        if "upper" in bounds:
            numbers.add(round(float(bounds["upper"]), 2))

    return numbers


def _has_close_match(number: float, real_numbers: set[float]) -> bool:
    """Checks if a number is within tolerance of any real computed number."""
    for real in real_numbers:
        if abs(number - real) <= NUMBER_MATCH_TOLERANCE:
            return True
    return False


def verify_insight(insight: dict, real_numbers: set[float]) -> dict:
    """
    Checks every number in an insight's supporting_data against the real
    computed numbers. Only supporting_data is checked (not the free-text
    finding), since supporting_data is explicitly meant to cite exact
    figures, while finding may reasonably include general context (e.g.
    "the recommended 8 hours of sleep") that isn't a dataset statistic.
    """
    supporting_text = insight.get("supporting_data", "")
    cited_numbers = extract_numbers(supporting_text)

    if not cited_numbers:
        return {
            "is_valid": False,
            "reason": "No numeric data found in supporting_data — cannot verify",
            "unmatched_numbers": [],
        }

    unmatched = [n for n in cited_numbers if not _has_close_match(n, real_numbers)]

    if unmatched:
        return {
            "is_valid": False,
            "reason": f"{len(unmatched)} cited number(s) don't match any real computed value",
            "unmatched_numbers": unmatched,
        }

    return {
        "is_valid": True,
        "reason": "All cited numbers match real computed values",
        "unmatched_numbers": [],
    }


def validate_insights(insights: list[dict], eda_results: dict) -> dict:
    """
    Validates every insight and returns both the annotated insights and a
    summary (counts, whether a retry is recommended).
    """
    real_numbers = flatten_eda_numbers(eda_results)

    validated = []
    n_valid = 0
    for insight in insights:
        check = verify_insight(insight, real_numbers)
        insight = {**insight, "validation": check}
        validated.append(insight)
        if check["is_valid"]:
            n_valid += 1

    total = len(validated)
    failure_rate = (total - n_valid) / total if total > 0 else 0

    return {
        "validated_insights": validated,
        "total_insights": total,
        "valid_count": n_valid,
        "invalid_count": total - n_valid,
        "failure_rate": round(failure_rate, 3),
        "needs_retry": failure_rate > FAILURE_RATE_FOR_RETRY,
    }