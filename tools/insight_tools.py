"""
Helper functions for the Insight Agent: writing the prompt sent to the
AI, and checking what it sends back.

Kept separate from the actual AI call so this logic can be tested on
its own, without needing a live connection every time.
"""

import json

MAX_INSIGHTS = 8


def _summarize_eda_for_prompt(eda_results: dict) -> str:
    """
    Condenses EDA results into a compact text block for the prompt.
    Full raw JSON (especially group_comparisons/correlation matrices) can
    get large and unfocused — this pulls out the numbers most likely to
    matter, so the LLM has real data to reason over without noise.
    """
    lines = []

    desc = eda_results.get("descriptive_stats", {})
    if desc:
        lines.append("DESCRIPTIVE STATISTICS:")
        for col, stats in desc.items():
            lines.append(f"  {col}: mean={stats['mean']}, median={stats['median']}, "
                         f"std={stats['std']}, min={stats['min']}, max={stats['max']}")

    corr = eda_results.get("correlation_matrix", {})
    if corr:
        lines.append("\nNOTABLE CORRELATIONS (|correlation| >= 0.3):")
        seen = set()
        for col_a, row in corr.items():
            for col_b, val in row.items():
                if col_a == col_b or val is None:
                    continue
                pair = tuple(sorted([col_a, col_b]))
                if pair in seen or abs(val) < 0.3:
                    continue
                seen.add(pair)
                lines.append(f"  {col_a} vs {col_b}: {val}")

    cat = eda_results.get("categorical_summary", {})
    if cat:
        lines.append("\nCATEGORICAL BREAKDOWN:")
        for col, info in cat.items():
            lines.append(f"  {col}: most common = {info['most_frequent']}, "
                         f"top values = {info['top_values']}")

    dist = eda_results.get("distribution_shape", {})
    if dist:
        lines.append("\nDISTRIBUTION SHAPE:")
        for col, info in dist.items():
            lines.append(f"  {col}: {info['shape']} (skewness={info['skewness']})")

    groups = eda_results.get("group_comparisons", {})
    if groups:
        lines.append("\nGROUP COMPARISONS (average of numeric column, by category):")
        for cat_col, numeric_dict in groups.items():
            for num_col, group_vals in numeric_dict.items():
                lines.append(f"  Average {num_col} by {cat_col}: {group_vals}")

    outliers = eda_results.get("outliers", [])
    if outliers:
        lines.append("\nOUTLIERS DETECTED:")
        for o in outliers:
            lines.append(f"  {o['column']}: {o['count']} outlier value(s), "
                         f"normal range is {o['bounds']['lower']} to {o['bounds']['upper']}")

    return "\n".join(lines)


def build_insight_prompt(data_profile: dict, cleaning_results: dict, eda_results: dict) -> str:
    """Builds the full prompt sent to the LLM for insight generation."""
    n_rows = data_profile.get("n_rows", "unknown")
    n_cols = data_profile.get("n_cols", "unknown")
    columns = data_profile.get("columns", [])
    cleaning_ops = cleaning_results.get("total_operations", 0)

    eda_summary = _summarize_eda_for_prompt(eda_results)

    prompt = f"""You are a data analyst writing insights about a dataset. You must ONLY use the exact numbers provided below — never invent, estimate, or guess any statistic that isn't explicitly given to you.

DATASET OVERVIEW:
- {n_rows} rows, {n_cols} columns
- Columns: {', '.join(columns)}
- {cleaning_ops} cleaning operation(s) were applied before this analysis

COMPUTED STATISTICS (this is the ONLY data you may reference):
{eda_summary}

TASK:
Write {MAX_INSIGHTS} insights about this dataset. Use your judgment to identify what the dataset is actually ABOUT and prioritize insights that matter most for that purpose — not just the statistically largest numbers. For example, if the dataset is clearly about an outcome (a score, a result, a target metric), prioritize insights connecting other columns to that outcome.

Respond with ONLY a JSON array, no other text, no markdown code fences. Each insight must be an object with exactly these fields:
- "finding": one clear sentence describing the insight
- "category": one of "correlation", "group_comparison", "distribution", "outlier", "general"
- "supporting_data": a short string quoting the EXACT numbers from above that support this finding
- "importance": one of "high", "medium", "low"

Example format:
[{{"finding": "...", "category": "correlation", "supporting_data": "...", "importance": "high"}}]
"""
    return prompt


def parse_insights_response(raw_text: str) -> list[dict]:
    """
    Parses the LLM's raw text response into a list of insight dicts.
    Strips markdown code fences if the model added them despite
    instructions not to, since that's a common LLM habit.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        insights = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse LLM response as JSON: {e}\nRaw response: {raw_text[:500]}")

    if not isinstance(insights, list):
        raise ValueError("Expected a JSON array of insights")

    required_fields = {"finding", "category", "supporting_data", "importance"}
    for i, insight in enumerate(insights):
        if not isinstance(insight, dict):
            raise ValueError(f"Insight {i} is not an object")
        missing = required_fields - set(insight.keys())
        if missing:
            raise ValueError(f"Insight {i} is missing fields: {missing}")

    return insights[:MAX_INSIGHTS]


def check_grounding(insights: list[dict], data_profile: dict) -> list[dict]:
    """
    Light sanity check: flags any insight that references a column name
    which doesn't actually exist in the dataset — a common sign of
    hallucination. Does NOT verify the specific numbers are correct
    (that's the Validation Agent's job in Phase 6) — this just catches
    the most obvious case of talking about data that isn't there.
    """
    real_columns = set(data_profile.get("columns", []))

    for insight in insights:
        text = f"{insight.get('finding', '')} {insight.get('supporting_data', '')}"
        mentioned_columns = [col for col in real_columns if col in text]
        insight["grounding_check"] = "ok" if mentioned_columns else "no_column_reference_found"
        insight["referenced_columns"] = mentioned_columns

    return insights