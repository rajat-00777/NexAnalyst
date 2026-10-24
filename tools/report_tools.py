"""
Builds the final report, section by section. No AI here — just
formats what earlier agents already found and checked. Only validated
insights and checked recommendations are included.

Written as clean Markdown: summary table up top, real tables instead
of long lists, key findings highlighted so they stand out.
"""

import os
from datetime import datetime

PRIORITY_ICON = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def _at_a_glance_table(data_profile: dict, cleaning_results: dict, validated_insights: list[dict],
                        recommendations: list[dict], visualization_results: dict) -> str:
    n_recs = len([r for r in recommendations if r.get("grounding", {}).get("is_valid")])
    n_charts = len([c for c in visualization_results.get("charts", []) if c.get("source") == "insight"])

    rows = [
        ("Rows", data_profile.get("n_rows", "?")),
        ("Columns", data_profile.get("n_cols", "?")),
        ("Cleaning operations applied", cleaning_results.get("total_operations", 0)),
        ("Validated findings", len(validated_insights)),
        ("Recommendations", n_recs),
        ("Key charts generated", n_charts),
    ]

    lines = ["| Metric | Value |", "|---|---|"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def _format_executive_summary(data_profile: dict, validated_insights: list[dict], recommendations: list[dict]) -> str:
    n_rows = data_profile.get("n_rows", "?")
    n_cols = data_profile.get("n_cols", "?")
    n_insights = len(validated_insights)
    n_recs = len(recommendations)

    summary = (
        f"This report analyzes a dataset of **{n_rows} rows** and **{n_cols} columns**. "
        f"The analysis produced **{n_insights} validated finding(s)** and **{n_recs} recommendation(s)**, "
        f"all fact-checked against the actual computed data — not estimated or guessed."
    )

    top_finding = next((i for i in validated_insights if i.get("importance") == "high"), None)
    if top_finding:
        summary += f"\n\n> **Headline finding:** {top_finding['finding']}"

    return summary


def _format_dataset_overview(data_profile: dict) -> str:
    numeric = ", ".join(data_profile.get("numeric_columns", [])) or "none"
    categorical = ", ".join(data_profile.get("categorical_columns", [])) or "none"
    datetime_cols = ", ".join(data_profile.get("datetime_like_columns", [])) or "none"

    lines = [
        "| Property | Value |",
        "|---|---|",
        f"| Rows | {data_profile.get('n_rows', '?')} |",
        f"| Columns | {data_profile.get('n_cols', '?')} |",
        f"| Numeric columns | {numeric} |",
        f"| Categorical columns | {categorical} |",
        f"| Date/time columns | {datetime_cols} |",
    ]
    return "\n".join(lines)


def _format_data_quality(cleaning_results: dict) -> str:
    total_ops = cleaning_results.get("total_operations", 0)
    rows_before = cleaning_results.get("rows_before", "?")
    rows_after = cleaning_results.get("rows_after", "?")

    header = f"**{total_ops} cleaning operation(s)** were applied. Rows: {rows_before} → {rows_after}.\n"

    ops = cleaning_results.get("operations_log", [])
    if not ops:
        return header + "\n_No cleaning operations were needed._"

    lines = [header, "| Operation | Column | Detail |", "|---|---|---|"]
    for op in ops:
        detail = op.get("detail") or op.get("fill_value") or "completed"
        col = op.get("column", "—")
        lines.append(f"| {op['operation']} | {col} | {detail} |")

    return "\n".join(lines)


def _format_key_findings(validated_insights: list[dict]) -> str:
    if not validated_insights:
        return "_No findings passed validation._"

    lines = []
    for insight in validated_insights:
        icon = PRIORITY_ICON.get(insight.get("importance", "low"), "⚪")
        if insight.get("importance") == "high":
            # Blockquote makes high-importance findings visually stand out
            lines.append(f"> {icon} **{insight['finding']}**\n> \n> _Supporting data: {insight['supporting_data']}_")
        else:
            lines.append(f"- {icon} **{insight['finding']}** _(supporting data: {insight['supporting_data']})_")
        lines.append("")

    return "\n".join(lines)


def _format_visualizations(visualization_results: dict) -> str:
    charts = visualization_results.get("charts", [])
    insight_charts = [c for c in charts if c.get("source") == "insight"]

    if not insight_charts:
        return "_No insight-driven visualizations were generated._"

    lines = ["The following charts were generated to illustrate the key findings above:\n"]
    lines.append("| Chart | File |")
    lines.append("|---|---|")
    for c in insight_charts:
        lines.append(f"| {c['description']} | `{os.path.basename(c['path'])}` |")
    return "\n".join(lines)


def _format_anomalies(cleaning_results: dict) -> str:
    outliers = cleaning_results.get("outliers_detected", [])
    if not outliers:
        return "_No significant outliers were detected._"

    lines = ["| Column | Outlier Count | Normal Range |", "|---|---|---|"]
    for o in outliers:
        lines.append(f"| {o['column']} | {o['count']} | {o['bounds']['lower']} to {o['bounds']['upper']} |")
    return "\n".join(lines)


def _format_recommendations(recommendations: list[dict]) -> str:
    valid_recs = [r for r in recommendations if r.get("grounding", {}).get("is_valid")]
    if not valid_recs:
        return "_No recommendations passed grounding checks._"

    lines = ["_Fact = confirmed from your data. Suggestion = AI's practical advice based on that fact._\n"]
    for r in valid_recs:
        icon = PRIORITY_ICON.get(r.get("priority", "low"), "⚪")
        lines.append(f"**{icon} {r['priority'].upper()} priority**")
        lines.append(f"- 📊 **Fact:** {r['fact']}")
        lines.append(f"- 💡 **Suggestion:** {r['suggestion']}")
        lines.append("")

    return "\n".join(lines)


def _format_conclusion(validated_insights: list[dict], recommendations: list[dict]) -> str:
    n_insights = len(validated_insights)
    n_recs = len([r for r in recommendations if r.get("grounding", {}).get("is_valid")])

    if n_insights == 0:
        return "No findings could be validated against the dataset. Consider reviewing the data quality or uploading a larger dataset for more reliable analysis."

    return (
        f"This analysis surfaced **{n_insights} data-backed finding(s)** and **{n_recs} actionable recommendation(s)**. "
        f"All findings and recommendations were programmatically checked against the actual computed statistics "
        f"to reduce the risk of inaccurate or fabricated claims."
    )


def assemble_report(state: dict) -> str:
    """
    Builds the full report as a Markdown string from already-computed,
    already-validated state. No new claims are generated here.
    """
    data_profile = state.get("data_profile", {})
    cleaning_results = state.get("cleaning_results", {})
    insights = state.get("insights", [])
    recommendations = state.get("recommendations", [])
    visualization_results = state.get("visualization_results", {})

    validated_insights = [i for i in insights if i.get("validation", {}).get("is_valid")]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""#AI Data Analyst — Report

_Generated: {timestamp}_

---

## 📋 At a Glance

{_at_a_glance_table(data_profile, cleaning_results, validated_insights, recommendations, visualization_results)}

---

## 📝 Executive Summary

{_format_executive_summary(data_profile, validated_insights, recommendations)}

---

## 📂 Dataset Overview

{_format_dataset_overview(data_profile)}

---

## 🧹 Data Quality

{_format_data_quality(cleaning_results)}

---

## 💡 Key Findings

{_format_key_findings(validated_insights)}

---

## 🖼️ Visualizations

{_format_visualizations(visualization_results)}

---

## ⚠️ Anomalies

{_format_anomalies(cleaning_results)}

---

## ✅ Recommendations

{_format_recommendations(recommendations)}

---

## 🏁 Conclusion

{_format_conclusion(validated_insights, recommendations)}
"""
    return report


def save_report(markdown_text: str, output_dir: str, filename: str = "analysis_report.md") -> str:
    """Writes the report to disk and returns the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    return path