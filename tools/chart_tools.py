"""
Builds all the charts using Plotly, saved as interactive HTML files
(no extra software needed).

Two kinds of charts:
1. Overview — general stats like histograms and heatmaps, picked
   automatically from the data.
2. Key Findings — one chart per validated insight, built around the
   exact columns that insight mentions.

A few things learned from testing on real data:
- Chart titles stay short (e.g. "X vs Y") — the full explanation is
  already shown above the chart, so long titles used to get cut off.
- Labels tilt sideways automatically when there are many categories,
  so they don't overlap.
- Each chart records its own ideal height, so it doesn't get cropped.
"""



import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from tools.stats_tools import identify_id_like_numeric_columns

MAX_HISTOGRAMS = 8
MAX_BAR_CHARTS = 3
MAX_BOX_PLOTS = 2
MAX_PIE_CHARTS = 2
MAX_SCATTER_PLOTS = 2
MAX_INSIGHT_CHARTS = 6
STRONG_CORRELATION_THRESHOLD = 0.5
PIE_MAX_CATEGORIES = 6

# Once a categorical axis has more than this many distinct values, rotate
# the tick labels so they don't overlap each other.
TICK_ROTATION_THRESHOLD = 5

COLOR_SEQUENCE = px.colors.qualitative.Set2
TEMPLATE = "plotly_white"
DEFAULT_HEIGHT = 500


def _apply_theme(fig, height=DEFAULT_HEIGHT):
    fig.update_layout(
        template=TEMPLATE, height=height, font=dict(size=13), title_font_size=18,
        margin=dict(l=60, r=40, t=70, b=90),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _rotate_ticks_if_crowded(fig, n_categories: int):
    """Rotates X-axis tick labels once there are enough categories that
    horizontal labels would overlap. Generic — works for any dataset's
    category count, not tuned to a specific column."""
    if n_categories > TICK_ROTATION_THRESHOLD:
        fig.update_xaxes(tickangle=-45)
    return fig


def _save(fig, output_dir: str, filename: str) -> str:
    path = os.path.join(output_dir, filename)
    # responsive=True makes the figure resize to fit whatever container it's
    # embedded in, rather than rendering at a fixed pixel width. Without
    # this, a chart given an explicit large width (like the heatmap) gets
    # cropped when displayed in a narrower container (e.g. a half-width
    # column in a 2-up grid) instead of shrinking to fit.
    fig.write_html(path, include_plotlyjs="cdn", config={"responsive": True})
    return path


def create_histograms(df: pd.DataFrame, output_dir: str) -> list[dict]:
    charts = []
    id_like_cols = identify_id_like_numeric_columns(df)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in id_like_cols][:MAX_HISTOGRAMS]
    for i, col in enumerate(numeric_cols):
        fig = px.histogram(df, x=col, title=f"Distribution of {col}",
                           color_discrete_sequence=[COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]], marginal="box")
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        fig.update_layout(xaxis_title=col, yaxis_title="Count")
        _apply_theme(fig)
        path = _save(fig, output_dir, f"histogram_{col}.html")
        charts.append({"type": "histogram", "column": col, "path": path, "height": DEFAULT_HEIGHT,
                       "description": f"Shows how {col} values are distributed, with a box plot above for quick outlier reference"})
    return charts


def create_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> dict | None:
    id_like_cols = identify_id_like_numeric_columns(df)
    numeric_df = df.select_dtypes(include="number").drop(columns=id_like_cols, errors="ignore")
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr(numeric_only=True).round(2)
    n = len(corr.columns)
    size_per_col = 90
    dynamic_height = max(500, 150 + n * size_per_col)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    aspect="auto", title="Correlation Heatmap")
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=12))
    fig.update_yaxes(tickfont=dict(size=12))
    fig.update_layout(template=TEMPLATE, height=dynamic_height, autosize=True,
                      font=dict(size=13), title_font_size=18, margin=dict(l=100, r=40, t=70, b=120))
    path = _save(fig, output_dir, "correlation_heatmap.html")
    return {"type": "heatmap", "path": path, "height": dynamic_height, "full_width": True,
            "description": "Shows how strongly each pair of numeric columns is related (closer to 1 or -1 = stronger relationship)"}


def create_bar_charts_from_groups(eda_results: dict, output_dir: str) -> list[dict]:
    """Ranks by normalized spread (spread / std) to fairly compare metrics
    on different scales, and enforces diversity across numeric metrics."""
    group_comparisons = eda_results.get("group_comparisons", {})
    descriptive_stats = eda_results.get("descriptive_stats", {})

    candidates = []
    for cat_col, numeric_dict in group_comparisons.items():
        for num_col, group_values in numeric_dict.items():
            values = list(group_values.values())
            if len(values) < 2:
                continue
            spread = max(values) - min(values)
            std = descriptive_stats.get(num_col, {}).get("std", 0)
            normalized_spread = spread / std if std > 0 else spread
            candidates.append((normalized_spread, cat_col, num_col, group_values))
    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = []
    used_num_cols = set()
    for cand in candidates:
        if len(selected) >= MAX_BAR_CHARTS:
            break
        if cand[2] not in used_num_cols:
            selected.append(cand)
            used_num_cols.add(cand[2])
    if len(selected) < MAX_BAR_CHARTS:
        selected_keys = {(c[1], c[2]) for c in selected}
        for cand in candidates:
            if len(selected) >= MAX_BAR_CHARTS:
                break
            if (cand[1], cand[2]) not in selected_keys:
                selected.append(cand)
                selected_keys.add((cand[1], cand[2]))

    charts = []
    for normalized_spread, cat_col, num_col, group_values in selected:
        sorted_items = sorted(group_values.items(), key=lambda x: x[1], reverse=True)
        categories = [k for k, v in sorted_items]
        values = [v for k, v in sorted_items]
        fig = px.bar(x=categories, y=values, title=f"Average {num_col} by {cat_col}",
                    labels={"x": cat_col, "y": f"Average {num_col}"}, color=categories,
                    color_discrete_sequence=COLOR_SEQUENCE, text=[f"{v:,.1f}" for v in values])
        fig.update_traces(textposition="outside", marker_line_color="rgba(0,0,0,0.3)", marker_line_width=1.2)
        fig.update_layout(showlegend=False, xaxis_title=cat_col, yaxis_title=f"Average {num_col}")
        _rotate_ticks_if_crowded(fig, len(categories))
        # Extra height when labels are rotated so they don't get clipped at the bottom
        height = DEFAULT_HEIGHT + (80 if len(categories) > TICK_ROTATION_THRESHOLD else 0)
        _apply_theme(fig, height=height)
        path = _save(fig, output_dir, f"bar_{num_col}_by_{cat_col}.html")
        charts.append({"type": "bar", "columns": [cat_col, num_col], "path": path, "height": height,
                       "description": f"Compares average {num_col} across {cat_col} groups, sorted highest to lowest"})
    return charts


def create_box_plots(df: pd.DataFrame, outlier_columns: list[str], output_dir: str) -> list[dict]:
    charts = []
    for i, col in enumerate(outlier_columns[:MAX_BOX_PLOTS]):
        if col not in df.columns:
            continue
        fig = px.box(df, y=col, title=f"{col} — Spread and Outliers", points="all",
                    color_discrete_sequence=[COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]])
        fig.update_traces(marker=dict(size=5, opacity=0.6), jitter=0.3)
        fig.update_layout(yaxis_title=col)
        _apply_theme(fig)
        path = _save(fig, output_dir, f"boxplot_{col}.html")
        charts.append({"type": "box", "column": col, "path": path, "height": DEFAULT_HEIGHT,
                       "description": f"Visualizes the spread of {col} — the box shows the middle 50% of values, dots are individual data points including outliers"})
    return charts


def create_pie_charts(df: pd.DataFrame, output_dir: str) -> list[dict]:
    charts = []
    categorical_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in categorical_cols:
        if len(charts) >= MAX_PIE_CHARTS:
            break
        n_unique = df[col].nunique(dropna=True)
        if n_unique == 0 or n_unique > PIE_MAX_CATEGORIES:
            continue
        counts = df[col].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, title=f"{col} Breakdown",
                    color_discrete_sequence=COLOR_SEQUENCE, hole=0.35)
        fig.update_traces(textinfo="label+percent+value", textposition="outside",
                          marker=dict(line=dict(color="white", width=2)))
        _apply_theme(fig)
        path = _save(fig, output_dir, f"pie_{col}.html")
        charts.append({"type": "pie", "column": col, "path": path, "height": DEFAULT_HEIGHT,
                       "description": f"Shows the proportion and count of each category in {col}"})
    return charts


def create_scatter_plots(df: pd.DataFrame, correlation_matrix: dict, output_dir: str) -> list[dict]:
    """Ranks by |correlation| strength (not dict order) before selecting top pairs."""
    seen_pairs = set()
    candidates = []
    for col_a, row in correlation_matrix.items():
        for col_b, corr_value in row.items():
            if col_a == col_b or corr_value is None:
                continue
            pair_key = tuple(sorted([col_a, col_b]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            if abs(corr_value) >= STRONG_CORRELATION_THRESHOLD:
                candidates.append((abs(corr_value), col_a, col_b, corr_value))
    candidates.sort(key=lambda x: x[0], reverse=True)

    charts = []
    for _, col_a, col_b, corr_value in candidates[:MAX_SCATTER_PLOTS]:
        direction = "positive" if corr_value > 0 else "negative"
        fig = px.scatter(df, x=col_a, y=col_b, trendline="ols",
                         title=f"{col_a} vs {col_b} ({direction}, r={corr_value})",
                         color_discrete_sequence=[COLOR_SEQUENCE[0]])
        fig.update_traces(marker=dict(size=9, opacity=0.7, line=dict(width=1, color="white")))
        fig.update_layout(xaxis_title=col_a, yaxis_title=col_b)
        _apply_theme(fig)
        path = _save(fig, output_dir, f"scatter_{col_a}_vs_{col_b}.html")
        charts.append({"type": "scatter", "columns": [col_a, col_b], "correlation": corr_value, "path": path, "height": DEFAULT_HEIGHT,
                       "description": f"Shows the {direction} relationship between {col_a} and {col_b} (correlation: {corr_value}), with a trend line"})
    return charts


def create_time_series_chart(df: pd.DataFrame, output_dir: str) -> dict | None:
    datetime_cols = df.select_dtypes(include="datetime").columns
    id_like_cols = identify_id_like_numeric_columns(df)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in id_like_cols]
    if len(datetime_cols) == 0 or len(numeric_cols) == 0:
        return None
    date_col = datetime_cols[0]
    num_col = numeric_cols[0]
    trend_df = df[[date_col, num_col]].dropna().sort_values(date_col)
    if trend_df.empty:
        return None
    fig = px.line(trend_df, x=date_col, y=num_col, title=f"{num_col} Over Time", markers=True,
                 color_discrete_sequence=[COLOR_SEQUENCE[1]])
    fig.update_traces(line=dict(width=2.5), marker=dict(size=6))
    fig.update_layout(xaxis_title=date_col, yaxis_title=num_col)
    _apply_theme(fig)
    path = _save(fig, output_dir, f"timeseries_{num_col}.html")
    return {"type": "time_series", "columns": [date_col, num_col], "path": path, "height": DEFAULT_HEIGHT,
            "description": f"Shows how {num_col} changes over time"}


def generate_visualizations(df: pd.DataFrame, eda_results: dict, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    charts = []
    charts += create_histograms(df, output_dir)
    heatmap = create_correlation_heatmap(df, output_dir)
    if heatmap:
        charts.append(heatmap)
    charts += create_bar_charts_from_groups(eda_results, output_dir)
    outlier_cols = [o["column"] for o in eda_results.get("outliers", [])]
    charts += create_box_plots(df, outlier_cols, output_dir)
    charts += create_pie_charts(df, output_dir)
    charts += create_scatter_plots(df, eda_results.get("correlation_matrix", {}), output_dir)
    ts_chart = create_time_series_chart(df, output_dir)
    if ts_chart:
        charts.append(ts_chart)
    for c in charts:
        c["source"] = "overview"
    return {"charts": charts, "total_charts": len(charts)}


def create_insight_driven_charts(df, insights, numeric_columns, categorical_columns, output_dir) -> list[dict]:
    """
    One chart per VALIDATED insight, matching chart type to category,
    using the exact columns that insight referenced — in the order they
    were mentioned in the text (see insight_tools.check_grounding), so
    X/Y axis assignment matches how a reader would interpret the finding.

    Chart titles stay SHORT ("X vs Y") rather than repeating the full
    insight sentence — the sentence is already shown as the card heading
    above the chart in the UI, so duplicating it inside the plot just
    causes truncation/overflow for longer findings.
    """
    numeric_set = set(numeric_columns)
    categorical_set = set(categorical_columns)
    importance_rank = {"high": 0, "medium": 1, "low": 2}
    candidates = [i for i in insights if i.get("validation", {}).get("is_valid")]
    candidates.sort(key=lambda i: importance_rank.get(i.get("importance", "low"), 3))

    charts = []
    used_chart_keys = set()

    for insight in candidates:
        if len(charts) >= MAX_INSIGHT_CHARTS:
            break
        referenced = insight.get("referenced_columns", [])
        category = insight.get("category", "general")
        # Preserve the order from referenced_columns (already text-order,
        # not set order) when filtering down to numeric/categorical.
        numeric_refs = [c for c in referenced if c in numeric_set]
        categorical_refs = [c for c in referenced if c in categorical_set]
        finding = insight.get("finding", "")
        chart = None

        if category == "correlation" and len(numeric_refs) >= 2:
            col_a, col_b = numeric_refs[0], numeric_refs[1]
            key = ("scatter", tuple(sorted([col_a, col_b])))
            if key not in used_chart_keys:
                fig = px.scatter(df, x=col_a, y=col_b, trendline="ols", title=f"{col_a} vs {col_b}",
                                 color_discrete_sequence=[COLOR_SEQUENCE[0]])
                fig.update_traces(marker=dict(size=9, opacity=0.7, line=dict(width=1, color="white")))
                fig.update_layout(xaxis_title=col_a, yaxis_title=col_b)
                _apply_theme(fig)
                path = _save(fig, output_dir, f"insight_scatter_{col_a}_vs_{col_b}.html")
                chart = {"type": "scatter", "columns": [col_a, col_b], "path": path, "height": DEFAULT_HEIGHT, "description": finding}
                used_chart_keys.add(key)

        elif category == "group_comparison" and categorical_refs and numeric_refs:
            cat_col, num_col = categorical_refs[0], numeric_refs[0]
            key = ("bar", cat_col, num_col)
            if key not in used_chart_keys:
                grouped = df.groupby(cat_col)[num_col].mean().round(4).sort_values(ascending=False)
                n_categories = len(grouped)
                fig = px.bar(x=grouped.index.tolist(), y=grouped.values.tolist(), title=f"{cat_col} vs {num_col}",
                            labels={"x": cat_col, "y": f"Average {num_col}"}, color=grouped.index.tolist(),
                            color_discrete_sequence=COLOR_SEQUENCE, text=[f"{v:,.1f}" for v in grouped.values])
                fig.update_traces(textposition="outside", marker_line_color="rgba(0,0,0,0.3)", marker_line_width=1.2)
                fig.update_layout(showlegend=False, xaxis_title=cat_col, yaxis_title=f"Average {num_col}")
                _rotate_ticks_if_crowded(fig, n_categories)
                height = DEFAULT_HEIGHT + (80 if n_categories > TICK_ROTATION_THRESHOLD else 0)
                _apply_theme(fig, height=height)
                path = _save(fig, output_dir, f"insight_bar_{num_col}_by_{cat_col}.html")
                chart = {"type": "bar", "columns": [cat_col, num_col], "path": path, "height": height, "description": finding}
                used_chart_keys.add(key)

        elif category == "outlier" and numeric_refs:
            num_col = numeric_refs[0]
            key = ("box", num_col)
            if key not in used_chart_keys:
                fig = px.box(df, y=num_col, title=f"{num_col} — Outliers", points="all", color_discrete_sequence=[COLOR_SEQUENCE[2]])
                fig.update_traces(marker=dict(size=5, opacity=0.6), jitter=0.3)
                fig.update_layout(yaxis_title=num_col)
                _apply_theme(fig)
                path = _save(fig, output_dir, f"insight_box_{num_col}.html")
                chart = {"type": "box", "column": num_col, "path": path, "height": DEFAULT_HEIGHT, "description": finding}
                used_chart_keys.add(key)

        elif category == "distribution" and numeric_refs:
            num_col = numeric_refs[0]
            key = ("histogram", num_col)
            if key not in used_chart_keys:
                fig = px.histogram(df, x=num_col, title=f"Distribution of {num_col}", marginal="box",
                                   color_discrete_sequence=[COLOR_SEQUENCE[1]])
                fig.update_traces(marker_line_color="white", marker_line_width=1)
                fig.update_layout(xaxis_title=num_col, yaxis_title="Count")
                _apply_theme(fig)
                path = _save(fig, output_dir, f"insight_histogram_{num_col}.html")
                chart = {"type": "histogram", "column": num_col, "path": path, "height": DEFAULT_HEIGHT, "description": finding}
                used_chart_keys.add(key)

        if chart:
            chart["source"] = "insight"
            chart["importance"] = insight.get("importance", "low")
            charts.append(chart)

    return charts