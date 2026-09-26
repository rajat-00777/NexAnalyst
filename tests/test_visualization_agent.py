"""
Phase 4 tests. Chart generation writes real files to a temp directory,
so these tests check file creation + metadata structure, not visual content.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import tempfile
import pandas as pd
from tools.chart_tools import (
    create_histograms,
    create_correlation_heatmap,
    create_bar_charts_from_groups,
    create_box_plots,
    create_pie_charts,
    create_scatter_plots,
    create_time_series_chart,
    generate_visualizations,
)


def _tmp_dir():
    d = tempfile.mkdtemp()
    return d


def test_create_histograms_excludes_id_column():
    df = pd.DataFrame({
        "order_id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
                     1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020],
        "revenue": [i % 8 for i in range(20)],  # realistic — has repeats
    })
    out = _tmp_dir()
    charts = create_histograms(df, out)
    columns_charted = [c["column"] for c in charts]
    assert "order_id" not in columns_charted
    assert "revenue" in columns_charted
    shutil.rmtree(out)


def test_create_histograms_creates_real_files():
    df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
    out = _tmp_dir()
    charts = create_histograms(df, out)
    assert len(charts) == 1
    assert os.path.exists(charts[0]["path"])
    shutil.rmtree(out)


def test_correlation_heatmap_needs_two_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "name": ["x", "y", "z"]})
    out = _tmp_dir()
    result = create_correlation_heatmap(df, out)
    assert result is None
    shutil.rmtree(out)


def test_correlation_heatmap_created_with_two_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    out = _tmp_dir()
    result = create_correlation_heatmap(df, out)
    assert result is not None
    assert os.path.exists(result["path"])
    shutil.rmtree(out)


def test_bar_charts_from_groups():
    eda_results = {
        "group_comparisons": {
            "region": {"revenue": {"North": 100.0, "South": 200.0}}
        },
        "descriptive_stats": {
            "revenue": {"std": 50.0}
        },
    }
    out = _tmp_dir()
    charts = create_bar_charts_from_groups(eda_results, out)
    assert len(charts) == 1
    assert charts[0]["type"] == "bar"
    shutil.rmtree(out)


def test_bar_charts_prefer_diverse_metrics():
    """With multiple categorical columns available, different numeric
    metrics should be represented rather than the same one repeated."""
    eda_results = {
        "group_comparisons": {
            "region": {
                "unlocks": {"North": 50.0, "South": 250.0},
                "score": {"North": 6.0, "South": 8.0},
            },
            "platform": {
                "unlocks": {"A": 60.0, "B": 240.0},
                "score": {"A": 5.5, "B": 8.5},
            },
        },
        "descriptive_stats": {
            "unlocks": {"std": 100.0},
            "score": {"std": 1.0},
        },
    }
    out = _tmp_dir()
    charts = create_bar_charts_from_groups(eda_results, out)
    metrics_charted = {c["columns"][1] for c in charts}
    assert "score" in metrics_charted
    shutil.rmtree(out)


def test_pie_charts_skips_high_cardinality():
    df = pd.DataFrame({"name": [f"person_{i}" for i in range(20)]})
    out = _tmp_dir()
    charts = create_pie_charts(df, out)
    assert len(charts) == 0  # 20 unique values > PIE_MAX_CATEGORIES
    shutil.rmtree(out)


def test_pie_charts_created_for_low_cardinality():
    df = pd.DataFrame({"status": ["Active", "Active", "Inactive", "Active"]})
    out = _tmp_dir()
    charts = create_pie_charts(df, out)
    assert len(charts) == 1
    shutil.rmtree(out)


def test_scatter_plots_only_for_strong_correlation():
    correlation_matrix = {
        "a": {"a": 1.0, "b": 0.1},
        "b": {"a": 0.1, "b": 1.0},
    }
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 1, 2]})
    out = _tmp_dir()
    charts = create_scatter_plots(df, correlation_matrix, out)
    assert len(charts) == 0  # correlation 0.1 is below threshold
    shutil.rmtree(out)


def test_scatter_plots_created_for_strong_correlation():
    correlation_matrix = {
        "a": {"a": 1.0, "b": 0.9},
        "b": {"a": 0.9, "b": 1.0},
    }
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
    out = _tmp_dir()
    charts = create_scatter_plots(df, correlation_matrix, out)
    assert len(charts) == 1
    shutil.rmtree(out)


def test_time_series_chart_requires_datetime_column():
    df = pd.DataFrame({"revenue": [1, 2, 3]})
    out = _tmp_dir()
    result = create_time_series_chart(df, out)
    assert result is None
    shutil.rmtree(out)


def test_time_series_chart_excludes_id_column():
    df = pd.DataFrame({
        "order_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 7),
        "order_id": list(range(1000, 1021)),
        "revenue": [i % 8 for i in range(21)],  # realistic — has repeats
    })
    out = _tmp_dir()
    result = create_time_series_chart(df, out)
    assert result is not None
    assert "order_id" not in result["columns"]
    assert "revenue" in result["columns"]
    shutil.rmtree(out)


def test_generate_visualizations_full_pipeline_on_sample_csv():
    from tools.cleaning_tools import clean_dataset
    from tools.stats_tools import run_eda

    path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_sales.csv")
    df = pd.read_csv(path)
    cleaned = clean_dataset(df, datetime_like_cols=["order_date"])["cleaned_df"]
    eda = run_eda(cleaned)

    out = _tmp_dir()
    result = generate_visualizations(cleaned, eda, out)

    assert result["total_charts"] > 0
    for chart in result["charts"]:
        assert os.path.exists(chart["path"])
        assert "order_id" not in str(chart.get("column", "")) 
        assert "order_id" not in chart.get("columns", [])

    shutil.rmtree(out)