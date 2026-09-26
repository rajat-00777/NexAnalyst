"""
Visualization Agent — Step 7 of the pipeline.

Builds two sets of charts:
1. "Key Findings" charts — built directly from the validated insights,
   so the most important charts match what was actually proven true.
2. "Overview" charts — general statistical charts for broader context.

Runs after the Insight and Validation agents on purpose, so the charts
can be based on findings that have already been checked for accuracy.
"""


import os
from tools.chart_tools import generate_visualizations, create_insight_driven_charts

CHARTS_OUTPUT_DIR = os.path.join("outputs", "charts")


def visualization_agent_node(state: dict) -> dict:
    try:
        cleaned_df = state.get("cleaned_df")
        eda_results = state.get("eda_results", {})
        insights = state.get("insights", [])
        data_profile = state.get("data_profile", {})

        if cleaned_df is None:
            raise ValueError("No cleaned_df found in state — Cleaning Agent must run first")

        overview_result = generate_visualizations(cleaned_df, eda_results, CHARTS_OUTPUT_DIR)
        insight_charts = create_insight_driven_charts(
            cleaned_df, insights, data_profile.get("numeric_columns", []),
            data_profile.get("categorical_columns", []), CHARTS_OUTPUT_DIR,
        )

        all_charts = insight_charts + overview_result["charts"]
        state["visualization_results"] = {
            "charts": all_charts, "total_charts": len(all_charts),
            "insight_chart_count": len(insight_charts), "overview_chart_count": len(overview_result["charts"]),
        }
        state["current_agent"] = "visualization_agent"

    except Exception as e:
        state.setdefault("errors", []).append(f"VisualizationAgent error: {e}")

    return state