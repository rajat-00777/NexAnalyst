"""
This is the web app (built with Streamlit) that ties everything together.

When you upload a file and click "Start Analysis," it runs the full
pipeline in order: Data → Cleaning → EDA → Insight → Validation →
Recommendation → Visualization → Report — then displays the results.
"""

import streamlit as st
import pandas as pd
import tempfile
import os

from graph.workflow import build_graph


def _column_role(col: str, profile: dict) -> str:
    if col in profile.get("datetime_like_columns", []):
        return "Date/Time"
    if col in profile.get("possible_id_columns", []):
        return "Identifier"
    if col in profile.get("numeric_columns", []):
        return "Numeric"
    if col in profile.get("categorical_columns", []):
        return "Categorical"
    return "Other"


st.set_page_config(page_title="NexAnalyst", page_icon="📊", layout="wide")
st.title("📈 NexAnalyst")
st.caption("Your autonomous AI data analyst: upload a dataset and get clean data, verified insights, charts, and a full report.")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.subheader("Preview (original, uncleaned)")
    preview_df = pd.read_csv(tmp_path) if suffix == ".csv" else pd.read_excel(tmp_path)
    st.dataframe(preview_df.head(10))

    if st.button("Start Analysis"):
        STEP_LABELS = {
            "main_agent": "Initializing pipeline",
            "data_agent": "📊 Profiling dataset",
            "cleaning_agent": "🧹 Cleaning data",
            "eda_agent": "📈 Running statistical analysis",
            "insight_agent": "💡 Generating insights (AI)",
            "validation_agent": "✅ Fact-checking insights",
            "recommendation_agent": "💡 Generating recommendations (AI)",
            "visualization_agent": "🖼️ Creating charts",
            "report_agent": "📄 Assembling report",
        }

        result = {}
        with st.status("Running analysis...", expanded=True) as status:
            graph = build_graph()
            for step_output in graph.stream({"dataset_path": tmp_path}, stream_mode="updates"):
                for node_name, node_state in step_output.items():
                    label = STEP_LABELS.get(node_name, node_name)
                    st.write(f"✅ {label}")
                    result = node_state  # each update is the full accumulated state so far

            if result.get("errors"):
                status.update(label="Finished with errors", state="error")
            else:
                status.update(label="Analysis complete!", state="complete")

        if result.get("errors"):
            st.error("**The analysis hit the following problem(s):**")
            for err in result["errors"]:
                st.error(err)
        else:
            dp = result["data_profile"]
            validation = result.get("validation_results", {})
            viz = result.get("visualization_results", {})
            n_recs = len([r for r in result.get("recommendations", []) if r.get("grounding", {}).get("is_valid")])

            st.subheader("📋 At a Glance")
            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("Rows", dp.get("n_rows", "?"))
            g2.metric("Columns", dp.get("n_cols", "?"))
            g3.metric("Validated Insights", validation.get("valid_count", 0))
            g4.metric("Recommendations", n_recs)
            g5.metric("Charts Generated", viz.get("total_charts", 0))
            st.divider()

            tab1, tab2, tab3, tab5, tab6, tab4, tab7 = st.tabs(
                ["📊 Data Profile", "🧹 Cleaning", "📈 EDA", "💡 Insights", "✅ Recommendations", "🖼️ Charts", "📄 Report"]
            )

            with tab1:
                dp = result["data_profile"]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows", dp.get("n_rows", "?"))
                m2.metric("Columns", dp.get("n_cols", "?"))
                m3.metric("Duplicate rows", dp.get("duplicate_rows", 0))
                m4.metric("Columns with missing data", sum(1 for v in dp.get("missing_values", {}).values() if v > 0))

                st.write("**Column Summary**")
                rows = []
                for col in dp.get("columns", []):
                    rows.append({
                        "Column": col,
                        "Role": _column_role(col, dp),
                        "Data Type": dp.get("dtypes", {}).get(col, "?"),
                        "Missing": dp.get("missing_values", {}).get(col, 0),
                        "Missing %": f'{dp.get("missing_pct", {}).get(col, 0)}%',
                        "Unique Values": dp.get("unique_counts", {}).get(col, "?"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                true_id_columns = [c for c in dp.get("possible_id_columns", [])
                                   if c not in dp.get("datetime_like_columns", [])]
                if true_id_columns:
                    st.caption(f"🔑 Identifier-like column(s) detected: {', '.join(true_id_columns)} "
                               f"(excluded from statistical analysis since they're not meaningful metrics)")

            with tab2:
                cr = result["cleaning_results"]
                m1, m2, m3 = st.columns(3)
                m1.metric("Total operations", cr["total_operations"])
                m2.metric("Rows before → after", f'{cr["rows_before"]} → {cr["rows_after"]}')
                m3.metric("Outlier columns flagged", len(cr["outliers_detected"]))

                st.write("**Operations log:**")
                if cr["operations_log"]:
                    rows = []
                    for op in cr["operations_log"]:
                        detail = op.get("detail")
                        if not detail and "fill_value" in op:
                            detail = f'Filled with {op["fill_value"]}' + (f' ({op["strategy"]})' if "strategy" in op else "")
                        if not detail:
                            detail = "completed"
                        rows.append({"Operation": op["operation"].replace("_", " ").title(),
                                    "Column": op.get("column", "—"), "Detail": detail})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.write("_No cleaning operations were needed._")

                if cr["outliers_detected"]:
                    st.write("**⚠️ Outliers flagged (not modified):**")
                    rows = [{"Column": o["column"], "Outlier Count": o["count"],
                            "Normal Range": f'{o["bounds"]["lower"]} to {o["bounds"]["upper"]}'} for o in cr["outliers_detected"]]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.write("**Cleaned data preview:**")
                st.dataframe(result["cleaned_df"].head(10))

            with tab3:
                eda = result["eda_results"]

                st.write("**📐 Descriptive Statistics**")
                desc = eda.get("descriptive_stats", {})
                if desc:
                    desc_df = pd.DataFrame(desc).T
                    desc_df.index.name = "Column"
                    st.dataframe(desc_df, use_container_width=True)
                else:
                    st.write("_No numeric columns to summarize._")

                st.write("**🔗 Correlation Matrix**")
                if eda["correlation_matrix"]:
                    corr_df = pd.DataFrame(eda["correlation_matrix"])
                    st.dataframe(corr_df.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1))
                else:
                    st.write("_Not enough numeric columns for a correlation matrix._")

                st.write("**🏷️ Categorical Summary**")
                cat = eda.get("categorical_summary", {})
                if cat:
                    rows = []
                    for col, info in cat.items():
                        top_str = ", ".join(f"{k} ({v})" for k, v in info.get("top_values", {}).items())
                        rows.append({"Column": col, "Unique Values": info["unique_values"],
                                    "Most Frequent": info["most_frequent"], "Top Values": top_str})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.write("_No categorical columns found._")

                st.write("**📊 Distribution Shape**")
                dist = eda.get("distribution_shape", {})
                if dist:
                    rows = [{"Column": c, "Skewness": i["skewness"], "Shape": i["shape"]} for c, i in dist.items()]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.write("_No distribution data available._")

                st.write("**🔍 Group Comparisons (mean by category)**")
                groups = eda.get("group_comparisons", {})
                if groups:
                    for cat_col, numeric_dict in groups.items():
                        with st.expander(f"By {cat_col}"):
                            pivot_df = pd.DataFrame(numeric_dict)
                            pivot_df.index.name = cat_col
                            st.dataframe(pivot_df, use_container_width=True)
                else:
                    st.write("_No group comparisons available._")

                if eda["outliers"]:
                    st.write("**⚠️ Outliers (post-cleaning)**")
                    rows = [{"Column": o["column"], "Outlier Count": o["count"],
                            "Normal Range": f'{o["bounds"]["lower"]} to {o["bounds"]["upper"]}'} for o in eda["outliers"]]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with tab5:
                insights = result.get("insights", [])
                validation = result.get("validation_results", {})

                if not insights:
                    st.warning("No insights were generated — check for errors above.")
                else:
                    if validation:
                        v1, v2, v3 = st.columns(3)
                        v1.metric("Total insights", validation.get("total_insights", 0))
                        v2.metric("Passed validation", validation.get("valid_count", 0))
                        v3.metric("Failed validation", validation.get("invalid_count", 0))

                    importance_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    for insight in insights:
                        icon = importance_icon.get(insight.get("importance", "low"), "⚪")
                        val = insight.get("validation", {})
                        val_icon = "✅" if val.get("is_valid") else "❌"

                        with st.container(border=True):
                            st.markdown(f"{icon} {val_icon} **{insight['finding']}**")
                            st.caption(f"Category: {insight['category']} | "
                                       f"Importance: {insight['importance']} | "
                                       f"Grounding check: {insight.get('grounding_check', 'n/a')}")
                            st.write(f"_Supporting data: {insight['supporting_data']}_")
                            if val:
                                st.caption(f"Validation: {val.get('reason', 'n/a')}")

            with tab6:
                recommendations = result.get("recommendations", [])
                if not recommendations:
                    st.info("No recommendations were generated — this can happen if no insights passed validation.")
                else:
                    priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    for rec in recommendations:
                        icon = priority_icon.get(rec.get("priority", "low"), "⚪")
                        grounding = rec.get("grounding", {})
                        ground_icon = "✅" if grounding.get("is_valid") else "❌"

                        with st.container(border=True):
                            st.markdown(f"{icon} {ground_icon} **Suggestion:** {rec['suggestion']}")
                            st.caption(f"📊 Based on fact: _{rec['fact']}_")
                            st.caption(f"Priority: {rec['priority']} | Grounding: {grounding.get('reason', 'n/a')}")

            with tab4:
                viz = result["visualization_results"]
                st.write(f"**{viz['total_charts']} chart(s) generated** "
                         f"({viz['insight_chart_count']} key findings, {viz['overview_chart_count']} overview)")

                charts = viz["charts"]
                insight_charts = [c for c in charts if c["source"] == "insight"]
                overview_charts = [c for c in charts if c["source"] == "overview"]

                def render_chart_card(chart):
                    with st.container(border=True):
                        st.markdown(f"**{chart['description']}**")
                        with open(chart["path"], "r", encoding="utf-8") as f:
                            html_content = f.read()
                        display_height = chart.get("height", 500) + 40
                        st.components.v1.html(html_content, height=display_height, scrolling=False)

                def render_chart_grid(chart_list):
                    full_width_charts = [c for c in chart_list if c.get("full_width")]
                    paired_charts = [c for c in chart_list if not c.get("full_width")]

                    for chart in full_width_charts:
                        render_chart_card(chart)

                    for i in range(0, len(paired_charts), 2):
                        row_charts = paired_charts[i:i + 2]
                        cols = st.columns(len(row_charts))
                        for col, chart in zip(cols, row_charts):
                            with col:
                                render_chart_card(chart)

                if insight_charts:
                    st.subheader("🎯 Key Findings")
                    st.caption("Charts generated directly from validated insights — these show what the analysis determined actually matters.")
                    render_chart_grid(insight_charts)
                    st.divider()

                if overview_charts:
                    st.subheader("📋 Overview")
                    st.caption("General statistical charts for broader context.")
                    render_chart_grid(overview_charts)

            with tab7:
                report_text = result.get("final_report", "")
                if not report_text:
                    st.warning("No report was generated — check for errors above.")
                else:
                    st.download_button(
                        "⬇️ Download Report (Markdown)",
                        data=report_text,
                        file_name="analysis_report.md",
                        mime="text/markdown",
                    )
                    st.markdown(report_text)