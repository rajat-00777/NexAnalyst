"""
Report Agent — Step 8, the final step of the pipeline.

Collects everything the earlier agents produced — the data profile,
cleaning summary, validated insights, recommendations, and charts — and
assembles it all into one final report.

No AI is called here. This step only formats and organizes information
that was already generated and checked earlier, which is what keeps the
final report as trustworthy as the steps that built it.
"""



import os
from tools.report_tools import assemble_report, save_report

REPORTS_OUTPUT_DIR = os.path.join("outputs", "reports")


def report_agent_node(state: dict) -> dict:
    try:
        report_text = assemble_report(state)
        report_path = save_report(report_text, REPORTS_OUTPUT_DIR)

        state["final_report"] = report_text
        state["report_path"] = report_path
        state["current_agent"] = "report_agent"

    except Exception as e:
        state.setdefault("errors", []).append(f"ReportAgent error: {e}")

    return state