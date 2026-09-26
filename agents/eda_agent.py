"""
EDA Agent — Step 3 of the pipeline.

Runs statistical analysis on the CLEANED dataset: averages, correlations,
distributions, comparisons between groups. Everything here is calculated
with Pandas/NumPy — no AI is involved in producing these numbers, so
they're exact and can be trusted by later agents.
"""



from tools.stats_tools import run_eda


def eda_agent_node(state: dict) -> dict:
    try:
        cleaned_df = state.get("cleaned_df")
        if cleaned_df is None:
            raise ValueError("No cleaned_df found in state — Cleaning Agent must run first")

        state["eda_results"] = run_eda(cleaned_df)
        state["current_agent"] = "eda_agent"

    except Exception as e:
        state.setdefault("errors", []).append(f"EDAAgent error: {e}")

    return state