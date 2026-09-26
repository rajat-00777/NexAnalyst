"""
Cleaning Agent — Step 2 of the pipeline.

Takes the raw dataset plus the profile from the Data Agent, and fixes
common data problems: duplicate rows, missing values, inconsistent text,
wrong data types.

Important: the original uploaded file is never touched. All cleaning
happens on a copy in memory. The cleaned version is saved separately in
shared state so every agent after this one (EDA, Insights, etc.) works
from clean data without affecting the original.
"""



from utils.data_loader import load_dataset
from tools.cleaning_tools import clean_dataset


def cleaning_agent_node(state: dict) -> dict:
    try:
        df = load_dataset(state["dataset_path"])

        profile = state.get("data_profile", {})
        datetime_like_cols = profile.get("datetime_like_columns", [])

        result = clean_dataset(df, datetime_like_cols=datetime_like_cols)

        cleaned_df = result.pop("cleaned_df")
        state["cleaned_df"] = cleaned_df
        state["cleaning_results"] = result
        state["current_agent"] = "cleaning_agent"

    except Exception as e:
        state.setdefault("errors", []).append(f"CleaningAgent error: {e}")

    return state
