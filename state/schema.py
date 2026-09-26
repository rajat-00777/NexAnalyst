"""
This defines the shared "notebook" that every agent reads from and
writes to as the pipeline runs.

Each agent adds its own results here (the data profile, cleaning summary,
insights, etc.), so the next agent in line always has access to
everything produced so far.
"""


from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict, total=False):
    # Input
    user_query: Optional[str]
    dataset_path: str

    # Populated by agents as the graph runs
    data_profile: Dict[str, Any]
    cleaned_df: Any                                # pandas DataFrame after Cleaning Agent runs
    cleaning_results: Dict[str, Any]
    eda_results: Dict[str, Any]
    visualization_results: Dict[str, Any]
    insights: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    final_report: str
    report_path: str

    # Control / bookkeeping
    errors: List[str]
    current_agent: str
    retry_count: int