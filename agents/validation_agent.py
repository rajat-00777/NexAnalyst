"""
Validation Agent — Step 5 of the pipeline.

Validation Agent — programmatically fact-checks the Insight Agent's output
against the real EDA numbers. No LLM call here; this is deterministic
verification, which is exactly the point (it can't be fooled the same way
another LLM call could be).
"""



from tools.validation_tools import validate_insights, MAX_RETRIES


def validation_agent_node(state: dict) -> dict:
    try:
        insights = state.get("insights", [])
        eda_results = state.get("eda_results", {})

        if not insights:
            raise ValueError("No insights found in state — Insight Agent must run first")

        result = validate_insights(insights, eda_results)

        state["insights"] = result["validated_insights"]
        state["validation_results"] = {
            "total_insights": result["total_insights"],
            "valid_count": result["valid_count"],
            "invalid_count": result["invalid_count"],
            "failure_rate": result["failure_rate"],
            "needs_retry": result["needs_retry"],
        }
        state["current_agent"] = "validation_agent"

        if result["needs_retry"]:
            state["retry_count"] = state.get("retry_count", 0) + 1

    except Exception as e:
        state.setdefault("errors", []).append(f"ValidationAgent error: {e}")

    return state


def should_retry_insights(state: dict) -> str:
    """
    Conditional edge function for the graph: decides whether to loop back
    to the Insight Agent for regeneration, or continue forward. Bounded by
    MAX_RETRIES so a persistently-failing LLM response can't loop forever.
    """
    validation = state.get("validation_results", {})
    retry_count = state.get("retry_count", 0)

    if validation.get("needs_retry") and retry_count <= MAX_RETRIES:
        return "retry"
    return "continue"