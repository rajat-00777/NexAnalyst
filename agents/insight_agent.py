"""
Insight Agent — Step 4 of the pipeline, and the first one that uses AI.

Reads the statistics already calculated by the EDA Agent (never the raw
data directly) and asks the AI to describe what they mean in plain
English. The AI is instructed to only talk about numbers that were
actually calculated — not to guess or make up new ones.
"""



from tools.insight_tools import build_insight_prompt, parse_insights_response, check_grounding
from utils.llm_client import get_llm


def insight_agent_node(state: dict) -> dict:
    try:
        data_profile = state.get("data_profile", {})
        cleaning_results = state.get("cleaning_results", {})
        eda_results = state.get("eda_results", {})

        if not eda_results:
            raise ValueError("No eda_results found in state — EDA Agent must run first")

        prompt = build_insight_prompt(data_profile, cleaning_results, eda_results)

        llm = get_llm(temperature=0.2)
        response = llm.invoke(prompt)
        raw_text = response.content

        insights = parse_insights_response(raw_text)
        insights = check_grounding(insights, data_profile)

        state["insights"] = insights
        state["current_agent"] = "insight_agent"

    except Exception as e:
        state.setdefault("errors", []).append(f"InsightAgent error: {e}")

    return state