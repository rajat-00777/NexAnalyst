"""
Recommendation Agent — Step 6 of the pipeline.

Takes only the insights that passed validation (the ones proven correct)
and turns them into practical suggestions. Every recommendation clearly
separates two things: the FACT (confirmed from the data) and the
SUGGESTION (the AI's opinion on what to do about it) — so it's always
obvious which part is proven and which part is advice.
"""



from tools.recommendation_tools import (
    build_recommendation_prompt,
    parse_recommendations_response,
    check_recommendation_grounding,
)
from utils.llm_client import get_llm


def recommendation_agent_node(state: dict) -> dict:
    try:
        insights = state.get("insights", [])
        validated_insights = [i for i in insights if i.get("validation", {}).get("is_valid")]

        if not validated_insights:
            state["recommendations"] = []
            state["current_agent"] = "recommendation_agent"
            return state

        prompt = build_recommendation_prompt(validated_insights)
        llm = get_llm(temperature=0.3)
        response = llm.invoke(prompt)
        raw_text = response.content

        recommendations = parse_recommendations_response(raw_text)
        recommendations = check_recommendation_grounding(recommendations, validated_insights)

        state["recommendations"] = recommendations
        state["current_agent"] = "recommendation_agent"

    except Exception as e:
        state.setdefault("errors", []).append(f"RecommendationAgent error: {e}")

    return state