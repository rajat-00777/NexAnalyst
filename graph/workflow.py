"""
This file wires all 8 agents together into one pipeline using LangGraph.

Order: Data → Cleaning → EDA → Insight → Validation → (if too many
insights fail validation, go back and try Insight again) → Recommendation
→ Visualization → Report → Done.
"""



from langgraph.graph import StateGraph, END
from state.schema import AgentState
from agents.data_agent import data_agent_node
from agents.cleaning_agent import cleaning_agent_node
from agents.eda_agent import eda_agent_node
from agents.visualization_agent import visualization_agent_node
from agents.insight_agent import insight_agent_node
from agents.validation_agent import validation_agent_node, should_retry_insights
from agents.recommendation_agent import recommendation_agent_node
from agents.report_agent import report_agent_node


def main_agent_node(state: dict) -> dict:
    state["current_agent"] = "main_agent"
    state.setdefault("errors", [])
    state.setdefault("retry_count", 0)
    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("main_agent", main_agent_node)
    graph.add_node("data_agent", data_agent_node)
    graph.add_node("cleaning_agent", cleaning_agent_node)
    graph.add_node("eda_agent", eda_agent_node)
    graph.add_node("insight_agent", insight_agent_node)
    graph.add_node("validation_agent", validation_agent_node)
    graph.add_node("recommendation_agent", recommendation_agent_node)
    graph.add_node("visualization_agent", visualization_agent_node)
    graph.add_node("report_agent", report_agent_node)

    graph.set_entry_point("main_agent")
    graph.add_edge("main_agent", "data_agent")
    graph.add_edge("data_agent", "cleaning_agent")
    graph.add_edge("cleaning_agent", "eda_agent")
    graph.add_edge("eda_agent", "insight_agent")
    graph.add_edge("insight_agent", "validation_agent")

    graph.add_conditional_edges(
        "validation_agent", should_retry_insights,
        {"retry": "insight_agent", "continue": "recommendation_agent"},
    )
    graph.add_edge("recommendation_agent", "visualization_agent")
    graph.add_edge("visualization_agent", "report_agent")
    graph.add_edge("report_agent", END)

    return graph.compile()