"""
Phase 5 tests. Prompt building, response parsing, and grounding checks are
pure functions and fully tested here without any live LLM call. The actual
API call (insight_agent_node with a real Groq key) is NOT tested here —
that must be verified manually with a real .env key.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from tools.insight_tools import (
    build_insight_prompt,
    parse_insights_response,
    check_grounding,
)


SAMPLE_DATA_PROFILE = {
    "n_rows": 20,
    "n_cols": 9,
    "columns": ["order_id", "product", "region", "revenue", "quantity"],
}

SAMPLE_CLEANING_RESULTS = {"total_operations": 3}

SAMPLE_EDA_RESULTS = {
    "descriptive_stats": {
        "revenue": {"mean": 100.5, "median": 90.0, "std": 45.2, "min": 25.5, "max": 240.0},
    },
    "correlation_matrix": {
        "quantity": {"quantity": 1.0, "revenue": 0.85},
        "revenue": {"quantity": 0.85, "revenue": 1.0},
    },
    "categorical_summary": {
        "region": {"most_frequent": "North", "top_values": {"North": 6, "South": 5}},
    },
    "distribution_shape": {
        "revenue": {"shape": "moderately right-skewed", "skewness": 0.6},
    },
    "group_comparisons": {
        "region": {"revenue": {"North": 150.0, "South": 90.0}},
    },
    "outliers": [
        {"column": "revenue", "count": 1, "bounds": {"lower": 10.0, "upper": 200.0}},
    ],
}


def test_build_insight_prompt_includes_real_numbers():
    prompt = build_insight_prompt(SAMPLE_DATA_PROFILE, SAMPLE_CLEANING_RESULTS, SAMPLE_EDA_RESULTS)
    assert "100.5" in prompt  # actual mean
    assert "0.85" in prompt   # actual correlation
    assert "revenue" in prompt
    assert "20 rows" in prompt


def test_build_insight_prompt_instructs_no_invention():
    prompt = build_insight_prompt(SAMPLE_DATA_PROFILE, SAMPLE_CLEANING_RESULTS, SAMPLE_EDA_RESULTS)
    assert "never invent" in prompt.lower() or "only use" in prompt.lower()


def test_parse_insights_response_clean_json():
    raw = json.dumps([
        {"finding": "Revenue correlates with quantity", "category": "correlation",
         "supporting_data": "correlation 0.85", "importance": "high"}
    ])
    insights = parse_insights_response(raw)
    assert len(insights) == 1
    assert insights[0]["finding"] == "Revenue correlates with quantity"


def test_parse_insights_response_strips_markdown_fences():
    raw = '```json\n[{"finding": "x", "category": "general", "supporting_data": "y", "importance": "low"}]\n```'
    insights = parse_insights_response(raw)
    assert len(insights) == 1


def test_parse_insights_response_rejects_missing_fields():
    raw = json.dumps([{"finding": "x"}])  # missing category, supporting_data, importance
    try:
        parse_insights_response(raw)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "missing fields" in str(e)


def test_parse_insights_response_rejects_invalid_json():
    raw = "This is not JSON at all"
    try:
        parse_insights_response(raw)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Could not parse" in str(e)


def test_parse_insights_response_caps_at_max():
    raw = json.dumps([
        {"finding": f"insight {i}", "category": "general", "supporting_data": "x", "importance": "low"}
        for i in range(20)
    ])
    insights = parse_insights_response(raw)
    assert len(insights) == 8  # MAX_INSIGHTS


def test_check_grounding_flags_real_column_reference():
    insights = [
        {"finding": "revenue is high", "category": "general", "supporting_data": "revenue avg 100.5", "importance": "high"}
    ]
    result = check_grounding(insights, SAMPLE_DATA_PROFILE)
    assert result[0]["grounding_check"] == "ok"
    assert "revenue" in result[0]["referenced_columns"]


def test_check_grounding_flags_missing_column_reference():
    insights = [
        {"finding": "something vague happened", "category": "general", "supporting_data": "no real column here", "importance": "low"}
    ]
    result = check_grounding(insights, SAMPLE_DATA_PROFILE)
    assert result[0]["grounding_check"] == "no_column_reference_found"