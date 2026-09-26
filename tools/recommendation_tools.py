"""
Pure functions supporting the Recommendation Agent: building the prompt,
parsing the LLM's response, and checking that each recommendation stays
grounded to the specific validated insight it claims to be based on.
"""

import json
import re

MAX_RECOMMENDATIONS = 6


def build_recommendation_prompt(validated_insights: list[dict]) -> str:
    """
    Only validated (is_valid=True) insights should be passed in — the
    caller filters before calling this. Each insight is given a numeric
    index the LLM must reference, so every recommendation can be traced
    back to a specific, fact-checked insight.
    """
    lines = []
    for i, insight in enumerate(validated_insights):
        lines.append(f"[{i}] {insight['finding']} (data: {insight['supporting_data']})")
    insights_block = "\n".join(lines)

    prompt = f"""You are a data analyst writing practical recommendations based ONLY on the validated findings below. These findings have already been fact-checked against the real data — you must not add any new numbers or statistics beyond what's already stated in them.

VALIDATED FINDINGS:
{insights_block}

TASK:
Write up to {MAX_RECOMMENDATIONS} practical recommendations. Each recommendation must be based on exactly ONE of the findings above (reference it by its index number). Clearly separate the FACT (restated from the finding, no new numbers) from your SUGGESTION (your practical advice — this is your judgment, not a data fact).

Respond with ONLY a JSON array, no other text, no markdown code fences. Each object must have exactly these fields:
- "based_on_insight_index": the integer index (from the [N] labels above) of the ONE finding this is based on
- "fact": one sentence restating the finding's data point, using ONLY numbers that already appear in that finding
- "suggestion": one or two sentences of practical, actionable advice based on that fact
- "priority": one of "high", "medium", "low"

Example format:
[{{"based_on_insight_index": 0, "fact": "...", "suggestion": "...", "priority": "high"}}]
"""
    return prompt


def parse_recommendations_response(raw_text: str) -> list[dict]:
    """Parses the LLM's response, stripping markdown fences if present."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        recommendations = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse LLM response as JSON: {e}\nRaw response: {raw_text[:500]}")

    if not isinstance(recommendations, list):
        raise ValueError("Expected a JSON array of recommendations")

    required_fields = {"based_on_insight_index", "fact", "suggestion", "priority"}
    for i, rec in enumerate(recommendations):
        if not isinstance(rec, dict):
            raise ValueError(f"Recommendation {i} is not an object")
        missing = required_fields - set(rec.keys())
        if missing:
            raise ValueError(f"Recommendation {i} is missing fields: {missing}")

    return recommendations[:MAX_RECOMMENDATIONS]


def _extract_numbers(text: str) -> list[float]:
    matches = re.findall(r"-?\d+\.?\d*", text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            continue
    return numbers


def check_recommendation_grounding(recommendations: list[dict], validated_insights: list[dict]) -> list[dict]:
    """
    For each recommendation:
    1. Verify based_on_insight_index actually points to a real insight.
    2. Verify every number in the "fact" field already appears in that
       specific insight's supporting_data — catches the recommendation
       inventing a new statistic not present in its cited source.
    Adds a "grounding" field to each recommendation with the result.
    """
    for rec in recommendations:
        idx = rec.get("based_on_insight_index")

        if not isinstance(idx, int) or idx < 0 or idx >= len(validated_insights):
            rec["grounding"] = {"is_valid": False, "reason": f"Invalid insight index: {idx}"}
            continue

        source_insight = validated_insights[idx]
        source_numbers = set(round(n, 2) for n in _extract_numbers(source_insight.get("supporting_data", "")))
        fact_numbers = _extract_numbers(rec.get("fact", ""))

        unmatched = [n for n in fact_numbers if round(n, 2) not in source_numbers]

        if unmatched:
            rec["grounding"] = {
                "is_valid": False,
                "reason": f"Fact cites number(s) not present in the source insight: {unmatched}",
            }
        else:
            rec["grounding"] = {
                "is_valid": True,
                "reason": "Fact matches the cited insight's data",
                "source_finding": source_insight.get("finding", ""),
            }

    return recommendations