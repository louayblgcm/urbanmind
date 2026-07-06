import json


def _json(value):
    return json.dumps(value, default=str, ensure_ascii=False, indent=2)


def _round_float(value, digits=2):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _safe_percent(value):
    try:
        return round(float(value) * 100, 1)
    except (TypeError, ValueError):
        return None


def _friendly_forecast_summary(forecast, metrics):
    source_freshness = forecast.get("source_freshness", {}) or {}
    top_hours = []

    for row in forecast.get("hourly_forecast", []):
        probability = _safe_percent(row.get("occurrence_probability"))
        if probability is None or probability < 0.5:
            continue
        top_hours.append(
            {
                "hour": row.get("hour"),
                "probability_percent": probability,
            }
        )

    top_hours = sorted(
        top_hours,
        key=lambda row: row["probability_percent"],
        reverse=True,
    )[:3]

    return {
        "location_name": forecast.get("location_name"),
        "safety_score": _round_float(
            metrics.get("safety_score", {}).get("value"), 1
        ),
        "ai_forecast_24h": _round_float(
            forecast.get("expected_crime_count_24h"), 2
        ),
        "static_baseline_24h": _round_float(
            forecast.get("static_baseline_count_24h"), 2
        ),
        "chance_of_any_crime_percent": _safe_percent(
            forecast.get("probability_any_crime_24h")
        ),
        "top_risk_hours": top_hours,
        "crime_feed_age_hours": _round_float(
            source_freshness.get("crime_source_age_hours"), 1
        ),
        "forecast_status": forecast.get("forecast_status"),
    }


def _compact_forecast(forecast):
    hourly = forecast.get("hourly_forecast", [])
    meaningful_hours = [
        row for row in hourly
        if float(row.get("occurrence_probability", 0) or 0) >= 0.001
    ]
    highest_risk_hours = sorted(
        meaningful_hours,
        key=lambda row: float(row.get("occurrence_probability", 0) or 0),
        reverse=True,
    )[:5]
    return {
        "reference_time": forecast.get("reference_time"),
        "horizon_hours": forecast.get("horizon_hours", 24),
        "forecast_status": forecast.get("forecast_status"),
        "model": forecast.get("model"),
        "predicted_crime_label": "AI forecast",
        "expected_crime_count_24h": _round_float(
            forecast.get("expected_crime_count_24h"), 4
        ),
        "static_baseline_label": "Static baseline",
        "static_baseline_count_24h": _round_float(
            forecast.get("static_baseline_count_24h"), 4
        ),
        "probability_any_crime_24h_percent": _safe_percent(
            forecast.get("probability_any_crime_24h")
        ),
        "static_average_hourly_count": _round_float(
            forecast.get("static_average_hourly_count"), 4
        ),
        "source_last_updated_label": "Official crime feed last updated",
        "source_freshness": forecast.get("source_freshness", {}),
        "highest_meaningful_risk_hours": [
            {
                "forecast_timestamp": row.get("forecast_timestamp"),
                "hour": row.get("hour"),
                "occurrence_probability_percent": _safe_percent(
                    row.get("occurrence_probability")
                ),
                "relative_to_static_average": _round_float(
                    row.get("relative_to_static_average"), 2
                ),
            }
            for row in highest_risk_hours
        ],
    }


def _compact_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"crime_timeline", "forecast_timeline", "forecast"}
    }


class PromptBuilder:
    def build_overview_prompt(self, context, metrics, semantic_profile):
        customer_summary = _friendly_forecast_summary(
            context.get("forecast", {}),
            metrics,
        )
        customer_summary["location_name"] = context.get("location_name")

        compact_context = {
            "location_name": context.get("location_name"),
            "cell_id": context.get("cell_id"),
            "forecast": _compact_forecast(context.get("forecast", {})),
            "customer_summary": customer_summary,
            "urban_environment": context.get("osm_context", {}).get(
                "urban_environment", {}
            ),
            "recent_crimes": context.get("recent_crimes", [])[:5],
            "recent_311": context.get("recent_311", [])[:5],
        }
        return f"""
Write a concise, customer-facing overview of this area in two short paragraphs.

Rules:
- Sound like a polished product, not a technical report.
- Use plain language and practical interpretation.
- Prefer rounded values such as "about 15%" or "around 0.16".
- Do not dump raw decimals unless there is a strong reason.
- Avoid heavy wording like "according to the static baseline" more than once.
- Explain the forecast in human terms: quieter than usual, close to usual, or more active than usual.
- If the crime feed is stale, mention that gently as a freshness note, not as the main story.
- Never claim certainty.

Paragraph 1:
- Summarize safety and the next-24-hour outlook.
- Mention the AI forecast and, if useful, whether it sits below or above the usual baseline.
- Mention the most relevant risk window in natural language.

Paragraph 2:
- Summarize the area's character using urban activity, transit, services, and environment.
- Make it feel like a place description a user would understand quickly.

CONTEXT
{_json(compact_context)}

METRICS
{_json(_compact_metrics(metrics))}

SEMANTIC PROFILE
{_json(semantic_profile)}
"""

    def build_chat_prompt(self, question, context, metrics, semantic_profile):
        customer_summary = _friendly_forecast_summary(
            context.get("forecast", {}),
            metrics,
        )
        customer_summary["location_name"] = context.get("location_name")

        evidence = {
            "location_name": context.get("location_name"),
            "forecast": _compact_forecast(context.get("forecast", {})),
            "customer_summary": customer_summary,
            "recent_crimes": context.get("recent_crimes", [])[:10],
            "recent_311": context.get("recent_311", [])[:10],
            "urban_environment": context.get("osm_context", {}).get(
                "urban_environment", {}
            ),
            "metrics": _compact_metrics(metrics),
            "semantic_profile": semantic_profile,
        }
        return f"""
Answer the question in at most two short paragraphs using only the evidence.

Rules:
- Be customer-facing, clear, and natural.
- Prefer plain language over technical wording.
- Round numbers when possible.
- Treat probabilities as probabilities, not certainties.
- Compare the AI forecast with the usual baseline only when it helps answer the question.
- If the crime feed is stale, mention it briefly and plainly.
- If the evidence cannot answer the question, say so clearly.

QUESTION
{question}

EVIDENCE
{_json(evidence)}
"""
