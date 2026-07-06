import math


def _bounded(value):
    return round(max(0.0, min(float(value), 100.0)), 1)


class MetricsEngine:
    """Translate trained/static signals and calibrated forecasts into UI metrics."""

    def build_metrics(self, cognition_packet, semantic_profile=None, context=None):
        context = context or {}
        cluster = cognition_packet.get("cluster", {})
        forecast = cognition_packet.get("forecast", {})
        temporal = cognition_packet.get("temporal_state", [])
        environment = (
            context.get("osm_context", {}).get("urban_environment", {})
        )

        static_crime = float(cluster.get("static_crime_score", 0) or 0)
        static_activity = float(cluster.get("static_activity_score", 0) or 0)
        crime_total = max(float(cluster.get("crime_total", 0) or 0), 1.0)
        property_share = float(cluster.get("property_crime_count", 0) or 0) / crime_total
        violent_share = float(cluster.get("violent_crime_count", 0) or 0) / crime_total
        request_pressure = min(
            math.log1p(float(cluster.get("requests_311_total", 0) or 0)) * 12.0,
            100.0,
        )
        probability_any = float(forecast.get("probability_any_crime_24h", 0) or 0)

        theft_risk = _bounded(static_crime * (0.55 + property_share))
        violence_risk = _bounded(static_crime * (0.45 + 1.5 * violent_share))
        civil_stress = _bounded(request_pressure * 0.65 + static_activity * 0.35)
        forecast_pressure = probability_any * 100.0
        instability = (
            static_crime * 0.45 + theft_risk * 0.15 + violence_risk * 0.20
            + civil_stress * 0.10 + forecast_pressure * 0.10
        )
        safety_score = _bounded(100.0 - instability)

        def env_score(name):
            return float(environment.get(name, {}).get("score", 0) or 0)

        nightlife = env_score("nightlife_semantic") or env_score("nightlife_density")
        transit = env_score("transit_corridor_activity") or env_score("transit_intensity")
        vitality = env_score("urban_vitality")
        workplace = env_score("workplace_activity")

        crime_timeline = forecast.get("static_hourly_distribution", [])
        if not crime_timeline:
            crime_counts = [float(row.get("crime_count", 0) or 0) for row in temporal]
            mean_count = sum(crime_counts) / max(len(crime_counts), 1)
            crime_timeline = [
                {
                    "hour": int(row.get("hour", index)),
                    "value": round(
                        min(100.0, 50.0 * float(row.get("crime_count", 0) or 0) / mean_count)
                        if mean_count else 0.0,
                        1,
                    ),
                }
                for index, row in enumerate(temporal)
            ]
        forecast_timeline = forecast.get("hourly_distribution", [])

        metrics = {
            "safety_score": {"value": safety_score},
            "theft_risk": {"percentage": theft_risk},
            "violence_risk": {"percentage": violence_risk},
            "nightlife_activity": {"percentage": _bounded(nightlife)},
            "transit_access": {"percentage": _bounded(transit)},
            "transit_intensity": {"value": _bounded(transit)},
            "civil_stress": {"percentage": civil_stress},
            "congestion": {"percentage": _bounded(env_score("transit_intensity"))},
            "crime_timeline": {"hourly_distribution": crime_timeline},
            "forecast_timeline": {"hourly_distribution": forecast_timeline},
            "forecast": {
                "predicted_crime": round(float(forecast.get("expected_crime_count_24h", 0) or 0), 4),
                "static_baseline_crime": round(float(forecast.get("static_baseline_count_24h", 0) or 0), 4),
                "predicted_crime_label": "AI forecast",
                "static_baseline_crime_label": "Static baseline",
                "probability_any_crime": round(probability_any * 100.0, 2),
                "confidence": round(probability_any * 100.0, 2),
                "confidence_definition": forecast.get("confidence_definition", ""),
                "reference_time": forecast.get("reference_time"),
                "forecast_status": forecast.get("forecast_status"),
                "model": forecast.get("model"),
                "source_last_updated_label": "Official crime feed last updated",
                "static_average_hourly_count": forecast.get("static_average_hourly_count"),
                "calibration_source": forecast.get("calibration_source"),
                "calibration_definition": forecast.get("calibration_definition"),
                "source_freshness": forecast.get("source_freshness", {}),
                "hourly_forecast": forecast.get("hourly_forecast", []),
            },
            "urban_personality": {
                "value": cluster.get("urban_profile", "Unknown")
            },
            "urban_vitality": {"percentage": _bounded(vitality)},
            "workplace_activity": {"percentage": _bounded(workplace)},
        }
        if semantic_profile:
            metrics["urban_personality"]["value"] = semantic_profile.get(
                "summary", {}
            ).get("urban_profile", metrics["urban_personality"]["value"])
        return metrics
