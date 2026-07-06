class SemanticEngine:
    def build_semantic_profile(self, metrics, cognition_packet=None):
        # Accept the old packet shape too, but prefer already-computed metrics.
        if "safety_score" not in metrics:
            cognition_packet = metrics
            metrics = cognition_packet.get("metrics", {})

        theft = metrics.get("theft_risk", {}).get("percentage", 0)
        violence = metrics.get("violence_risk", {}).get("percentage", 0)
        nightlife = metrics.get("nightlife_activity", {}).get("percentage", 0)
        transit = metrics.get("transit_access", {}).get("percentage", 0)
        civil = metrics.get("civil_stress", {}).get("percentage", 0)
        vitality = metrics.get("urban_vitality", {}).get("percentage", 0)
        workplace = metrics.get("workplace_activity", {}).get("percentage", 0)
        base_profile = metrics.get("urban_personality", {}).get("value", "Balanced Urban Zone")

        tags = []
        if nightlife > 65:
            tags.append("nightlife-heavy")
        if transit > 60:
            tags.append("transit-connected")
        if workplace > 65:
            tags.append("business-district")
        if vitality > 70:
            tags.append("high-urban-vitality")
        if violence > 65:
            tags.append("violent-crime-pressure")
        if theft > 65:
            tags.append("property-crime-pressure")
        if civil > 60:
            tags.append("urban-friction")

        if nightlife < 20:
            nightlife_text = "Nighttime urban activity is relatively limited."
        elif transit > 65:
            nightlife_text = "Nighttime movement is supported by transit and pedestrian activity."
        else:
            nightlife_text = "The area has moderate nighttime activity."

        combined_risk = theft * 0.4 + violence * 0.4 + civil * 0.2
        risk_label = (
            "High Urban Risk" if combined_risk > 70
            else "Moderate Urban Risk" if combined_risk > 45
            else "Relatively Stable"
        )
        return {
            "summary": {
                "nightlife_interpretation": nightlife_text,
                "urban_profile": base_profile,
                "risk_label": risk_label,
                "tags": tags,
            }
        }
