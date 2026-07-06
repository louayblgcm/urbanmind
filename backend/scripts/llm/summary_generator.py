from scripts.llm.groq_client import generate_groq_response


def generate_ai_snapshot(area_data):
    urban = area_data["urban_signals"]
    contextual = area_data["contextual_signals"]
    comparison = area_data["city_comparison"]
    prompt = f"""
Write two short urban intelligence summaries: first crime and safety, then
services, traffic, events, nightlife, and environment. Each must be at most
50 words. Use only the supplied values.

Violence: {urban['violent_activity']['score']}
Theft: {urban['theft_activity']['score']}
Vehicle crime: {urban['vehicle_activity']['score']}
Night activity: {urban['night_activity']['score']}
Parking safety: {urban['parking_safety']['score']}
City comparison: {comparison['percent_of_average']}% of city average
Traffic density: {contextual['traffic_density']['score']}
Event intensity: {contextual['event_intensity']['score']}
Nightlife density: {contextual['nightlife_density']['score']}
Commercial density: {contextual['commercial_density']['score']}
"""
    response = generate_groq_response(prompt)
    sections = [section.strip() for section in response.splitlines() if section.strip()]
    return {
        "crime_overview": sections[0] if sections else "",
        "service_overview": sections[1] if len(sections) > 1 else "",
    }
