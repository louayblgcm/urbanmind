from scripts.cognition.cognition_retriever import CognitionRetriever
from scripts.context.osm_context import get_osm_context
from scripts.context.reverse_geocoder import reverse_geocode


class ContextBuilder:
    def __init__(self, retriever=None):
        self.retriever = retriever or CognitionRetriever()

    def build_context(self, lat, lon, cognition_packet=None):
        packet = cognition_packet or self.retriever.build_cognition_packet(lat, lon)
        return {
            "location_name": reverse_geocode(lat, lon),
            "cell_id": packet["cell_id"],
            "recent_crimes": packet.get("recent_crimes", []),
            "recent_311": packet.get("recent_311", []),
            "temporal_state": packet.get("temporal_state", []),
            "forecast": packet.get("forecast", {}),
            "osm_context": get_osm_context(lat, lon),
        }
