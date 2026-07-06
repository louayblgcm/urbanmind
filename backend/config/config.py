# =====================================================
# CONFIG - URBANMIND V2
# =====================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Always load the backend environment, regardless of the process working directory.
BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

# =========================
# DATABASE CONFIG
# =========================

DB_NAME = os.getenv("DB_NAME", "urbanmindv2")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# =========================
# API CONFIG
# =========================

CRIME_API_URL = os.getenv("CRIME_API_URL", "")
REQUESTS_311_API_URL = os.getenv("REQUESTS_311_API_URL", "")
TRAFFIC_API_URL = os.getenv("TRAFFIC_API_URL", "")
BUSINESS_API_URL = os.getenv("BUSINESS_API_URL", "")
PERMITS_API_URL = os.getenv("PERMITS_API_URL", "")
NOMINATIM_REVERSE_URL = os.getenv("NOMINATIM_REVERSE_URL", "")
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "UrbanMindV2/1.0")
SOCRATA_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = os.getenv("GROQ_API_URL", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

# =========================
# FETCH SETTINGS
# =========================

PAGE_SIZE = int(os.getenv("PAGE_SIZE", "500000"))
