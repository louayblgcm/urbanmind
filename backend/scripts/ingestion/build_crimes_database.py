import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.config import (
    CRIME_API_URL, DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER,
    SOCRATA_APP_TOKEN,
)
from scripts.ingestion.socrata_utils import date_windows, iso_literal, request_json
from utils.spatial import latlon_to_cell


PAGE_SIZE = int(os.getenv("CRIME_INGEST_PAGE_SIZE", "50000"))
DEFAULT_HISTORY_YEARS = float(os.getenv("INGEST_HISTORY_YEARS", "5"))
OVERLAP_DAYS = int(os.getenv("INGEST_OVERLAP_DAYS", "7"))
DEBUG_DIR = Path("backend") / "data" / "debug" / "ingestion"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

VIOLENT_CRIMES = {
    "ASSAULT", "BATTERY", "HOMICIDE", "KIDNAPPING",
    "CRIMINAL SEXUAL ASSAULT", "SEX OFFENSE",
}
PROPERTY_CRIMES = {"THEFT", "BURGLARY", "ROBBERY", "DECEPTIVE PRACTICE", "ARSON"}
VEHICLE_CRIMES = {"MOTOR VEHICLE THEFT"}
NARCOTICS_CRIMES = {"NARCOTICS"}
PUBLIC_DISORDER = {
    "CRIMINAL DAMAGE", "CRIMINAL TRESPASS", "PUBLIC PEACE VIOLATION",
    "WEAPONS VIOLATION", "INTIMIDATION",
}


def map_crime_category(primary_type):
    if primary_type in VIOLENT_CRIMES:
        return "violent"
    if primary_type in PROPERTY_CRIMES:
        return "property"
    if primary_type in VEHICLE_CRIMES:
        return "vehicle"
    if primary_type in NARCOTICS_CRIMES:
        return "narcotics"
    if primary_type in PUBLIC_DISORDER:
        return "disorder"
    return "other"


def connect():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT,
    )


def ensure_schema(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crimes (id BIGINT PRIMARY KEY);
        ALTER TABLE crimes ADD COLUMN IF NOT EXISTS case_number TEXT;
        ALTER TABLE crimes ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP;
        ALTER TABLE crimes ADD COLUMN IF NOT EXISTS primary_type TEXT;
        ALTER TABLE crimes ADD COLUMN IF NOT EXISTS description TEXT;
        ALTER TABLE crimes ADD COLUMN IF NOT EXISTS arrest BOOLEAN;
        ALTER TABLE crimes ADD COLUMN IF NOT EXISTS domestic BOOLEAN;
        ALTER TABLE crimes ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION;
        ALTER TABLE crimes ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION;
        ALTER TABLE crimes ADD COLUMN IF NOT EXISTS cell_id TEXT;
        ALTER TABLE crimes ADD COLUMN IF NOT EXISTS crime_category TEXT;
        CREATE INDEX IF NOT EXISTS idx_crimes_timestamp ON crimes(timestamp);
        CREATE INDEX IF NOT EXISTS idx_crimes_cell_id ON crimes(cell_id);
        CREATE INDEX IF NOT EXISTS idx_crimes_primary_type ON crimes(primary_type);
    """)


def database_range(cursor):
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM crimes")
    return cursor.fetchone()


def fetch_window(start, end):
    cursor_date = None
    cursor_id = None
    while True:
        where = [
            f"date >= '{iso_literal(start)}'",
            f"date < '{iso_literal(end)}'",
            "latitude IS NOT NULL",
            "longitude IS NOT NULL",
        ]
        if cursor_date is not None:
            where.append(
                f"(date > '{cursor_date}' OR "
                f"(date = '{cursor_date}' AND id > {int(cursor_id)}))"
            )
        rows = request_json(
            CRIME_API_URL,
            {
                "$where": " AND ".join(where),
                "$limit": PAGE_SIZE,
                "$order": "date ASC, id ASC",
            },
            SOCRATA_APP_TOKEN,
        )
        if not rows:
            break
        yield rows
        last = rows[-1]
        cursor_date = str(last["date"]).replace("'", "''")
        cursor_id = last["id"]
        if len(rows) < PAGE_SIZE:
            break


def fetch_source_latest():
    rows = request_json(
        CRIME_API_URL,
        {
            "$select": "max(date) as latest",
            "$where": "date IS NOT NULL",
        },
        SOCRATA_APP_TOKEN,
    )
    value = (rows or [{}])[0].get("latest")
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def transform(row):
    if not row.get("id") or not row.get("date"):
        return None
    if row.get("latitude") is None or row.get("longitude") is None:
        return None
    latitude = float(row["latitude"])
    longitude = float(row["longitude"])
    primary_type = row.get("primary_type")
    return (
        int(row["id"]), row.get("case_number"), row["date"], primary_type,
        row.get("description"), str(row.get("arrest", "")).lower() == "true",
        str(row.get("domestic", "")).lower() == "true", latitude, longitude,
        latlon_to_cell(latitude, longitude), map_crime_category(primary_type),
    )


def upsert_page(cursor, rows):
    values = [value for value in (transform(row) for row in rows) if value]
    if not values:
        return 0
    execute_values(cursor, """
        INSERT INTO crimes (
            id, case_number, timestamp, primary_type, description, arrest,
            domestic, latitude, longitude, cell_id, crime_category
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            case_number = EXCLUDED.case_number,
            timestamp = EXCLUDED.timestamp,
            primary_type = EXCLUDED.primary_type,
            description = EXCLUDED.description,
            arrest = EXCLUDED.arrest,
            domestic = EXCLUDED.domestic,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            cell_id = EXCLUDED.cell_id,
            crime_category = EXCLUDED.crime_category
    """, values, page_size=5000)
    return len(values)


def sync_range(connection, start, end, label):
    processed = 0
    for window_start, window_end in date_windows(start, end):
        print(f"{label}: {window_start.date()} to {window_end.date()}")
        for rows in fetch_window(window_start, window_end):
            with connection.cursor() as cursor:
                processed += upsert_page(cursor, rows)
            connection.commit()
            print(f"  processed {processed:,} rows")
    return processed


def main(history_years=DEFAULT_HISTORY_YEARS):
    now = datetime.now()
    target_start = now - timedelta(days=round(365.25 * history_years))
    with connect() as connection:
        with connection.cursor() as cursor:
            ensure_schema(cursor)
        connection.commit()
        with connection.cursor() as cursor:
            minimum, maximum = database_range(cursor)

        total = 0
        if minimum is None:
            total += sync_range(connection, target_start, now + timedelta(days=1), "initial")
        else:
            if minimum > target_start:
                total += sync_range(connection, target_start, minimum, "backfill")
            incremental_start = max(target_start, maximum - timedelta(days=OVERLAP_DAYS))
            total += sync_range(
                connection, incremental_start, now + timedelta(days=1), "incremental"
            )
        with connection.cursor() as cursor:
            _, database_latest = database_range(cursor)
    source_latest = fetch_source_latest()
    metadata = {
        "checked_at": now.isoformat(),
        "database_latest": database_latest.isoformat() if database_latest else None,
        "source_latest": source_latest.isoformat() if source_latest else None,
        "database_age_hours": (
            round((now - database_latest).total_seconds() / 3600.0, 2)
            if database_latest else None
        ),
        "source_age_hours": (
            round((now - source_latest).total_seconds() / 3600.0, 2)
            if source_latest else None
        ),
        "database_vs_source_gap_hours": (
            round((source_latest - database_latest).total_seconds() / 3600.0, 2)
            if source_latest and database_latest else None
        ),
    }
    with (DEBUG_DIR / "crime_source_watermark.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(json.dumps(metadata, indent=2))
    print(f"Crime synchronization complete; processed {total:,} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-years", type=float, default=DEFAULT_HISTORY_YEARS)
    args = parser.parse_args()
    main(args.history_years)
