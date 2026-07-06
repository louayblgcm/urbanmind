import argparse
import os
import sys
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import Json, execute_values

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.config import (
    DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER,
    REQUESTS_311_API_URL, SOCRATA_APP_TOKEN,
)
from scripts.ingestion.socrata_utils import date_windows, iso_literal, parse_timestamp, request_json
from utils.spatial import latlon_to_cell


PAGE_SIZE = int(os.getenv("REQUEST_311_INGEST_PAGE_SIZE", "50000"))
DEFAULT_HISTORY_YEARS = float(os.getenv("INGEST_HISTORY_YEARS", "5"))
OVERLAP_DAYS = int(os.getenv("INGEST_OVERLAP_DAYS", "7"))
RAW_RETENTION_DAYS = int(os.getenv("INGEST_RAW_RETENTION_DAYS", "90"))


def connect():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT,
    )


def ensure_schema(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requests_311 (sr_number TEXT PRIMARY KEY);
        ALTER TABLE requests_311 ADD COLUMN IF NOT EXISTS created_date TIMESTAMP;
        ALTER TABLE requests_311 ADD COLUMN IF NOT EXISTS status TEXT;
        ALTER TABLE requests_311 ADD COLUMN IF NOT EXISTS sr_type TEXT;
        ALTER TABLE requests_311 ADD COLUMN IF NOT EXISTS street_address TEXT;
        ALTER TABLE requests_311 ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION;
        ALTER TABLE requests_311 ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION;
        ALTER TABLE requests_311 ADD COLUMN IF NOT EXISTS cell_id TEXT;
        ALTER TABLE requests_311 ADD COLUMN IF NOT EXISTS duplicate BOOLEAN;
        ALTER TABLE requests_311 ADD COLUMN IF NOT EXISTS origin TEXT;
        ALTER TABLE requests_311 ADD COLUMN IF NOT EXISTS raw_data JSONB;
        CREATE INDEX IF NOT EXISTS idx_311_created_date ON requests_311(created_date);
        CREATE INDEX IF NOT EXISTS idx_311_cell_id ON requests_311(cell_id);
        CREATE INDEX IF NOT EXISTS idx_311_sr_type ON requests_311(sr_type);
    """)


def database_range(cursor):
    cursor.execute("SELECT MIN(created_date), MAX(created_date) FROM requests_311")
    return cursor.fetchone()


def _escape(value):
    return str(value).replace("'", "''")


def fetch_window(start, end):
    cursor_date = None
    cursor_number = None
    while True:
        where = [
            f"created_date >= '{iso_literal(start)}'",
            f"created_date < '{iso_literal(end)}'",
            "latitude IS NOT NULL",
            "longitude IS NOT NULL",
            "sr_type != '311 INFORMATION ONLY CALL'",
        ]
        if cursor_date is not None:
            where.append(
                f"(created_date > '{cursor_date}' OR "
                f"(created_date = '{cursor_date}' AND sr_number > '{cursor_number}'))"
            )
        rows = request_json(
            REQUESTS_311_API_URL,
            {
                "$where": " AND ".join(where),
                "$limit": PAGE_SIZE,
                "$order": "created_date ASC, sr_number ASC",
            },
            SOCRATA_APP_TOKEN,
        )
        if not rows:
            break
        yield rows
        last = rows[-1]
        cursor_date = _escape(last["created_date"])
        cursor_number = _escape(last["sr_number"])
        if len(rows) < PAGE_SIZE:
            break


def keep_row(row):
    if not row.get("sr_number") or not row.get("created_date"):
        return False
    if row.get("latitude") is None or row.get("longitude") is None:
        return False
    if str(row.get("street_address", "")).upper() == "121 N LA SALLE ST":
        return False
    if str(row.get("duplicate", "")).lower() == "true":
        return False
    if str(row.get("origin", "")).lower() == "generated in house":
        return False
    return True


def transform(row, raw_cutoff):
    if not keep_row(row):
        return None
    latitude = float(row["latitude"])
    longitude = float(row["longitude"])
    created = parse_timestamp(row["created_date"])
    raw_data = Json(row) if created >= raw_cutoff else None
    return (
        row["sr_number"], row["created_date"], row.get("status"),
        row.get("sr_type"), row.get("street_address"), latitude, longitude,
        latlon_to_cell(latitude, longitude),
        str(row.get("duplicate", "")).lower() == "true",
        row.get("origin"), raw_data,
    )


def upsert_page(cursor, rows, raw_cutoff):
    values = [value for value in (transform(row, raw_cutoff) for row in rows) if value]
    if not values:
        return 0
    execute_values(cursor, """
        INSERT INTO requests_311 (
            sr_number, created_date, status, sr_type, street_address, latitude,
            longitude, cell_id, duplicate, origin, raw_data
        ) VALUES %s
        ON CONFLICT (sr_number) DO UPDATE SET
            created_date = EXCLUDED.created_date,
            status = EXCLUDED.status,
            sr_type = EXCLUDED.sr_type,
            street_address = EXCLUDED.street_address,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            cell_id = EXCLUDED.cell_id,
            duplicate = EXCLUDED.duplicate,
            origin = EXCLUDED.origin,
            raw_data = COALESCE(EXCLUDED.raw_data, requests_311.raw_data)
    """, values, page_size=5000)
    return len(values)


def sync_range(connection, start, end, label, raw_cutoff):
    processed = 0
    for window_start, window_end in date_windows(start, end):
        print(f"{label}: {window_start.date()} to {window_end.date()}")
        for rows in fetch_window(window_start, window_end):
            with connection.cursor() as cursor:
                processed += upsert_page(cursor, rows, raw_cutoff)
            connection.commit()
            print(f"  processed {processed:,} rows")
    return processed


def main(history_years=DEFAULT_HISTORY_YEARS):
    now = datetime.now()
    target_start = now - timedelta(days=round(365.25 * history_years))
    raw_cutoff = now - timedelta(days=RAW_RETENTION_DAYS)
    with connect() as connection:
        with connection.cursor() as cursor:
            ensure_schema(cursor)
        connection.commit()
        with connection.cursor() as cursor:
            minimum, maximum = database_range(cursor)

        total = 0
        if minimum is None:
            total += sync_range(
                connection, target_start, now + timedelta(days=1), "initial", raw_cutoff
            )
        else:
            if minimum > target_start:
                total += sync_range(
                    connection, target_start, minimum, "backfill", raw_cutoff
                )
            incremental_start = max(target_start, maximum - timedelta(days=OVERLAP_DAYS))
            total += sync_range(
                connection, incremental_start, now + timedelta(days=1),
                "incremental", raw_cutoff,
            )
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE requests_311 SET raw_data = NULL "
                "WHERE created_date < %s AND raw_data IS NOT NULL",
                (raw_cutoff,),
            )
            print(f"Released old raw 311 payloads: {cursor.rowcount:,} rows")
        connection.commit()
    print(f"311 synchronization complete; processed {total:,} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-years", type=float, default=DEFAULT_HISTORY_YEARS)
    args = parser.parse_args()
    main(args.history_years)
