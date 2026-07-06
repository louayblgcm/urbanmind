import sys
import os
import json
import time
import hashlib
import requests
import psycopg2
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.config import (
    DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT,
    BUSINESS_API_URL, PERMITS_API_URL,
)
from utils.spatial import latlon_to_cell

BUSINESS_URL = BUSINESS_API_URL
PERMITS_URL = PERMITS_API_URL

PAGE_SIZE = 10000
REQUEST_TIMEOUT = 180
MAX_RETRIES = 3
SLEEP_BETWEEN_PAGES = 1

BUSINESS_START_DATE = datetime(2018, 1, 1)
PERMITS_START_DATE = datetime(2018, 1, 1)


def connect():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )


def ensure_business_schema(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS business_licenses (
            record_id TEXT PRIMARY KEY
        );
    """)

    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS license_id TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS account_number TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS site_number TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS legal_name TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS doing_business_as_name TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS license_code TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS license_description TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS license_start_date TIMESTAMP;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS license_term_expiration_date TIMESTAMP;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS address TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS city TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS state TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS zip_code TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS ward TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS precinct TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS police_district TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS cell_id TEXT;""")
    cursor.execute("""ALTER TABLE business_licenses ADD COLUMN IF NOT EXISTS raw_data JSONB;""")

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_business_license_start_date
        ON business_licenses(license_start_date);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_business_cell_id
        ON business_licenses(cell_id);
    """)


def ensure_permits_schema(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS building_permits (
            record_id TEXT PRIMARY KEY
        );
    """)

    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS permit_ TEXT;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS permit_type TEXT;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS review_type TEXT;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS application_start_date TIMESTAMP;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS issue_date TIMESTAMP;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS processing_time TEXT;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS street_number TEXT;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS street_direction TEXT;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS street_name TEXT;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS suffix TEXT;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS work_description TEXT;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS building_fee_paid DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS zoning_fee_paid DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS other_fee_paid DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS subtotal_paid DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS building_fee_unpaid DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS zoning_fee_unpaid DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS other_fee_unpaid DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS subtotal_unpaid DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS cell_id TEXT;""")
    cursor.execute("""ALTER TABLE building_permits ADD COLUMN IF NOT EXISTS raw_data JSONB;""")

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_permits_issue_date
        ON building_permits(issue_date);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_permits_cell_id
        ON building_permits(cell_id);
    """)


def get_latest_date(cursor, table_name, date_column, default_start_date):
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    count = cursor.fetchone()[0]

    if count == 0:
        return default_start_date

    cursor.execute(f"SELECT MAX({date_column}) FROM {table_name};")
    latest = cursor.fetchone()[0]

    return latest if latest is not None else default_start_date


def fetch_page(url, date_column, start_date, offset):
    params = {
        "$limit": PAGE_SIZE,
        "$offset": offset,
        "$order": f"{date_column} ASC",
        "$where": f"{date_column} > '{start_date.strftime('%Y-%m-%dT%H:%M:%S')}'"
    }

    headers = {
        "User-Agent": "UrbanMindV2/1.0"
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ReadTimeout:
            print(f"Read timeout at offset {offset} (attempt {attempt}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(5)

        except requests.exceptions.RequestException as e:
            print(f"Request error at offset {offset}: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(5)

    return []


def to_float(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def build_business_record_id(row):
    key_parts = [
        str(row.get("license_id", "")),
        str(row.get("account_number", "")),
        str(row.get("site_number", "")),
        str(row.get("license_start_date", ""))
    ]
    return hashlib.md5("||".join(key_parts).encode("utf-8")).hexdigest()


def build_permit_record_id(row):
    key_parts = [
        str(row.get("permit_", "")),
        str(row.get("issue_date", "")),
        str(row.get("street_number", "")),
        str(row.get("street_name", ""))
    ]
    return hashlib.md5("||".join(key_parts).encode("utf-8")).hexdigest()


def get_cell_id(row):
    lat = row.get("latitude")
    lon = row.get("longitude")

    if lat is None or lon is None:
        return None

    if str(lat).strip() == "" or str(lon).strip() == "":
        return None

    return latlon_to_cell(float(lat), float(lon))


def keep_business_row(row):
    if not row.get("license_start_date"):
        return False
    return True


def keep_permit_row(row):
    if not row.get("issue_date"):
        return False
    return True


def insert_business_row(cursor, row):
    if not keep_business_row(row):
        return False

    record_id = build_business_record_id(row)
    cell_id = get_cell_id(row)

    cursor.execute("""
        INSERT INTO business_licenses (
            record_id,
            license_id,
            account_number,
            site_number,
            legal_name,
            doing_business_as_name,
            license_code,
            license_description,
            license_start_date,
            license_term_expiration_date,
            address,
            city,
            state,
            zip_code,
            ward,
            precinct,
            police_district,
            latitude,
            longitude,
            cell_id,
            raw_data
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (record_id) DO NOTHING;
    """, (
        record_id,
        row.get("license_id"),
        row.get("account_number"),
        row.get("site_number"),
        row.get("legal_name"),
        row.get("doing_business_as_name"),
        row.get("license_code"),
        row.get("license_description"),
        row.get("license_start_date"),
        row.get("license_term_expiration_date"),
        row.get("address"),
        row.get("city"),
        row.get("state"),
        row.get("zip_code"),
        row.get("ward"),
        row.get("precinct"),
        row.get("police_district"),
        to_float(row.get("latitude")),
        to_float(row.get("longitude")),
        cell_id,
        json.dumps(row)
    ))

    return True


def insert_permit_row(cursor, row):
    if not keep_permit_row(row):
        return False

    record_id = build_permit_record_id(row)
    cell_id = get_cell_id(row)

    cursor.execute("""
        INSERT INTO building_permits (
            record_id,
            permit_,
            permit_type,
            review_type,
            application_start_date,
            issue_date,
            processing_time,
            street_number,
            street_direction,
            street_name,
            suffix,
            work_description,
            building_fee_paid,
            zoning_fee_paid,
            other_fee_paid,
            subtotal_paid,
            building_fee_unpaid,
            zoning_fee_unpaid,
            other_fee_unpaid,
            subtotal_unpaid,
            latitude,
            longitude,
            cell_id,
            raw_data
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (record_id) DO NOTHING;
    """, (
        record_id,
        row.get("permit_"),
        row.get("permit_type"),
        row.get("review_type"),
        row.get("application_start_date"),
        row.get("issue_date"),
        row.get("processing_time"),
        row.get("street_number"),
        row.get("street_direction"),
        row.get("street_name"),
        row.get("suffix"),
        row.get("work_description"),
        to_float(row.get("building_fee_paid")),
        to_float(row.get("zoning_fee_paid")),
        to_float(row.get("other_fee_paid")),
        to_float(row.get("subtotal_paid")),
        to_float(row.get("building_fee_unpaid")),
        to_float(row.get("zoning_fee_unpaid")),
        to_float(row.get("other_fee_unpaid")),
        to_float(row.get("subtotal_unpaid")),
        to_float(row.get("latitude")),
        to_float(row.get("longitude")),
        cell_id,
        json.dumps(row)
    ))

    return True


def fetch_and_insert_business(cursor, start_date):
    offset = 0
    inserted = 0

    while True:
        rows = fetch_page(BUSINESS_URL, "license_start_date", start_date, offset)

        if not rows:
            break

        print(f"Fetched {len(rows)} business license rows at offset {offset}")

        for row in rows:
            try:
                ok = insert_business_row(cursor, row)
                if ok:
                    inserted += 1
            except Exception as e:
                cursor.connection.rollback()
                ensure_business_schema(cursor)
                cursor.connection.commit()
                print("Error:", e)
                print("Bad row:", json.dumps(row, indent=2))

        cursor.connection.commit()
        offset += PAGE_SIZE
        time.sleep(SLEEP_BETWEEN_PAGES)

    return inserted


def fetch_and_insert_permits(cursor, start_date):
    offset = 0
    inserted = 0

    while True:
        rows = fetch_page(PERMITS_URL, "issue_date", start_date, offset)

        if not rows:
            break

        print(f"Fetched {len(rows)} building permit rows at offset {offset}")

        for row in rows:
            try:
                ok = insert_permit_row(cursor, row)
                if ok:
                    inserted += 1
            except Exception as e:
                cursor.connection.rollback()
                ensure_permits_schema(cursor)
                cursor.connection.commit()
                print("Error:", e)
                print("Bad row:", json.dumps(row, indent=2))

        cursor.connection.commit()
        offset += PAGE_SIZE
        time.sleep(SLEEP_BETWEEN_PAGES)

    return inserted


def main():
    conn = connect()
    cursor = conn.cursor()

    ensure_business_schema(cursor)
    ensure_permits_schema(cursor)
    conn.commit()

    business_start_date = get_latest_date(
        cursor,
        "business_licenses",
        "license_start_date",
        BUSINESS_START_DATE
    )
    print("Fetching business licenses from:", business_start_date)
    business_inserted = fetch_and_insert_business(cursor, business_start_date)

    permits_start_date = get_latest_date(
        cursor,
        "building_permits",
        "issue_date",
        PERMITS_START_DATE
    )
    print("Fetching building permits from:", permits_start_date)
    permits_inserted = fetch_and_insert_permits(cursor, permits_start_date)

    cursor.close()
    conn.close()

    print(f"Done. Inserted {business_inserted} new business license rows.")
    print(f"Done. Inserted {permits_inserted} new building permit rows.")


if __name__ == "__main__":
    main()
