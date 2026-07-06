import sys
import os
import json
import time
import hashlib
import requests
import psycopg2
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, TRAFFIC_API_URL
from utils.spatial import latlon_to_cell

API_URL = TRAFFIC_API_URL
PAGE_SIZE = 10000
REQUEST_TIMEOUT = 180
MAX_RETRIES = 3
SLEEP_BETWEEN_PAGES = 1
MAX_ROWS_PER_RUN = 1500000


def connect():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )


def ensure_schema(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traffic_data (
            record_id TEXT PRIMARY KEY
        );
    """)

    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS segment_id TEXT;""")
    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS time TIMESTAMP;""")
    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS street TEXT;""")
    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS from_street TEXT;""")
    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS to_street TEXT;""")
    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS traffic DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS speed DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS length DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS heading TEXT;""")
    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS start_latitude DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS start_longitude DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS end_latitude DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS end_longitude DOUBLE PRECISION;""")
    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS cell_id TEXT;""")
    cursor.execute("""ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS raw_data JSONB;""")

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_traffic_time
        ON traffic_data(time);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_traffic_segment_id
        ON traffic_data(segment_id);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_traffic_cell_id
        ON traffic_data(cell_id);
    """)


def get_start_date(cursor):
    cursor.execute("SELECT COUNT(*) FROM traffic_data;")
    count = cursor.fetchone()[0]

    if count == 0:
        return datetime(2018, 1, 1)

    cursor.execute("SELECT MAX(time) FROM traffic_data;")
    latest = cursor.fetchone()[0]

    if latest is None:
        return datetime(2018, 1, 1)

    return latest


def fetch_page(start_date, offset):
    params = {
        "$limit": PAGE_SIZE,
        "$offset": offset,
        "$order": "time ASC",
        "$where": f"time > '{start_date.strftime('%Y-%m-%dT%H:%M:%S')}'"
    }

    headers = {
        "User-Agent": "UrbanMindV2/1.0"
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                API_URL,
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


def first_value(row, candidates):
    for key in candidates:
        if key in row and row.get(key) not in [None, ""]:
            return row.get(key)
    return None


def to_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def get_segment_id(row):
    value = first_value(row, [
        "segmentid",
        "segment_id",
        "segmentid_str",
        "segment",
        "id"
    ])
    return None if value is None else str(value)


def get_time_value(row):
    value = first_value(row, [
        "time",
        "traffic_datetime",
        "datetime",
        "measurement_time",
        "last_updated"
    ])
    return value


def get_street(row):
    value = first_value(row, [
        "street",
        "street_name"
    ])
    return None if value is None else str(value)


def get_from_street(row):
    value = first_value(row, [
        "fromst",
        "from_street",
        "fromstreet",
        "start_street"
    ])
    return None if value is None else str(value)


def get_to_street(row):
    value = first_value(row, [
        "tost",
        "to_street",
        "tostreet",
        "end_street"
    ])
    return None if value is None else str(value)


def get_traffic_value(row):
    return to_float(first_value(row, [
        "traffic",
        "bus_count",
        "vehicle_count",
        "count",
        "traffic_volume"
    ]))


def get_speed_value(row):
    return to_float(first_value(row, [
        "speed",
        "average_speed"
    ]))


def get_length_value(row):
    return to_float(first_value(row, [
        "length",
        "segment_length"
    ]))


def get_heading_value(row):
    value = first_value(row, [
        "street_heading",
        "heading",
        "direction"
    ])
    return None if value is None else str(value)


def get_start_lat(row):
    return to_float(first_value(row, [
        "start_latitude",
        "start_lat",
        "latitude",
        "lat"
    ]))


def get_start_lon(row):
    return to_float(first_value(row, [
        "start_longitude",
        "start_lon",
        "longitude",
        "lon"
    ]))


def get_end_lat(row):
    return to_float(first_value(row, [
        "end_latitude",
        "end_lat"
    ]))


def get_end_lon(row):
    return to_float(first_value(row, [
        "end_longitude",
        "end_lon"
    ]))


def build_record_id(row):
    key_parts = [
        str(get_segment_id(row) or ""),
        str(get_time_value(row) or ""),
        str(get_street(row) or ""),
        str(get_from_street(row) or ""),
        str(get_to_street(row) or ""),
        str(get_traffic_value(row) or "")
    ]
    key = "||".join(key_parts)
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def get_cell_id(row):
    start_lat = get_start_lat(row)
    start_lon = get_start_lon(row)

    if start_lat is not None and start_lon is not None:
        return latlon_to_cell(start_lat, start_lon)

    end_lat = get_end_lat(row)
    end_lon = get_end_lon(row)

    if end_lat is not None and end_lon is not None:
        return latlon_to_cell(end_lat, end_lon)

    return None


def keep_row(row):
    time_value = get_time_value(row)
    segment_id = get_segment_id(row)

    if not time_value:
        return False

    if not segment_id:
        return False

    return True


def insert_row(cursor, row):
    if not keep_row(row):
        return False

    record_id = build_record_id(row)
    cell_id = get_cell_id(row)

    cursor.execute("""
        INSERT INTO traffic_data (
            record_id,
            segment_id,
            time,
            street,
            from_street,
            to_street,
            traffic,
            speed,
            length,
            heading,
            start_latitude,
            start_longitude,
            end_latitude,
            end_longitude,
            cell_id,
            raw_data
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (record_id) DO NOTHING;
    """, (
        record_id,
        get_segment_id(row),
        get_time_value(row),
        get_street(row),
        get_from_street(row),
        get_to_street(row),
        get_traffic_value(row),
        get_speed_value(row),
        get_length_value(row),
        get_heading_value(row),
        get_start_lat(row),
        get_start_lon(row),
        get_end_lat(row),
        get_end_lon(row),
        cell_id,
        json.dumps(row)
    ))

    return True


def fetch_and_insert(cursor, start_date):
    offset = 0
    inserted = 0
    printed_sample = False

    while inserted < MAX_ROWS_PER_RUN:
        rows = fetch_page(start_date, offset)

        if not rows:
            break

        print(f"Fetched {len(rows)} traffic rows at offset {offset}")

        if not printed_sample and len(rows) > 0:
            print("Sample row keys:", list(rows[0].keys()))
            print("Sample row:", json.dumps(rows[0], indent=2))
            printed_sample = True

        page_inserted = 0

        for row in rows:
            if inserted >= MAX_ROWS_PER_RUN:
                break

            try:
                ok = insert_row(cursor, row)
                if ok:
                    inserted += 1
                    page_inserted += 1

            except Exception as e:
                cursor.connection.rollback()
                ensure_schema(cursor)
                cursor.connection.commit()
                print("Error:", e)
                print("Bad row:", json.dumps(row, indent=2))

        cursor.connection.commit()
        print(f"Committed {page_inserted} rows this page. Total inserted so far: {inserted}")

        if inserted >= MAX_ROWS_PER_RUN:
            print(f"Reached max limit of {MAX_ROWS_PER_RUN} rows for this run.")
            break

        offset += PAGE_SIZE
        time.sleep(SLEEP_BETWEEN_PAGES)

    return inserted


def main():
    conn = connect()
    cursor = conn.cursor()

    ensure_schema(cursor)
    conn.commit()

    start_date = get_start_date(cursor)
    print("Fetching traffic data from:", start_date)

    inserted = fetch_and_insert(cursor, start_date)

    cursor.close()
    conn.close()

    print(f"Done. Inserted {inserted} new traffic rows.")


if __name__ == "__main__":
    main()
