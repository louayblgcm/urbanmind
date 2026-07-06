"""Import the bundled Chicago-area OpenStreetMap shapefiles into ``osm_pois``."""

import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.config import BACKEND_DIR, DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from utils.spatial import latlon_to_cell


SHAPEFILE_DIR = Path(os.getenv(
    "OSM_SHAPEFILE_DIR",
    BACKEND_DIR / "data" / "osm" / "illinois_shp",
))
CHICAGO_BOUNDS = (-87.95, 41.63, -87.50, 42.05)
SOURCE_FILES = (
    "gis_osm_pois_free_1.shp",
    "gis_osm_pois_a_free_1.shp",
    "gis_osm_transport_free_1.shp",
    "gis_osm_transport_a_free_1.shp",
    "gis_osm_traffic_free_1.shp",
    "gis_osm_traffic_a_free_1.shp",
    "gis_osm_pofw_free_1.shp",
    "gis_osm_pofw_a_free_1.shp",
)

FOOD_DRINK = {
    "restaurant", "fast_food", "cafe", "bar", "pub", "biergarten",
    "food_court", "bakery",
}
RETAIL = {
    "mall", "supermarket", "convenience", "market_place", "department_store",
    "kiosk", "clothes", "hairdresser", "beauty_shop", "mobile_phone_shop",
}
HEALTHCARE = {
    "hospital", "clinic", "doctors", "dentist", "pharmacy", "veterinary",
}
EDUCATION = {"school", "college", "university", "kindergarten", "library"}
TRANSPORT = {
    "bus_stop", "bus_station", "railway_station", "tram_stop", "subway_entrance",
    "taxi", "ferry_terminal", "airport", "parking", "parking_multistorey",
    "bicycle_rental", "bicycle_parking", "car_sharing", "fuel",
}
LEISURE = {
    "park", "playground", "pitch", "sports_centre", "stadium", "theatre",
    "cinema", "museum", "community_centre", "arts_centre", "attraction", "zoo",
    "swimming_pool", "artwork", "memorial",
}
PUBLIC_SERVICE = {
    "police", "fire_station", "post_office", "post_box", "town_hall", "courthouse",
    "social_facility", "public_building", "toilet", "recycling", "shelter",
}


def connect():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def ensure_schema(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS osm_pois (
            osm_id TEXT PRIMARY KEY,
            osm_type TEXT,
            name TEXT,
            category TEXT NOT NULL,
            subtype TEXT,
            latitude DOUBLE PRECISION NOT NULL,
            longitude DOUBLE PRECISION NOT NULL,
            cell_id TEXT NOT NULL,
            osm_timestamp TIMESTAMP,
            raw_data JSONB,
            source_file TEXT
        );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_osm_pois_cell_id ON osm_pois(cell_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_osm_pois_category ON osm_pois(category);")


def category_for(subtype, source_name):
    subtype = str(subtype or "").strip().lower()
    if "transport" in source_name or "traffic" in source_name or subtype in TRANSPORT:
        return "transport"
    if subtype in FOOD_DRINK:
        return "food_drink"
    if subtype in RETAIL or subtype.startswith("shop_"):
        return "retail"
    if subtype in HEALTHCARE:
        return "healthcare"
    if subtype in EDUCATION:
        return "education"
    if subtype in LEISURE:
        return "leisure_culture"
    if "pofw" in source_name or subtype in PUBLIC_SERVICE:
        return "public_service"
    return "other"


def clean_scalar(value):
    if value is None or pd.isna(value):
        return None
    return str(value)


def load_file(path):
    frame = gpd.read_file(path, bbox=CHICAGO_BOUNDS)
    if frame.empty:
        return []
    frame = frame.set_crs(4326) if frame.crs is None else frame.to_crs(4326)
    points = frame.geometry.representative_point()
    rows = []

    for index, record in frame.iterrows():
        point = points.loc[index]
        geometry = record.get("geometry")
        if point is None or point.is_empty or geometry is None or geometry.is_empty:
            continue
        longitude = float(point.x)
        latitude = float(point.y)
        if not (
            CHICAGO_BOUNDS[0] <= longitude <= CHICAGO_BOUNDS[2]
            and CHICAGO_BOUNDS[1] <= latitude <= CHICAGO_BOUNDS[3]
        ):
            continue

        subtype = clean_scalar(record.get("fclass")) or clean_scalar(record.get("type"))
        name = clean_scalar(record.get("name"))
        raw_data = {
            str(key): clean_scalar(value)
            for key, value in record.items()
            if key != "geometry"
        }
        rows.append((
            str(record.get("osm_id", index)),
            "area" if geometry.geom_type in {"Polygon", "MultiPolygon"} else "node",
            name,
            category_for(subtype, path.name),
            subtype,
            latitude,
            longitude,
            latlon_to_cell(latitude, longitude),
            None,
            json.dumps(raw_data),
            path.name,
        ))
    return rows


def upsert_rows(cursor, rows):
    if not rows:
        return
    execute_values(
        cursor,
        """
        INSERT INTO osm_pois (
            osm_id, osm_type, name, category, subtype, latitude, longitude,
            cell_id, osm_timestamp, raw_data, source_file
        ) VALUES %s
        ON CONFLICT (osm_id) DO UPDATE SET
            osm_type = EXCLUDED.osm_type,
            name = EXCLUDED.name,
            category = EXCLUDED.category,
            subtype = EXCLUDED.subtype,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            cell_id = EXCLUDED.cell_id,
            raw_data = EXCLUDED.raw_data,
            source_file = EXCLUDED.source_file;
        """,
        rows,
        page_size=2000,
    )


def main(rebuild=False, dry_run=False):
    paths = [SHAPEFILE_DIR / filename for filename in SOURCE_FILES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing OSM shapefiles: " + ", ".join(missing))

    loaded = [(path, load_file(path)) for path in paths]
    for path, rows in loaded:
        print(f"Prepared {len(rows):,} rows from {path.name}")
    if dry_run:
        print(f"Dry run complete: {sum(len(rows) for _, rows in loaded):,} rows")
        return

    with connect() as connection, connection.cursor() as cursor:
        ensure_schema(cursor)
        if rebuild:
            cursor.execute("TRUNCATE TABLE osm_pois;")
        for _, rows in loaded:
            upsert_rows(cursor, rows)

    print(f"OSM import complete: {sum(len(rows) for _, rows in loaded):,} source rows")


if __name__ == "__main__":
    main(
        rebuild="--rebuild" in sys.argv,
        dry_run="--dry-run" in sys.argv,
    )
