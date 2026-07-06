import math

import psycopg2
from psycopg2.extras import RealDictCursor

from config.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from utils.spatial import latlon_to_cell


LAT_CELL_METERS = 150.0
LON_CELL_METERS = 137.0
MAX_CONTEXT_RADIUS_METERS = 900.0


def _score(value, multiplier=24.0):
    return round(min(math.log1p(max(float(value or 0), 0.0)) * multiplier, 100.0), 2)


def _cell_indices(cell_id):
    lat_index, lon_index = str(cell_id).split("_")
    return int(lat_index), int(lon_index)


def _weighted_sum(rows, target_indices, columns, radius_meters, decay_meters):
    target_lat, target_lon = target_indices
    total = 0.0
    for row in rows:
        row_lat, row_lon = _cell_indices(row["cell_id"])
        distance = math.hypot(
            (row_lat - target_lat) * LAT_CELL_METERS,
            (row_lon - target_lon) * LON_CELL_METERS,
        )
        if distance > radius_meters:
            continue
        weight = math.exp(-distance / decay_meters)
        total += weight * sum(float(row[column] or 0) for column in columns)
    return total


def empty_osm():
    keys = (
        "nightlife_density", "commercial_density", "transit_intensity",
        "healthcare_access", "restaurant_activity", "pedestrian_activity",
        "urban_diversity", "workplace_activity", "urban_vitality",
        "nightlife_semantic", "transit_corridor_activity",
    )
    return {
        "urban_environment": {key: {"score": 0.0} for key in keys},
        "spatial_context": {"cells_considered": 0},
    }


def get_osm_context(lat, lon):
    candidate = latlon_to_cell(lat, lon)
    nearest_query = """
        SELECT f.cell_id
        FROM static_cell_features f
        ORDER BY CASE WHEN f.cell_id = %s THEN 0 ELSE 1 END,
                 POWER(split_part(f.cell_id, '_', 1)::DOUBLE PRECISION * 0.00135 - %s, 2)
                 + POWER(split_part(f.cell_id, '_', 2)::DOUBLE PRECISION * 0.00165 - %s, 2)
        LIMIT 1
    """
    neighborhood_query = """
        SELECT cell_id, poi_total, poi_food_drink, poi_retail, poi_healthcare,
               poi_education, poi_transport, poi_leisure_culture,
               poi_public_service, business_total, permits_total,
               traffic_segment_count
        FROM static_cell_features
        WHERE ABS(split_part(cell_id, '_', 1)::INTEGER - %s) <= 7
          AND ABS(split_part(cell_id, '_', 2)::INTEGER - %s) <= 7
    """

    try:
        with psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT,
        ) as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(nearest_query, (candidate, lat, lon))
            target = cursor.fetchone()
            if not target:
                return empty_osm()
            target_indices = _cell_indices(target["cell_id"])
            cursor.execute(neighborhood_query, target_indices)
            rows = cursor.fetchall()

        food_500 = _weighted_sum(rows, target_indices, ("poi_food_drink",), 500, 220)
        retail_business_600 = _weighted_sum(
            rows, target_indices, ("poi_retail", "business_total"), 600, 260
        )
        transit_650 = _weighted_sum(
            rows, target_indices, ("poi_transport", "traffic_segment_count"), 650, 280
        )
        healthcare_900 = _weighted_sum(
            rows, target_indices, ("poi_healthcare",), 900, 380
        )
        pedestrian_550 = _weighted_sum(
            rows, target_indices, ("poi_total", "traffic_segment_count"), 550, 240
        )

        restaurant = _score(food_500, 26)
        nightlife = _score(food_500, 21)
        commercial = _score(retail_business_600, 18)
        transit = _score(transit_650, 22)
        healthcare = _score(healthcare_900, 28)
        pedestrian = _score(pedestrian_550, 17)

        diversity_columns = (
            "poi_food_drink", "poi_retail", "poi_healthcare", "poi_education",
            "poi_transport", "poi_leisure_culture", "poi_public_service",
        )
        diversity_present = sum(
            _weighted_sum(rows, target_indices, (column,), 600, 260) >= 0.15
            for column in diversity_columns
        )
        diversity = round(min(diversity_present / len(diversity_columns) * 100.0, 100.0), 2)
        workplace = round(min(commercial * 0.65 + transit * 0.35, 100.0), 2)
        vitality = round(min(
            pedestrian * 0.35 + diversity * 0.25 + commercial * 0.20 + transit * 0.20,
            100.0,
        ), 2)
        nightlife_semantic = round(min(nightlife * 0.7 + restaurant * 0.3, 100.0), 2)
        transit_corridor = round(min(transit * 0.7 + pedestrian * 0.3, 100.0), 2)
        values = {
            "nightlife_density": nightlife,
            "commercial_density": commercial,
            "transit_intensity": transit,
            "healthcare_access": healthcare,
            "restaurant_activity": restaurant,
            "pedestrian_activity": pedestrian,
            "urban_diversity": diversity,
            "workplace_activity": workplace,
            "urban_vitality": vitality,
            "nightlife_semantic": nightlife_semantic,
            "transit_corridor_activity": transit_corridor,
        }
        return {
            "urban_environment": {
                key: {"score": value} for key, value in values.items()
            },
            "spatial_context": {
                "target_cell": target["cell_id"],
                "cells_considered": len(rows),
                "radii_meters": {
                    "nightlife_restaurant": 500,
                    "commercial_workplace": 600,
                    "transit": 650,
                    "healthcare": 900,
                },
                "distance_weighting": "exponential_decay",
            },
        }
    except (psycopg2.Error, TypeError, ValueError):
        return empty_osm()
