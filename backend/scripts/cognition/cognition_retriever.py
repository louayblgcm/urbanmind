"""Database retrieval layer for the v2 urban-intelligence pipeline."""

import logging

import psycopg2
from psycopg2.extras import RealDictCursor

from config.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from scripts.intelligence.dynamic_forecast_service import forecast_next_24_hours
from utils.spatial import latlon_to_cell


LOGGER = logging.getLogger(__name__)


def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


class CognitionRetriever:
    def resolve_cell(self, lat, lon):
        candidate = latlon_to_cell(lat, lon)
        with get_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM static_gnn_profiles WHERE cell_id = %s", (candidate,)
            )
            if cursor.fetchone():
                return candidate
            cursor.execute(
                """
                SELECT p.cell_id
                FROM static_gnn_profiles p
                ORDER BY
                    POWER(split_part(p.cell_id, '_', 1)::DOUBLE PRECISION * 0.00135 - %s, 2)
                    + POWER(split_part(p.cell_id, '_', 2)::DOUBLE PRECISION * 0.00165 - %s, 2)
                LIMIT 1
                """,
                (lat, lon),
            )
            row = cursor.fetchone()
        if row is None:
            raise LookupError("No trained cells are available in static_gnn_profiles")
        return row[0]

    def get_temporal_state(self, cell_id):
        with get_connection() as connection, connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:
            cursor.execute(
                """
                WITH hours AS (SELECT generate_series(0, 23) AS hour),
                crime_by_hour AS (
                    SELECT EXTRACT(HOUR FROM timestamp)::INTEGER AS hour,
                           COUNT(*)::DOUBLE PRECISION AS crime_count
                    FROM crimes WHERE cell_id = %s GROUP BY 1
                ),
                request_by_hour AS (
                    SELECT EXTRACT(HOUR FROM created_date)::INTEGER AS hour,
                           COUNT(*)::DOUBLE PRECISION AS request_311_count
                    FROM requests_311 WHERE cell_id = %s GROUP BY 1
                )
                SELECT h.hour,
                       COALESCE(c.crime_count, 0) AS crime_count,
                       COALESCE(r.request_311_count, 0) AS request_311_count
                FROM hours h
                LEFT JOIN crime_by_hour c USING (hour)
                LEFT JOIN request_by_hour r USING (hour)
                ORDER BY h.hour
                """,
                (cell_id, cell_id),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_forecast(self, cell_id):
        return forecast_next_24_hours(cell_id)

    def get_cluster(self, cell_id):
        with get_connection() as connection, connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:
            cursor.execute(
                """
                SELECT p.cell_id, p.cluster, p.urban_profile,
                       p.static_crime_score, p.static_activity_score,
                       f.crime_total, f.crime_per_day, f.violent_crime_count,
                       f.property_crime_count, f.vehicle_crime_count,
                       f.requests_311_total, f.requests_311_per_day,
                       f.poi_total, f.poi_food_drink, f.poi_retail,
                       f.poi_healthcare, f.poi_education, f.poi_transport,
                       f.poi_leisure_culture, f.poi_public_service,
                       f.business_total, f.permits_total, f.traffic_mean
                FROM static_gnn_profiles p
                JOIN static_cell_features f ON f.cell_id = p.cell_id
                WHERE p.cell_id = %s
                """,
                (cell_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else {}

    def get_recent_crimes(self, lat, lon, limit=20, cell_id=None):
        cell_id = cell_id or self.resolve_cell(lat, lon)
        with get_connection() as connection, connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:
            cursor.execute(
                """
                SELECT case_number, timestamp, primary_type, description,
                       arrest, domestic, latitude, longitude, crime_category
                FROM crimes
                WHERE cell_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (cell_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_311(self, lat, lon, limit=20, cell_id=None):
        cell_id = cell_id or self.resolve_cell(lat, lon)
        with get_connection() as connection, connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:
            cursor.execute(
                """
                SELECT sr_number, created_date, status, sr_type, street_address,
                       latitude, longitude, origin
                FROM requests_311
                WHERE cell_id = %s
                ORDER BY created_date DESC
                LIMIT %s
                """,
                (cell_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_embeddings(self, cell_id):
        with get_connection() as connection, connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:
            cursor.execute(
                """
                SELECT cell_id, embedding_vector, cluster, urban_profile
                FROM static_gnn_profiles WHERE cell_id = %s
                """,
                (cell_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else {}

    def build_cognition_packet(self, lat, lon):
        cell_id = self.resolve_cell(lat, lon)
        return {
            "cell_id": cell_id,
            "temporal_state": self.get_temporal_state(cell_id),
            "forecast": self.get_forecast(cell_id),
            "cluster": self.get_cluster(cell_id),
            "recent_crimes": self.get_recent_crimes(lat, lon, cell_id=cell_id),
            "recent_311": self.get_recent_311(lat, lon, cell_id=cell_id),
            "embeddings": self.get_embeddings(cell_id),
        }
