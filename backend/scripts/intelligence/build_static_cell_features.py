import sys
import os
import json
import psycopg2
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
from config.modeling import chronological_boundaries

STATIC_DEBUG_DIR = os.path.join("backend", "data", "debug", "static_features")


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
        CREATE TABLE IF NOT EXISTS static_cell_features (
            cell_id TEXT PRIMARY KEY,

            crime_total DOUBLE PRECISION,
            crime_per_day DOUBLE PRECISION,
            violent_crime_count DOUBLE PRECISION,
            property_crime_count DOUBLE PRECISION,
            vehicle_crime_count DOUBLE PRECISION,
            narcotics_crime_count DOUBLE PRECISION,
            disorder_crime_count DOUBLE PRECISION,
            crime_diversity DOUBLE PRECISION,
            arrest_ratio DOUBLE PRECISION,
            domestic_ratio DOUBLE PRECISION,

            requests_311_total DOUBLE PRECISION,
            requests_311_per_day DOUBLE PRECISION,
            requests_311_type_diversity DOUBLE PRECISION,

            poi_total DOUBLE PRECISION,
            poi_food_drink DOUBLE PRECISION,
            poi_retail DOUBLE PRECISION,
            poi_healthcare DOUBLE PRECISION,
            poi_education DOUBLE PRECISION,
            poi_transport DOUBLE PRECISION,
            poi_leisure_culture DOUBLE PRECISION,
            poi_public_service DOUBLE PRECISION,
            poi_other DOUBLE PRECISION,

            business_total DOUBLE PRECISION,
            permits_total DOUBLE PRECISION,
            permit_fee_total DOUBLE PRECISION,

            traffic_mean DOUBLE PRECISION,
            traffic_max DOUBLE PRECISION,
            traffic_segment_count DOUBLE PRECISION,

            baseline_crime_hour_00 DOUBLE PRECISION,
            baseline_crime_hour_01 DOUBLE PRECISION,
            baseline_crime_hour_02 DOUBLE PRECISION,
            baseline_crime_hour_03 DOUBLE PRECISION,
            baseline_crime_hour_04 DOUBLE PRECISION,
            baseline_crime_hour_05 DOUBLE PRECISION,
            baseline_crime_hour_06 DOUBLE PRECISION,
            baseline_crime_hour_07 DOUBLE PRECISION,
            baseline_crime_hour_08 DOUBLE PRECISION,
            baseline_crime_hour_09 DOUBLE PRECISION,
            baseline_crime_hour_10 DOUBLE PRECISION,
            baseline_crime_hour_11 DOUBLE PRECISION,
            baseline_crime_hour_12 DOUBLE PRECISION,
            baseline_crime_hour_13 DOUBLE PRECISION,
            baseline_crime_hour_14 DOUBLE PRECISION,
            baseline_crime_hour_15 DOUBLE PRECISION,
            baseline_crime_hour_16 DOUBLE PRECISION,
            baseline_crime_hour_17 DOUBLE PRECISION,
            baseline_crime_hour_18 DOUBLE PRECISION,
            baseline_crime_hour_19 DOUBLE PRECISION,
            baseline_crime_hour_20 DOUBLE PRECISION,
            baseline_crime_hour_21 DOUBLE PRECISION,
            baseline_crime_hour_22 DOUBLE PRECISION,
            baseline_crime_hour_23 DOUBLE PRECISION,

            created_at TIMESTAMP DEFAULT NOW()
        );
    """)


def rebuild_static_table(cursor, feature_cutoff):
    cursor.execute("TRUNCATE TABLE static_cell_features;")

    cursor.execute("""
        INSERT INTO static_cell_features (
            cell_id,

            crime_total,
            crime_per_day,
            violent_crime_count,
            property_crime_count,
            vehicle_crime_count,
            narcotics_crime_count,
            disorder_crime_count,
            crime_diversity,
            arrest_ratio,
            domestic_ratio,

            requests_311_total,
            requests_311_per_day,
            requests_311_type_diversity,

            poi_total,
            poi_food_drink,
            poi_retail,
            poi_healthcare,
            poi_education,
            poi_transport,
            poi_leisure_culture,
            poi_public_service,
            poi_other,

            business_total,
            permits_total,
            permit_fee_total,

            traffic_mean,
            traffic_max,
            traffic_segment_count,

            baseline_crime_hour_00,
            baseline_crime_hour_01,
            baseline_crime_hour_02,
            baseline_crime_hour_03,
            baseline_crime_hour_04,
            baseline_crime_hour_05,
            baseline_crime_hour_06,
            baseline_crime_hour_07,
            baseline_crime_hour_08,
            baseline_crime_hour_09,
            baseline_crime_hour_10,
            baseline_crime_hour_11,
            baseline_crime_hour_12,
            baseline_crime_hour_13,
            baseline_crime_hour_14,
            baseline_crime_hour_15,
            baseline_crime_hour_16,
            baseline_crime_hour_17,
            baseline_crime_hour_18,
            baseline_crime_hour_19,
            baseline_crime_hour_20,
            baseline_crime_hour_21,
            baseline_crime_hour_22,
            baseline_crime_hour_23
        )
        WITH
        model_cutoff AS (
            SELECT %s::TIMESTAMP AS cutoff_ts
        ),
        all_cells AS (
            SELECT DISTINCT cell_id FROM crimes
            WHERE cell_id IS NOT NULL
              AND timestamp < (SELECT cutoff_ts FROM model_cutoff)
            UNION
            SELECT DISTINCT cell_id FROM requests_311
            WHERE cell_id IS NOT NULL
              AND created_date < (SELECT cutoff_ts FROM model_cutoff)
            UNION
            SELECT DISTINCT cell_id FROM osm_pois WHERE cell_id IS NOT NULL
            UNION
            SELECT DISTINCT cell_id FROM business_licenses
            WHERE cell_id IS NOT NULL
              AND license_start_date < (SELECT cutoff_ts FROM model_cutoff)
            UNION
            SELECT DISTINCT cell_id FROM building_permits
            WHERE cell_id IS NOT NULL
              AND issue_date < (SELECT cutoff_ts FROM model_cutoff)
            UNION
            SELECT DISTINCT cell_id FROM traffic_data
            WHERE cell_id IS NOT NULL
              AND time < (SELECT cutoff_ts FROM model_cutoff)
        ),

        crime_agg AS (
            SELECT
                cell_id,
                COUNT(*)::DOUBLE PRECISION AS crime_total,
                COUNT(*)::DOUBLE PRECISION / GREATEST(COUNT(DISTINCT DATE(timestamp)), 1) AS crime_per_day,

                SUM(CASE WHEN crime_category = 'violent' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS violent_crime_count,
                SUM(CASE WHEN crime_category = 'property' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS property_crime_count,
                SUM(CASE WHEN crime_category = 'vehicle' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS vehicle_crime_count,
                SUM(CASE WHEN crime_category = 'narcotics' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS narcotics_crime_count,
                SUM(CASE WHEN crime_category = 'disorder' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS disorder_crime_count,

                COUNT(DISTINCT primary_type)::DOUBLE PRECISION AS crime_diversity,

                AVG(CASE WHEN arrest IS TRUE THEN 1.0 ELSE 0.0 END)::DOUBLE PRECISION AS arrest_ratio,
                AVG(CASE WHEN domestic IS TRUE THEN 1.0 ELSE 0.0 END)::DOUBLE PRECISION AS domestic_ratio
            FROM crimes
            WHERE cell_id IS NOT NULL
              AND timestamp < (SELECT cutoff_ts FROM model_cutoff)
            GROUP BY cell_id
        ),

        crime_hours AS (
            SELECT
                cell_id,
                EXTRACT(HOUR FROM timestamp)::INT AS hour_of_day,
                COUNT(*)::DOUBLE PRECISION AS hour_count
            FROM crimes
            WHERE cell_id IS NOT NULL
              AND timestamp IS NOT NULL
              AND timestamp < (SELECT cutoff_ts FROM model_cutoff)
            GROUP BY cell_id, EXTRACT(HOUR FROM timestamp)
        ),

        crime_hour_pivot AS (
            SELECT
                cell_id,

                COALESCE(SUM(CASE WHEN hour_of_day = 0  THEN hour_count END), 0) AS h00,
                COALESCE(SUM(CASE WHEN hour_of_day = 1  THEN hour_count END), 0) AS h01,
                COALESCE(SUM(CASE WHEN hour_of_day = 2  THEN hour_count END), 0) AS h02,
                COALESCE(SUM(CASE WHEN hour_of_day = 3  THEN hour_count END), 0) AS h03,
                COALESCE(SUM(CASE WHEN hour_of_day = 4  THEN hour_count END), 0) AS h04,
                COALESCE(SUM(CASE WHEN hour_of_day = 5  THEN hour_count END), 0) AS h05,
                COALESCE(SUM(CASE WHEN hour_of_day = 6  THEN hour_count END), 0) AS h06,
                COALESCE(SUM(CASE WHEN hour_of_day = 7  THEN hour_count END), 0) AS h07,
                COALESCE(SUM(CASE WHEN hour_of_day = 8  THEN hour_count END), 0) AS h08,
                COALESCE(SUM(CASE WHEN hour_of_day = 9  THEN hour_count END), 0) AS h09,
                COALESCE(SUM(CASE WHEN hour_of_day = 10 THEN hour_count END), 0) AS h10,
                COALESCE(SUM(CASE WHEN hour_of_day = 11 THEN hour_count END), 0) AS h11,
                COALESCE(SUM(CASE WHEN hour_of_day = 12 THEN hour_count END), 0) AS h12,
                COALESCE(SUM(CASE WHEN hour_of_day = 13 THEN hour_count END), 0) AS h13,
                COALESCE(SUM(CASE WHEN hour_of_day = 14 THEN hour_count END), 0) AS h14,
                COALESCE(SUM(CASE WHEN hour_of_day = 15 THEN hour_count END), 0) AS h15,
                COALESCE(SUM(CASE WHEN hour_of_day = 16 THEN hour_count END), 0) AS h16,
                COALESCE(SUM(CASE WHEN hour_of_day = 17 THEN hour_count END), 0) AS h17,
                COALESCE(SUM(CASE WHEN hour_of_day = 18 THEN hour_count END), 0) AS h18,
                COALESCE(SUM(CASE WHEN hour_of_day = 19 THEN hour_count END), 0) AS h19,
                COALESCE(SUM(CASE WHEN hour_of_day = 20 THEN hour_count END), 0) AS h20,
                COALESCE(SUM(CASE WHEN hour_of_day = 21 THEN hour_count END), 0) AS h21,
                COALESCE(SUM(CASE WHEN hour_of_day = 22 THEN hour_count END), 0) AS h22,
                COALESCE(SUM(CASE WHEN hour_of_day = 23 THEN hour_count END), 0) AS h23
            FROM crime_hours
            GROUP BY cell_id
        ),

        crime_span AS (
            SELECT
                GREATEST(
                    MAX(timestamp)::DATE - MIN(timestamp)::DATE + 1,
                    1
                )::DOUBLE PRECISION AS observation_days
            FROM crimes
            WHERE timestamp IS NOT NULL
              AND timestamp < (SELECT cutoff_ts FROM model_cutoff)
        ),

        crime_hist AS (
            SELECT
                p.cell_id,
                h00 / s.observation_days AS baseline_crime_hour_00,
                h01 / s.observation_days AS baseline_crime_hour_01,
                h02 / s.observation_days AS baseline_crime_hour_02,
                h03 / s.observation_days AS baseline_crime_hour_03,
                h04 / s.observation_days AS baseline_crime_hour_04,
                h05 / s.observation_days AS baseline_crime_hour_05,
                h06 / s.observation_days AS baseline_crime_hour_06,
                h07 / s.observation_days AS baseline_crime_hour_07,
                h08 / s.observation_days AS baseline_crime_hour_08,
                h09 / s.observation_days AS baseline_crime_hour_09,
                h10 / s.observation_days AS baseline_crime_hour_10,
                h11 / s.observation_days AS baseline_crime_hour_11,
                h12 / s.observation_days AS baseline_crime_hour_12,
                h13 / s.observation_days AS baseline_crime_hour_13,
                h14 / s.observation_days AS baseline_crime_hour_14,
                h15 / s.observation_days AS baseline_crime_hour_15,
                h16 / s.observation_days AS baseline_crime_hour_16,
                h17 / s.observation_days AS baseline_crime_hour_17,
                h18 / s.observation_days AS baseline_crime_hour_18,
                h19 / s.observation_days AS baseline_crime_hour_19,
                h20 / s.observation_days AS baseline_crime_hour_20,
                h21 / s.observation_days AS baseline_crime_hour_21,
                h22 / s.observation_days AS baseline_crime_hour_22,
                h23 / s.observation_days AS baseline_crime_hour_23
            FROM crime_hour_pivot p
            CROSS JOIN crime_span s
        ),

        requests_311_agg AS (
            SELECT
                cell_id,
                COUNT(*)::DOUBLE PRECISION AS requests_311_total,
                COUNT(*)::DOUBLE PRECISION / GREATEST(COUNT(DISTINCT DATE(created_date)), 1) AS requests_311_per_day,
                COUNT(DISTINCT sr_type)::DOUBLE PRECISION AS requests_311_type_diversity
            FROM requests_311
            WHERE cell_id IS NOT NULL
              AND created_date < (SELECT cutoff_ts FROM model_cutoff)
            GROUP BY cell_id
        ),

        osm_agg AS (
            SELECT
                cell_id,
                COUNT(*)::DOUBLE PRECISION AS poi_total,
                SUM(CASE WHEN category = 'food_drink' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS poi_food_drink,
                SUM(CASE WHEN category = 'retail' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS poi_retail,
                SUM(CASE WHEN category = 'healthcare' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS poi_healthcare,
                SUM(CASE WHEN category = 'education' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS poi_education,
                SUM(CASE WHEN category = 'transport' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS poi_transport,
                SUM(CASE WHEN category = 'leisure_culture' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS poi_leisure_culture,
                SUM(CASE WHEN category = 'public_service' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS poi_public_service,
                SUM(CASE WHEN category = 'other' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS poi_other
            FROM osm_pois
            WHERE cell_id IS NOT NULL
            GROUP BY cell_id
        ),

        business_agg AS (
            SELECT
                cell_id,
                COUNT(*)::DOUBLE PRECISION AS business_total
            FROM business_licenses
            WHERE cell_id IS NOT NULL
              AND license_start_date < (SELECT cutoff_ts FROM model_cutoff)
            GROUP BY cell_id
        ),

        permits_agg AS (
            SELECT
                cell_id,
                COUNT(*)::DOUBLE PRECISION AS permits_total,
                COALESCE(SUM(
                    COALESCE(building_fee_paid, 0) +
                    COALESCE(zoning_fee_paid, 0) +
                    COALESCE(other_fee_paid, 0) +
                    COALESCE(subtotal_paid, 0)
                ), 0)::DOUBLE PRECISION AS permit_fee_total
            FROM building_permits
            WHERE cell_id IS NOT NULL
              AND issue_date < (SELECT cutoff_ts FROM model_cutoff)
            GROUP BY cell_id
        ),

        traffic_agg AS (
            SELECT
                cell_id,
                AVG(traffic)::DOUBLE PRECISION AS traffic_mean,
                MAX(traffic)::DOUBLE PRECISION AS traffic_max,
                COUNT(DISTINCT segment_id)::DOUBLE PRECISION AS traffic_segment_count
            FROM traffic_data
            WHERE cell_id IS NOT NULL
              AND time < (SELECT cutoff_ts FROM model_cutoff)
            GROUP BY cell_id
        )

        SELECT
            c.cell_id,

            COALESCE(cr.crime_total, 0),
            COALESCE(cr.crime_per_day, 0),
            COALESCE(cr.violent_crime_count, 0),
            COALESCE(cr.property_crime_count, 0),
            COALESCE(cr.vehicle_crime_count, 0),
            COALESCE(cr.narcotics_crime_count, 0),
            COALESCE(cr.disorder_crime_count, 0),
            COALESCE(cr.crime_diversity, 0),
            COALESCE(cr.arrest_ratio, 0),
            COALESCE(cr.domestic_ratio, 0),

            COALESCE(rq.requests_311_total, 0),
            COALESCE(rq.requests_311_per_day, 0),
            COALESCE(rq.requests_311_type_diversity, 0),

            COALESCE(osm.poi_total, 0),
            COALESCE(osm.poi_food_drink, 0),
            COALESCE(osm.poi_retail, 0),
            COALESCE(osm.poi_healthcare, 0),
            COALESCE(osm.poi_education, 0),
            COALESCE(osm.poi_transport, 0),
            COALESCE(osm.poi_leisure_culture, 0),
            COALESCE(osm.poi_public_service, 0),
            COALESCE(osm.poi_other, 0),

            COALESCE(bl.business_total, 0),
            COALESCE(pm.permits_total, 0),
            COALESCE(pm.permit_fee_total, 0),

            COALESCE(tf.traffic_mean, 0),
            COALESCE(tf.traffic_max, 0),
            COALESCE(tf.traffic_segment_count, 0),

            COALESCE(ch.baseline_crime_hour_00, 0),
            COALESCE(ch.baseline_crime_hour_01, 0),
            COALESCE(ch.baseline_crime_hour_02, 0),
            COALESCE(ch.baseline_crime_hour_03, 0),
            COALESCE(ch.baseline_crime_hour_04, 0),
            COALESCE(ch.baseline_crime_hour_05, 0),
            COALESCE(ch.baseline_crime_hour_06, 0),
            COALESCE(ch.baseline_crime_hour_07, 0),
            COALESCE(ch.baseline_crime_hour_08, 0),
            COALESCE(ch.baseline_crime_hour_09, 0),
            COALESCE(ch.baseline_crime_hour_10, 0),
            COALESCE(ch.baseline_crime_hour_11, 0),
            COALESCE(ch.baseline_crime_hour_12, 0),
            COALESCE(ch.baseline_crime_hour_13, 0),
            COALESCE(ch.baseline_crime_hour_14, 0),
            COALESCE(ch.baseline_crime_hour_15, 0),
            COALESCE(ch.baseline_crime_hour_16, 0),
            COALESCE(ch.baseline_crime_hour_17, 0),
            COALESCE(ch.baseline_crime_hour_18, 0),
            COALESCE(ch.baseline_crime_hour_19, 0),
            COALESCE(ch.baseline_crime_hour_20, 0),
            COALESCE(ch.baseline_crime_hour_21, 0),
            COALESCE(ch.baseline_crime_hour_22, 0),
            COALESCE(ch.baseline_crime_hour_23, 0)

        FROM all_cells c
        LEFT JOIN crime_agg cr ON c.cell_id = cr.cell_id
        LEFT JOIN requests_311_agg rq ON c.cell_id = rq.cell_id
        LEFT JOIN osm_agg osm ON c.cell_id = osm.cell_id
        LEFT JOIN business_agg bl ON c.cell_id = bl.cell_id
        LEFT JOIN permits_agg pm ON c.cell_id = pm.cell_id
        LEFT JOIN traffic_agg tf ON c.cell_id = tf.cell_id
        LEFT JOIN crime_hist ch ON c.cell_id = ch.cell_id;
    """, (feature_cutoff,))


def main():
    conn = connect()
    cursor = conn.cursor()

    ensure_schema(cursor)
    conn.commit()

    cursor.execute("SELECT MAX(timestamp) FROM crimes;")
    latest_timestamp = cursor.fetchone()[0]
    if latest_timestamp is None:
        raise ValueError("No crimes available to define the modeling cutoff")
    boundaries = chronological_boundaries(latest_timestamp)
    feature_cutoff = boundaries["train_end"]

    print(f"Building leakage-safe static features using data before {feature_cutoff} ...")
    rebuild_static_table(cursor, feature_cutoff)
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM static_cell_features;")
    count = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    os.makedirs(STATIC_DEBUG_DIR, exist_ok=True)
    with open(os.path.join(STATIC_DEBUG_DIR, "build_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "latest_crime_timestamp": latest_timestamp.isoformat(),
                "static_feature_cutoff_exclusive": feature_cutoff.isoformat(),
                "chronological_boundaries": {
                    key: value.isoformat() for key, value in boundaries.items()
                },
                "leakage_safe_backtest_snapshot": True,
            },
            file,
            indent=2,
        )

    print(f"Done. Built {count} static cell rows.")
    print(f"Finished at {datetime.now()}")


if __name__ == "__main__":
    main()
