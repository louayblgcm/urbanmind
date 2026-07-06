import json
import os
import sys
from datetime import datetime

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER


DEBUG_DIR = os.path.join("backend", "data", "debug", "dynamic_shadow")
os.makedirs(DEBUG_DIR, exist_ok=True)


def connect():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def ensure_shadow_table(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dynamic_forecast_shadow (
                cell_id TEXT NOT NULL,
                reference_time TIMESTAMP NOT NULL,
                model_version TEXT NOT NULL,
                model_name TEXT NOT NULL,
                forecast_status TEXT NOT NULL,
                abstained BOOLEAN NOT NULL,
                target_start TIMESTAMP NOT NULL,
                target_end TIMESTAMP NOT NULL,
                forecast JSONB NOT NULL,
                realized_total_24h DOUBLE PRECISION,
                realized_any_24h BOOLEAN,
                evaluated_at TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                PRIMARY KEY (cell_id, reference_time, model_version)
            )
            """
        )
    connection.commit()


def update_realized_outcomes(connection):
    with connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            """
            SELECT cell_id, reference_time, model_version, target_start, target_end
            FROM dynamic_forecast_shadow
            WHERE target_end <= NOW()
              AND realized_total_24h IS NULL
            ORDER BY reference_time
            """
        )
        pending = cursor.fetchall()
        for row in pending:
            cursor.execute(
                """
                SELECT COUNT(*)::DOUBLE PRECISION AS total
                FROM crimes
                WHERE cell_id = %s
                  AND timestamp >= %s
                  AND timestamp < %s
                """,
                (row["cell_id"], row["target_start"], row["target_end"]),
            )
            total = float(cursor.fetchone()["total"] or 0.0)
            cursor.execute(
                """
                UPDATE dynamic_forecast_shadow
                SET realized_total_24h = %s,
                    realized_any_24h = %s,
                    evaluated_at = NOW()
                WHERE cell_id = %s
                  AND reference_time = %s
                  AND model_version = %s
                """,
                (
                    total,
                    total > 0,
                    row["cell_id"],
                    row["reference_time"],
                    row["model_version"],
                ),
            )
    connection.commit()


def summarize(connection):
    with connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            """
            SELECT
                model_name,
                forecast_status,
                abstained,
                COUNT(*) AS rows,
                AVG(ABS(COALESCE((forecast->>'expected_crime_count_24h')::DOUBLE PRECISION, 0) - realized_total_24h)) AS mae_24h,
                AVG(COALESCE((forecast->>'expected_crime_count_24h')::DOUBLE PRECISION, 0)) AS avg_pred_24h,
                AVG(realized_total_24h) AS avg_true_24h
            FROM dynamic_forecast_shadow
            WHERE realized_total_24h IS NOT NULL
            GROUP BY model_name, forecast_status, abstained
            ORDER BY rows DESC, model_name
            """
        )
        rows = [dict(row) for row in cursor.fetchall()]
    summary = {
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "groups": rows,
    }
    with open(os.path.join(DEBUG_DIR, "shadow_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    with connect() as connection:
        ensure_shadow_table(connection)
        update_realized_outcomes(connection)
        summarize(connection)


if __name__ == "__main__":
    main()
