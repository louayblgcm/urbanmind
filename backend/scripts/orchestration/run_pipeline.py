"""Safe ingestion, retraining, validation, promotion, and rollback pipeline."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import Json, RealDictCursor


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent
LOG_DIR = BACKEND_DIR / "data" / "logs" / "pipeline"
MODEL_DIR = BACKEND_DIR / "data" / "models"
DEBUG_DIR = BACKEND_DIR / "data" / "debug"
ARCHIVE_DIR = MODEL_DIR / "archive"

sys.path.insert(0, str(BACKEND_DIR))
from config.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER  # noqa: E402


INGESTION_STAGES = (
    ("ingest_crimes", "scripts/ingestion/build_crimes_database.py"),
    ("ingest_311", "scripts/ingestion/build_311_database.py"),
    ("ingest_urban_growth", "scripts/ingestion/build_urban_growth_database.py"),
)
OSM_STAGE = ("ingest_osm", "scripts/ingestion/build_osm_database.py")
ENABLE_BOOSTED_CHALLENGER = os.getenv("PIPELINE_ENABLE_BOOSTED_CHALLENGER", "").strip().lower() in {
    "1", "true", "yes", "on"
}

TRAINING_STAGES = [
    ("build_static_features", "scripts/intelligence/build_static_cell_features.py"),
    ("train_static_gnn", "scripts/intelligence/train_static_gnn.py"),
    ("build_dynamic_tensors", "scripts/intelligence/build_dynamic_cell_tensors.py"),
    ("train_dynamic_forecaster", "scripts/intelligence/train_dynamic_forecaster.py"),
]
if ENABLE_BOOSTED_CHALLENGER:
    TRAINING_STAGES.append(
        ("train_dynamic_boosted", "scripts/intelligence/train_dynamic_boosted.py")
    )
TRAINING_STAGES = tuple(TRAINING_STAGES)

ACTIVE_MODEL_FILES = (
    MODEL_DIR / "static_gnn.pt",
    MODEL_DIR / "hierarchical_forecaster_3h.pt",
    MODEL_DIR / "hierarchical_forecaster_3h.joblib",
)
CANDIDATE_DYNAMIC_MODELS = [
    {
        "model_family": "neural_two_head",
        "candidate_path": MODEL_DIR / "hierarchical_forecaster_3h.candidate.pt",
        "active_path": MODEL_DIR / "hierarchical_forecaster_3h.pt",
        "metrics_path": DEBUG_DIR / "hierarchical_dynamic_trainer" / "metrics_summary.json",
        "label": "hierarchical_forecaster_500m_3h_neural",
    },
]
if ENABLE_BOOSTED_CHALLENGER:
    CANDIDATE_DYNAMIC_MODELS.append(
        {
            "model_family": "boosted_hurdle",
            "candidate_path": MODEL_DIR / "hierarchical_forecaster_3h.boosted.candidate.joblib",
            "active_path": MODEL_DIR / "hierarchical_forecaster_3h.joblib",
            "metrics_path": DEBUG_DIR / "hierarchical_dynamic_boosted_trainer" / "metrics_summary.json",
            "label": "hierarchical_forecaster_500m_3h_boosted",
        }
    )
CANDIDATE_DYNAMIC_MODELS = tuple(CANDIDATE_DYNAMIC_MODELS)
BACKUP_TABLES = (
    ("static_cell_features", "pipeline_backup_static_cell_features"),
    ("static_gnn_profiles", "pipeline_backup_static_gnn_profiles"),
)

RETRAIN_INTERVAL_DAYS = float(os.getenv("PIPELINE_RETRAIN_INTERVAL_DAYS", "7"))
INGEST_INTERVAL_HOURS = float(os.getenv("PIPELINE_INGEST_INTERVAL_HOURS", "24"))
MAX_CRIME_AGE_HOURS = float(os.getenv("PIPELINE_MAX_CRIME_AGE_HOURS", "48"))
MIN_MAE_IMPROVEMENT = float(os.getenv("PIPELINE_MIN_MAE_IMPROVEMENT", "0.05"))
MIN_AP_LIFT = float(os.getenv("PIPELINE_MIN_AP_LIFT", "3.0"))
MIN_TOP1_RECALL = float(os.getenv("PIPELINE_MIN_TOP1_RECALL", "0.10"))
MAX_TOTAL_BIAS = float(os.getenv("PIPELINE_MAX_TOTAL_BIAS", "0.10"))
MAX_CLUSTER_CALIBRATION_ERROR = float(
    os.getenv("PIPELINE_MAX_CLUSTER_CALIBRATION_ERROR", "0.08")
)
DATA_TIMEZONE = ZoneInfo(os.getenv("PIPELINE_DATA_TIMEZONE", "America/Chicago"))


class PipelineError(RuntimeError):
    pass


class PromotionRejected(PipelineError):
    def __init__(self, gate_report):
        super().__init__("Candidate model failed promotion gates")
        self.gate_report = gate_report


def connect():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def ensure_history_schema(connection):
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_pipeline_runs (
                run_id BIGSERIAL PRIMARY KEY,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                finished_at TIMESTAMPTZ,
                stages JSONB NOT NULL DEFAULT '[]'::JSONB,
                data_watermarks JSONB,
                metrics JSONB,
                gate_report JSONB,
                error TEXT,
                log_directory TEXT
            )
        """)


def acquire_lock(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_try_advisory_lock(hashtext('urbanmind_model_pipeline'))"
        )
        if not cursor.fetchone()[0]:
            raise PipelineError("Another UrbanMind pipeline run is already active")


def release_lock(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_advisory_unlock(hashtext('urbanmind_model_pipeline'))"
        )


def start_run(connection, mode):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO model_pipeline_runs (mode, status)
            VALUES (%s, 'running') RETURNING run_id
            """,
            (mode,),
        )
        return int(cursor.fetchone()[0])


def finish_run(connection, run_id, **values):
    assignments = ["finished_at = NOW()"]
    params = []
    for column, value in values.items():
        assignments.append(f"{column} = %s")
        params.append(Json(value) if isinstance(value, (dict, list)) else value)
    params.append(run_id)
    with connection.cursor() as cursor:
        cursor.execute(
            f"UPDATE model_pipeline_runs SET {', '.join(assignments)} WHERE run_id = %s",
            params,
        )


def update_progress(connection, run_id, stages, watermarks=None):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE model_pipeline_runs
            SET stages = %s, data_watermarks = COALESCE(%s, data_watermarks)
            WHERE run_id = %s
            """,
            (Json(stages), Json(watermarks) if watermarks else None, run_id),
        )


def data_watermarks(connection):
    with connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("SELECT MAX(timestamp) AS latest FROM crimes")
        crime = cursor.fetchone()["latest"]
        cursor.execute("SELECT MAX(created_date) AS latest FROM requests_311")
        requests_311 = cursor.fetchone()["latest"]

    now_local = datetime.now(DATA_TIMEZONE).replace(tzinfo=None)
    crime_age = (
        (now_local - crime).total_seconds() / 3600.0
        if crime is not None else None
    )
    source_meta_path = DEBUG_DIR / "ingestion" / "crime_source_watermark.json"
    source_meta = {}
    if source_meta_path.exists():
        try:
            with source_meta_path.open("r", encoding="utf-8") as file:
                source_meta = json.load(file)
        except (OSError, json.JSONDecodeError):
            source_meta = {}

    return {
        "crime_latest": crime.isoformat() if crime else None,
        "requests_311_latest": requests_311.isoformat() if requests_311 else None,
        "crime_age_hours": round(crime_age, 2) if crime_age is not None else None,
        "crime_source_latest": source_meta.get("source_latest"),
        "crime_source_age_hours": source_meta.get("source_age_hours"),
        "crime_database_vs_source_gap_hours": source_meta.get("database_vs_source_gap_hours"),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def run_stage(stage_name, relative_script, run_log_dir):
    script = BACKEND_DIR / relative_script
    if not script.exists():
        raise PipelineError(f"Missing stage script: {script}")
    log_path = run_log_dir / f"{stage_name}.log"
    started = datetime.now(timezone.utc)
    print(f"\n[{stage_name}] starting")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-u", str(script)],
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    finished = datetime.now(timezone.utc)
    result = {
        "stage": stage_name,
        "status": "completed" if return_code == 0 else "failed",
        "return_code": return_code,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 2),
        "log": str(log_path),
    }
    if return_code != 0:
        raise PipelineError(f"Stage {stage_name} failed; see {log_path}")
    return result


def table_exists(connection, table_name):
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
        return cursor.fetchone()[0] is not None


def backup_training_state(connection, run_id):
    backup_dir = ARCHIVE_DIR / f"run_{run_id}_previous"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for model_path in ACTIVE_MODEL_FILES:
        if model_path.exists():
            shutil.copy2(model_path, backup_dir / model_path.name)

    with connection.cursor() as cursor:
        for source, backup in BACKUP_TABLES:
            cursor.execute(f"DROP TABLE IF EXISTS {backup}")
            if table_exists(connection, source):
                cursor.execute(
                    f"CREATE TABLE {backup} (LIKE {source} INCLUDING ALL)"
                )
                cursor.execute(f"INSERT INTO {backup} SELECT * FROM {source}")
    return backup_dir


def restore_training_state(connection, backup_dir, run_id):
    rejected_dir = ARCHIVE_DIR / f"run_{run_id}_rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    for model_path in ACTIVE_MODEL_FILES:
        if model_path.exists():
            shutil.copy2(model_path, rejected_dir / model_path.name)
        previous = backup_dir / model_path.name
        if previous.exists():
            shutil.copy2(previous, model_path)

    metrics_path = DEBUG_DIR / "hierarchical_dynamic_trainer" / "metrics_summary.json"
    if metrics_path.exists():
        shutil.copy2(metrics_path, rejected_dir / "metrics_summary.json")

    with connection.cursor() as cursor:
        for target, backup in BACKUP_TABLES:
            if table_exists(connection, backup):
                cursor.execute(f"TRUNCATE TABLE {target}")
                cursor.execute(f"INSERT INTO {target} SELECT * FROM {backup}")


def cleanup_backups(connection):
    with connection.cursor() as cursor:
        for _, backup in BACKUP_TABLES:
            cursor.execute(f"DROP TABLE IF EXISTS {backup}")


def load_json(path):
    if not path.exists():
        raise PipelineError(f"Required diagnostics missing: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _evaluate_single_candidate(candidate, watermarks):
    metrics = load_json(candidate["metrics_path"])
    feature_diagnostics = load_json(
        DEBUG_DIR / "hierarchical_dynamic_features" / "dynamic_diagnostics.json"
    )
    overall = metrics["overall"]
    baseline_mae = float(overall["seasonal_baseline_mae"])
    model_mae = float(overall["model_expected_count_mae"])
    mae_improvement = (baseline_mae - model_mae) / max(baseline_mae, 1e-12)
    true_total = float(overall["avg_true_total_24h"])
    predicted_total = float(overall["avg_pred_total_24h"])
    total_bias = abs(predicted_total - true_total) / max(true_total, 1e-12)
    rolling = metrics.get("rolling_temporal_backtests", [])
    rolling_improvements = [float(item["mae_improvement"]) for item in rolling]
    median_rolling_improvement = (
        float(sorted(rolling_improvements)[len(rolling_improvements) // 2])
        if rolling_improvements else None
    )
    cluster_errors = [
        float(item["absolute_calibration_error"])
        for item in metrics.get("cluster_calibration", [])
    ]
    maximum_cluster_error = max(cluster_errors) if cluster_errors else None

    checks = {
        "full_dataset_build": feature_diagnostics.get("debug_max_cells") is None,
        "leakage_safe_feature_timing": bool(
            feature_diagnostics.get("sequence_ends_before_reference")
            and feature_diagnostics.get("rolling_features_end_before_reference")
            and feature_diagnostics.get("forecast_target_starts_at_reference")
        ),
        "mae_improvement_at_least_threshold": mae_improvement >= MIN_MAE_IMPROVEMENT,
        "three_temporal_backtest_windows_present": len(rolling) >= 3,
        "median_temporal_backtest_improvement_at_least_threshold": (
            median_rolling_improvement is not None
            and median_rolling_improvement >= MIN_MAE_IMPROVEMENT
        ),
        "cluster_calibration_within_threshold": (
            maximum_cluster_error is not None
            and maximum_cluster_error <= MAX_CLUSTER_CALIBRATION_ERROR
        ),
        "average_precision_lift_at_least_threshold": (
            float(overall["average_precision_lift_over_prevalence"]) >= MIN_AP_LIFT
        ),
        "brier_better_than_constant": (
            float(overall["occurrence_brier_score"])
            <= float(overall["constant_prevalence_brier_score"])
        ),
        "top_1pct_recall_at_least_threshold": (
            float(overall["recall_in_top_1pct_block_risks"]) >= MIN_TOP1_RECALL
        ),
        "total_24h_bias_within_threshold": total_bias <= MAX_TOTAL_BIAS,
        "crime_data_fresh_enough": (
            watermarks.get("crime_age_hours") is not None
            and float(watermarks["crime_age_hours"]) <= MAX_CRIME_AGE_HOURS
        ),
    }
    report = {
        "model_family": candidate["model_family"],
        "model_label": candidate["label"],
        "candidate_path": str(candidate["candidate_path"]),
        "active_path": str(candidate["active_path"]),
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "mae_improvement": mae_improvement,
            "average_precision_lift": overall["average_precision_lift_over_prevalence"],
            "top_1pct_recall": overall["recall_in_top_1pct_block_risks"],
            "total_24h_relative_bias": total_bias,
            "crime_age_hours": watermarks.get("crime_age_hours"),
            "median_temporal_backtest_improvement": median_rolling_improvement,
            "maximum_cluster_calibration_error": maximum_cluster_error,
        },
        "thresholds": {
            "minimum_mae_improvement": MIN_MAE_IMPROVEMENT,
            "minimum_average_precision_lift": MIN_AP_LIFT,
            "minimum_top_1pct_recall": MIN_TOP1_RECALL,
            "maximum_total_24h_relative_bias": MAX_TOTAL_BIAS,
            "maximum_crime_age_hours": MAX_CRIME_AGE_HOURS,
            "maximum_cluster_calibration_error": MAX_CLUSTER_CALIBRATION_ERROR,
        },
    }
    return metrics, report


def evaluate_candidate(watermarks):
    reports = []
    passing = []
    for candidate in CANDIDATE_DYNAMIC_MODELS:
        if not candidate["metrics_path"].exists():
            reports.append({
                "model_family": candidate["model_family"],
                "model_label": candidate["label"],
                "candidate_path": str(candidate["candidate_path"]),
                "active_path": str(candidate["active_path"]),
                "passed": False,
                "missing_diagnostics": True,
            })
            continue
        metrics, report = _evaluate_single_candidate(candidate, watermarks)
        reports.append(report)
        if report["passed"]:
            passing.append((metrics, report, candidate))

    if not reports:
        raise PipelineError("No candidate diagnostics were found")

    if not passing:
        return None, {
            "passed": False,
            "selection": "no_candidate_passed",
            "candidates": reports,
        }

    passing.sort(
        key=lambda item: (
            float(item[1]["observed"]["mae_improvement"]),
            float(item[1]["observed"]["average_precision_lift"]),
        ),
        reverse=True,
    )
    metrics, report, candidate = passing[0]
    return metrics, {
        "passed": True,
        "selection": "best_passing_candidate",
        "selected_model_family": candidate["model_family"],
        "selected_model_label": candidate["label"],
        "selected_candidate_path": str(candidate["candidate_path"]),
        "selected_active_path": str(candidate["active_path"]),
        "selected_report": report,
        "candidates": reports,
    }


def last_promoted_run(connection):
    with connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("""
            SELECT finished_at, data_watermarks
            FROM model_pipeline_runs
            WHERE status = 'promoted'
            ORDER BY finished_at DESC NULLS LAST
            LIMIT 1
        """)
        row = cursor.fetchone()
        return dict(row) if row else None


def training_due(connection, watermarks, force):
    if force:
        return True, "forced"
    previous = last_promoted_run(connection)
    if previous is None:
        return True, "no_previous_promoted_run"
    elapsed_days = (
        datetime.now(timezone.utc) - previous["finished_at"]
    ).total_seconds() / 86400.0
    previous_watermark = (previous.get("data_watermarks") or {}).get("crime_latest")
    has_new_data = watermarks.get("crime_latest") != previous_watermark
    if elapsed_days >= RETRAIN_INTERVAL_DAYS and has_new_data:
        return True, "interval_elapsed_with_new_crime_data"
    return False, "not_due_or_no_new_crime_data"


def ingestion_due(connection, force):
    if force:
        return True, "forced"
    with connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("""
            SELECT finished_at
            FROM model_pipeline_runs
            WHERE mode IN ('auto', 'ingest', 'full')
              AND status NOT IN ('running', 'dry_run', 'failed_rolled_back')
              AND finished_at IS NOT NULL
            ORDER BY finished_at DESC
            LIMIT 1
        """)
        previous = cursor.fetchone()
    if previous is None:
        return True, "no_previous_ingestion_run"
    elapsed_hours = (
        datetime.now(timezone.utc) - previous["finished_at"]
    ).total_seconds() / 3600.0
    if elapsed_hours >= INGEST_INTERVAL_HOURS:
        return True, "ingestion_interval_elapsed"
    return False, "ingestion_interval_not_elapsed"


def print_status(connection):
    watermarks = data_watermarks(connection)
    previous = last_promoted_run(connection)
    try:
        _, gate_preview = evaluate_candidate(watermarks)
    except (KeyError, PipelineError, TypeError, ValueError) as error:
        gate_preview = {"passed": False, "error": str(error)}
        print(json.dumps({
            "data_watermarks": watermarks,
            "last_promoted_run": previous,
            "active_dynamic_models": [str(path) for path in ACTIVE_MODEL_FILES[1:]],
            "active_model_exists": any(path.exists() for path in ACTIVE_MODEL_FILES[1:]),
            "current_artifact_gate_preview": gate_preview,
        }, default=str, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("auto", "ingest", "train", "full", "status"),
        default="auto",
    )
    parser.add_argument("--include-osm", action="store_true")
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    connection = connect()
    connection.autocommit = True
    ensure_history_schema(connection)
    if args.mode == "status":
        print_status(connection)
        connection.close()
        return 0

    try:
        acquire_lock(connection)
    except PipelineError as error:
        connection.close()
        if args.mode == "auto" and "already active" in str(error).lower():
            print(str(error))
            return 0
        raise
    run_id = start_run(connection, args.mode)
    run_log_dir = LOG_DIR / f"run_{run_id}"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    stages = []
    backup_dir = None

    try:
        ingestion_plan = list(INGESTION_STAGES)
        if args.include_osm:
            ingestion_plan.append(OSM_STAGE)
        should_ingest = args.mode in {"ingest", "full"}
        ingestion_reason = "explicit_mode"
        if args.mode == "auto":
            should_ingest, ingestion_reason = ingestion_due(
                connection, args.force_ingest
            )

        if args.dry_run:
            print(json.dumps({
                "run_id": run_id,
                "ingestion": ingestion_plan if should_ingest else [],
                "ingestion_decision": ingestion_reason,
                "training": TRAINING_STAGES if args.mode in {"train", "full"} else "auto-decision",
            }, indent=2))
            finish_run(connection, run_id, status="dry_run", stages=stages)
            return 0

        if should_ingest:
            for stage_name, script in ingestion_plan:
                result = run_stage(stage_name, script, run_log_dir)
                stages.append(result)
                update_progress(connection, run_id, stages)
        elif args.mode == "auto":
            print(f"Ingestion skipped: {ingestion_reason}")

        watermarks = data_watermarks(connection)
        update_progress(connection, run_id, stages, watermarks)

        should_train = args.mode in {"train", "full"}
        reason = "explicit_mode"
        if args.mode == "auto":
            should_train, reason = training_due(
                connection, watermarks, args.force_train
            )

        if (
            args.mode == "auto"
            and should_train
            and (
                watermarks.get("crime_age_hours") is None
                or float(watermarks["crime_age_hours"]) > MAX_CRIME_AGE_HOURS
            )
        ):
            finish_run(
                connection, run_id, status="ingested_data_stale_training_skipped",
                stages=stages, data_watermarks=watermarks,
                gate_report={
                    "training_decision": "crime_data_too_stale",
                    "maximum_crime_age_hours": MAX_CRIME_AGE_HOURS,
                },
                log_directory=str(run_log_dir),
            )
            print(
                "Training skipped because the crime watermark is too stale "
                "for a deployable next-24-hour model"
            )
            return 0

        if not should_train:
            finish_run(
                connection, run_id, status=(
                    "ingested_training_skipped" if should_ingest
                    else "no_action_needed"
                ),
                stages=stages, data_watermarks=watermarks,
                gate_report={"training_decision": reason},
                log_directory=str(run_log_dir),
            )
            print(f"Training skipped: {reason}")
            return 0

        backup_dir = backup_training_state(connection, run_id)
        for stage_name, script in TRAINING_STAGES:
            result = run_stage(stage_name, script, run_log_dir)
            stages.append(result)
            update_progress(connection, run_id, stages, watermarks)

        metrics, gate_report = evaluate_candidate(watermarks)
        if not gate_report["passed"]:
            raise PromotionRejected(gate_report)

        selected_candidate = Path(gate_report["selected_candidate_path"])
        selected_active = Path(gate_report["selected_active_path"])
        if not selected_candidate.exists():
            raise PipelineError("Certified candidate checkpoint is missing")
        for path in ACTIVE_MODEL_FILES[1:]:
            if path != selected_active and path.exists():
                path.unlink()
        os.replace(selected_candidate, selected_active)

        cleanup_backups(connection)
        finish_run(
            connection, run_id, status="promoted", stages=stages,
            data_watermarks=watermarks, metrics=metrics,
            gate_report=gate_report, log_directory=str(run_log_dir),
        )
        print("Candidate model passed every gate and was promoted")
        return 0

    except PromotionRejected as error:
        if backup_dir is not None:
            restore_training_state(connection, backup_dir, run_id)
        cleanup_backups(connection)
        finish_run(
            connection, run_id, status="rejected_rolled_back", stages=stages,
            data_watermarks=data_watermarks(connection),
            gate_report=error.gate_report, error=str(error),
            log_directory=str(run_log_dir),
        )
        print(json.dumps(error.gate_report, indent=2))
        return 2
    except Exception as error:
        if backup_dir is not None:
            restore_training_state(connection, backup_dir, run_id)
        cleanup_backups(connection)
        finish_run(
            connection, run_id, status="failed_rolled_back", stages=stages,
            data_watermarks=data_watermarks(connection), error=(
                f"{error}\n{traceback.format_exc()}"
            ), log_directory=str(run_log_dir),
        )
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 1
    finally:
        release_lock(connection)
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
