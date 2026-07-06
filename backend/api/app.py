"""FastAPI application exposing UrbanMind intelligence endpoints."""

import logging
from contextlib import asynccontextmanager

import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from psycopg2.extras import RealDictCursor

from config.config import CORS_ORIGINS, DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from scripts.cognition.cognition_retriever import CognitionRetriever
from scripts.context.context_builder import ContextBuilder
from scripts.context.context_cache import build_context_id, get_cached_context, set_cached_context
from scripts.intelligence.metrics_engine import MetricsEngine
from scripts.intelligence.semantic_engine import SemanticEngine
from scripts.llm.groq_client import generate_groq_response, groq_is_configured
from scripts.llm.prompt_builder import PromptBuilder
from scripts.orchestration.api_scheduler import ApiPipelineScheduler


LOGGER = logging.getLogger(__name__)
pipeline_scheduler = ApiPipelineScheduler()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    pipeline_scheduler.start()
    try:
        yield
    finally:
        await pipeline_scheduler.stop()


app = FastAPI(
    title="UrbanMind V2 Intelligence API", version="2.0", lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = CognitionRetriever()
context_builder = ContextBuilder(retriever)
metrics_engine = MetricsEngine()
semantic_engine = SemanticEngine()
prompt_builder = PromptBuilder()


class AreaRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class ChatRequest(BaseModel):
    context_id: str
    question: str = Field(min_length=1, max_length=1000)


def build_pipeline(lat, lon):
    """Build the full intelligence payload for a map location."""
    try:
        cognition = retriever.build_cognition_packet(lat, lon)
        context = context_builder.build_context(lat, lon, cognition_packet=cognition)
        metrics = metrics_engine.build_metrics(cognition, context=context)
        semantic = semantic_engine.build_semantic_profile(metrics, cognition)
        # Apply the human-readable trained profile after semantics are available.
        metrics = metrics_engine.build_metrics(cognition, semantic, context)
        return {
            "cognition_packet": cognition,
            "semantic_profile": semantic,
            "context": context,
            "metrics": metrics,
            "raw_activity_feed": {
                "crimes": cognition.get("recent_crimes", []),
                "requests_311": cognition.get("recent_311", []),
            },
        }
    except (
        psycopg2.Error, FileNotFoundError, LookupError, RuntimeError, ValueError
    ) as error:
        LOGGER.exception("Area pipeline failed")
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/area-metrics")
def area_metrics(request: AreaRequest):
    pipeline = build_pipeline(request.lat, request.lon)
    return {
        "metrics": pipeline["metrics"],
        "semantic_profile": pipeline["semantic_profile"],
        "context": pipeline["context"],
    }


@app.post("/area-overview")
def area_overview(request: AreaRequest):
    context_id = build_context_id(request.lat, request.lon)
    pipeline = get_cached_context(context_id)
    if pipeline is None:
        pipeline = build_pipeline(request.lat, request.lon)
        set_cached_context(context_id, pipeline)

    prompt = prompt_builder.build_overview_prompt(
        context=pipeline["context"],
        metrics=pipeline["metrics"],
        semantic_profile=pipeline["semantic_profile"],
    )
    return {
        "context_id": context_id,
        "overview": generate_groq_response(prompt),
        "metrics": pipeline["metrics"],
        "semantic_profile": pipeline["semantic_profile"],
        "context": pipeline["context"],
        "raw_activity_feed": pipeline["raw_activity_feed"],
    }


@app.post("/area-chat")
def area_chat(request: ChatRequest):
    pipeline = get_cached_context(request.context_id)
    if pipeline is None:
        raise HTTPException(status_code=410, detail="Context expired")
    prompt = prompt_builder.build_chat_prompt(
        question=request.question,
        context=pipeline["context"],
        metrics=pipeline["metrics"],
        semantic_profile=pipeline["semantic_profile"],
    )
    return {"response": generate_groq_response(prompt)}


@app.get("/grid-cells")
def grid_cells():
    try:
        with psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT,
        ) as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT p.cell_id,
                       split_part(p.cell_id, '_', 1)::DOUBLE PRECISION * 0.00135 AS lat_cell,
                       split_part(p.cell_id, '_', 2)::DOUBLE PRECISION * 0.00165 AS lon_cell,
                       GREATEST(0, LEAST(100, 100 - p.static_crime_score)) AS safety_score,
                       CASE WHEN f.crime_total > 0
                            THEN f.vehicle_crime_count / f.crime_total ELSE 0 END AS vehicle_theft_risk,
                       CASE WHEN f.crime_total > 0 THEN
                           (f.baseline_crime_hour_20 + f.baseline_crime_hour_21
                           + f.baseline_crime_hour_22 + f.baseline_crime_hour_23
                           + f.baseline_crime_hour_00 + f.baseline_crime_hour_01
                           + f.baseline_crime_hour_02 + f.baseline_crime_hour_03
                           + f.baseline_crime_hour_04 + f.baseline_crime_hour_05)
                           / NULLIF((SELECT SUM(value) FROM unnest(ARRAY[
                               f.baseline_crime_hour_00, f.baseline_crime_hour_01,
                               f.baseline_crime_hour_02, f.baseline_crime_hour_03,
                               f.baseline_crime_hour_04, f.baseline_crime_hour_05,
                               f.baseline_crime_hour_06, f.baseline_crime_hour_07,
                               f.baseline_crime_hour_08, f.baseline_crime_hour_09,
                               f.baseline_crime_hour_10, f.baseline_crime_hour_11,
                               f.baseline_crime_hour_12, f.baseline_crime_hour_13,
                               f.baseline_crime_hour_14, f.baseline_crime_hour_15,
                               f.baseline_crime_hour_16, f.baseline_crime_hour_17,
                               f.baseline_crime_hour_18, f.baseline_crime_hour_19,
                               f.baseline_crime_hour_20, f.baseline_crime_hour_21,
                               f.baseline_crime_hour_22, f.baseline_crime_hour_23
                           ]) AS u(value)), 0)
                            ELSE 0 END AS night_risk_ratio,
                       p.static_activity_score AS relative_density
                FROM static_gnn_profiles p
                JOIN static_cell_features f ON f.cell_id = p.cell_id
                ORDER BY p.cell_id
                """
            )
            return [dict(row) for row in cursor.fetchall()]
    except psycopg2.Error as error:
        raise HTTPException(status_code=503, detail="Grid data unavailable") from error


@app.get("/")
def root():
    return {
        "status": "Urban Intelligence API Online",
        "groq_configured": groq_is_configured(),
        "forecast_model": "hierarchical_forecaster_500m_6h_neural",
        "forecast_mode": "dynamic_advisory",
        "pipeline_scheduler": pipeline_scheduler.status,
    }
