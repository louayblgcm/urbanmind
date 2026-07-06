# Urban Mind

Urban Mind is a full-stack urban intelligence project that combines spatial data engineering, machine learning, and an interactive map interface.

The project ingests public city datasets, builds long-term neighborhood profiles, and serves a next-24-hour crime-pressure forecast together with urban context signals like transit, nightlife, workplace activity, and 311 pressure.

## Why this project exists

I built this project because I wanted to see whether urban data could be turned into something more useful than a set of dashboards. I was especially interested in combining:

- geospatial preprocessing
- data ingestion automation
- graph-based static modeling
- dynamic forecasting
- LLM-assisted natural-language summaries
- a map-first frontend that makes the outputs easier to explore

## Tech stack

- Backend: FastAPI, PostgreSQL, Psycopg2, PyTorch
- Frontend: Next.js, React, TypeScript, Tailwind-style utility classes
- Data sources: Chicago open data APIs, 311 records, OSM-derived context
- LLM layer: Groq OpenAI-compatible chat completions

## Project structure

```text
urbanmindv2/
|-- backend/
|   |-- api/                 # FastAPI app
|   |-- config/              # environment-driven configuration
|   |-- models/              # reusable model definitions
|   |-- scripts/
|   |   |-- cognition/       # cell-level retrieval pipeline
|   |   |-- context/         # reverse geocoding + OSM/context assembly
|   |   |-- ingestion/       # API/database ingestion jobs
|   |   |-- intelligence/    # static + dynamic modeling and forecast logic
|   |   |-- llm/             # Groq client + prompt builder
|   |   `-- orchestration/   # scheduled ingestion/training pipeline
|   |-- .env.example
|   `-- requirements.txt
|-- frontend/
|   |-- app/                 # Next.js app router UI
|   |-- public/
|   |-- types/
|   `-- .env.example
|-- main.py                  # local backend entrypoint
`-- README.md
```

## Core system design

The system has two modeling layers:

1. Static intelligence
   - Builds leakage-safe cell features from historical data
   - Trains a static graph model to estimate persistent neighborhood risk and activity structure

2. Dynamic intelligence
   - Uses recent crime + 311 sequences, neighborhood context, and static baselines
   - Produces a next-24-hour advisory forecast at a coarser spatiotemporal resolution

The API combines those outputs into:

- map metrics
- overview narratives
- chat responses
- timeline and forecast visualizations

## How the training pipeline works

The training setup is basically split into two stages.

First, the static pipeline builds the long-run profile of each area. It takes historical city data, turns it into cell-level features, and trains a static graph model. That gives each cell a stable baseline, like its general crime pressure, activity structure, and neighborhood context.

Then the dynamic pipeline builds on top of that baseline instead of starting from zero. It uses recent crime and 311 sequences together with the static outputs to train a short-term forecasting model for the next 24 hours.

In simple terms, the flow is:

- ingest city data into PostgreSQL
- build static cell features
- train the static GNN and save the area profiles
- build dynamic training tensors using recent sequences plus static features
- train the dynamic forecaster
- store processed outputs and cached forecasts so the API can respond quickly

At runtime, the app uses both layers together:

- the static model answers what an area is usually like
- the dynamic model estimates how the next day may differ from that usual pattern

## Running locally

### 1. Backend

Create `backend/.env` from `backend/.env.example`, then install backend dependencies.

Run the API from the project root:

```powershell
python main.py
```

Or directly from the backend:

```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

### 2. Frontend

Create `frontend/.env.local` from `frontend/.env.example`.

Then run:

```powershell
cd frontend
npm install
npm run dev
```

## Data + training pipeline

The backend includes an orchestration pipeline for ingestion and model refreshes, so the whole flow can be rerun in a structured way.

From `backend/`:

```powershell
.\run_pipeline.cmd --mode auto
```

More details are documented in [backend/ORCHESTRATION.md](backend/ORCHESTRATION.md).

## What I learned

This project taught me that the hard part is not only training a model. The harder part is keeping ingestion, feature engineering, training, API responses, and frontend presentation aligned so the whole system still makes sense end to end.

It also pushed me to think more carefully about uncertainty. With urban data, especially short-term crime forecasting, it is easy to produce numbers that look precise without actually being reliable enough. A lot of the work here was about building safer baselines, checking model behavior honestly, and making the UI communicate that clearly.

## Notes on the current forecast

The dynamic forecast is currently served as an advisory model rather than a fully certified production forecaster.

That is intentional and explicit in the API payload:

- the static baseline remains the most stable reference
- the dynamic model is exposed because it adds non-baseline movement
- upstream crime-feed freshness can still limit reliability

## Current limitations

- Forecast quality still depends on how fresh the upstream crime data is.
- Very fine-grained prediction is difficult because the data is sparse in both space and time.
- The dynamic model is shown as an advisory layer, not as a guaranteed production-grade forecaster.
- The strongest part of the system right now is the overall pipeline design and integration, not the claim of perfect prediction.

## What this repo is meant to show

This repository is mainly a portfolio project. The parts I think are strongest are:

- end-to-end system design
- data pipeline thinking
- geospatial feature engineering
- ML experimentation and evaluation discipline
- API and UI integration across a fairly complex stack

## Author

Louay - student builder focused on practical AI, data systems, and urban intelligence workflows.
