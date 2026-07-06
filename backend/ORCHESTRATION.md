# UrbanMind data and model pipeline

Run this command daily from the backend directory. The `.cmd` launcher works
even when PowerShell script execution is disabled:

```powershell
.\run_pipeline.cmd --mode auto
```

When the API is started with Uvicorn, the same `auto` pipeline is launched in
a separate background process after 10 seconds and checked every 24 hours.
The API remains responsive during ingestion and training. PostgreSQL advisory
locking prevents duplicate runs after Uvicorn reloads. Set
`PIPELINE_RUN_WITH_API=false` to disable this behavior.

`auto` always refreshes crime, 311, traffic, business, and permit data. It
rebuilds both models only when seven days have elapsed since the last promoted
run and the crime watermark has changed. OSM uses the bundled snapshot and is
included only when explicitly requested:

```powershell
.\run_pipeline.cmd --mode full --include-osm
```

Other useful commands:

```powershell
.\run_pipeline.cmd --mode status
.\run_pipeline.cmd --mode ingest
.\run_pipeline.cmd --mode train
.\run_pipeline.cmd --mode auto --force-train
.\run_pipeline.cmd --mode full --dry-run
```

## Promotion requirements

A candidate replaces the active model only when all checks pass:

- sequence and rolling features end before the forecast target;
- expected-count MAE improves at least 3% over the trained static baseline;
- average-precision lift is at least 3x prevalence;
- Brier score beats the constant-prevalence predictor;
- top-1% risk recall is at least 10%;
- total 24-hour bias is no more than 10%;
- the crime watermark is no more than 48 hours old.

On a failed stage or gate, the previous checkpoints, `static_cell_features`,
and `static_gnn_profiles` are restored automatically. Runs and gate reports are
stored in PostgreSQL table `model_pipeline_runs`; logs are written under
`backend/data/logs/pipeline`.

Use Windows Task Scheduler to execute `run_pipeline.cmd --mode auto` once per
day. Scheduling is intentionally not installed automatically because the run
time and machine account are deployment choices.
