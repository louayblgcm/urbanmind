param(
    [ValidateSet("auto", "ingest", "train", "full", "status")]
    [string]$Mode = "auto",
    [switch]$IncludeOsm,
    [switch]$ForceIngest,
    [switch]$ForceTrain,
    [switch]$DryRun
)

$BackendDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $BackendDir "..\venv\Scripts\python.exe"
$Orchestrator = Join-Path $BackendDir "scripts\orchestration\run_pipeline.py"

$Arguments = @($Orchestrator, "--mode", $Mode)
if ($IncludeOsm) { $Arguments += "--include-osm" }
if ($ForceIngest) { $Arguments += "--force-ingest" }
if ($ForceTrain) { $Arguments += "--force-train" }
if ($DryRun) { $Arguments += "--dry-run" }

& $Python @Arguments
exit $LASTEXITCODE
