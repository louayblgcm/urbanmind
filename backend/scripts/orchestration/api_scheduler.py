"""Non-blocking pipeline scheduler owned by the FastAPI application lifecycle."""

import asyncio
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


LOGGER = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent
ORCHESTRATOR = BACKEND_DIR / "scripts" / "orchestration" / "run_pipeline.py"
LOG_DIR = BACKEND_DIR / "data" / "logs" / "pipeline"


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ApiPipelineScheduler:
    def __init__(self):
        self.enabled = _env_bool("PIPELINE_RUN_WITH_API", True)
        self.start_delay_seconds = float(
            os.getenv("PIPELINE_API_START_DELAY_SECONDS", "10")
        )
        self.interval_hours = float(
            os.getenv("PIPELINE_API_INTERVAL_HOURS", "24")
        )
        self._task = None
        self._process = None

    @property
    def status(self):
        return {
            "enabled": self.enabled,
            "running_pipeline": bool(
                self._process is not None and self._process.returncode is None
            ),
            "interval_hours": self.interval_hours,
        }

    def start(self):
        if self.enabled and self._task is None:
            self._task = asyncio.create_task(
                self._run_loop(), name="urbanmind-pipeline-scheduler"
            )
            LOGGER.info("API pipeline scheduler enabled")

    async def stop(self):
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run_loop(self):
        await asyncio.sleep(max(self.start_delay_seconds, 0.0))
        while True:
            await self._launch_auto_pipeline()
            await asyncio.sleep(max(self.interval_hours * 3600.0, 60.0))

    async def _launch_auto_pipeline(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = LOG_DIR / f"api_scheduler_{timestamp}.log"
        creation_flags = (
            subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            self._process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(ORCHESTRATOR),
                "--mode",
                "auto",
                cwd=str(PROJECT_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
            return_code = await self._process.wait()
        if return_code == 0:
            LOGGER.info("Scheduled pipeline check completed")
        elif return_code == 2:
            LOGGER.warning("Candidate model was rejected and rolled back")
        else:
            LOGGER.error(
                "Scheduled pipeline failed with code %s; see %s",
                return_code,
                log_path,
            )
        self._process = None
