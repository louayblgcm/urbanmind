@echo off
set "BACKEND_DIR=%~dp0"
"%BACKEND_DIR%..\venv\Scripts\python.exe" "%BACKEND_DIR%scripts\orchestration\run_pipeline.py" %*
exit /b %ERRORLEVEL%
