@echo off
REM Start the AMPL Chatbot API server on Windows.
REM
REM Usage:
REM   scripts\run_api.bat          - production mode
REM   scripts\run_api.bat --dev    - development with hot reload

cd /d "%~dp0\.."

if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "%%A=%%B"
    )
)

if not defined API_HOST set API_HOST=0.0.0.0
if not defined API_PORT set API_PORT=8000
if not defined LOG_LEVEL set LOG_LEVEL=info

if "%~1"=="--dev" (
    echo Starting AMPL Chatbot API in development mode...
    uvicorn api.main:app --host %API_HOST% --port %API_PORT% --reload --log-level %LOG_LEVEL%
) else (
    echo Starting AMPL Chatbot API...
    uvicorn api.main:app --host %API_HOST% --port %API_PORT% --workers 4 --log-level %LOG_LEVEL%
)
