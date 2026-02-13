@echo off
REM Initial setup script for AMPL Chatbot on Windows.

cd /d "%~dp0\.."

echo === AMPL Chatbot Setup ===

REM 1. Virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

REM 2. Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM 3. Create .env
if not exist ".env" (
    echo Creating .env from .env.example...
    copy .env.example .env
    echo >>> Edit .env with your API keys before running <<<
)

REM 4. Create data directory
if not exist "data" mkdir data

echo.
echo === Setup complete ===
echo.
echo Next steps:
echo   1. Edit .env with your API keys
echo   2. Ingest data:    scripts\ingest_data.bat defaults
echo   3. Start the API:  scripts\run_api.bat --dev
echo   4. Open widget:    start frontend\widget\index.html
echo.
