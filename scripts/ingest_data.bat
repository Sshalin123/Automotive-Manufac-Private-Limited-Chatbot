@echo off
REM Run data ingestion for AMPL Chatbot on Windows.
REM
REM Usage:
REM   scripts\ingest_data.bat                        - ingest all
REM   scripts\ingest_data.bat inventory data\inventory.csv
REM   scripts\ingest_data.bat faq data\faqs.json
REM   scripts\ingest_data.bat defaults

cd /d "%~dp0\.."

if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "%%A=%%B"
    )
)

set SOURCE=%~1
if "%SOURCE%"=="" set SOURCE=all

if "%SOURCE%"=="all" (
    echo Ingesting all data from .\data ...
    python -m ingest_ampl.main --source all --dir ./data
) else if "%SOURCE%"=="defaults" (
    echo Ingesting default FAQs...
    python -m ingest_ampl.main --source defaults
) else (
    echo Ingesting %SOURCE%: %~2
    python -m ingest_ampl.main --source %SOURCE% --file %~2
)

echo Ingestion complete.
