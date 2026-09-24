@echo off
REM Commit + push ONLY the backtest data files, so the Streamlit Cloud app
REM (which deploys from GitHub main) picks up each night's new results.
REM "--only" commits just these paths: any code you have staged or modified
REM elsewhere in the repo is left untouched and never pushed by this script.
cd /d "%~dp0\.."
set FILES=docs/backtest/analyst_comparison.csv docs/backtest/progress.json docs/backtest/raw_runs.json docs/backtest/results.csv

git diff --quiet -- %FILES%
if %errorlevel%==0 (
    echo publish: no new backtest results, nothing to push
    exit /b 0
)

for /f "delims=" %%n in ('venv\Scripts\python.exe -c "import json;print(json.load(open('docs/backtest/progress.json'))['filings_done'])"') do set DONE=%%n
git commit --only -m "data: backtest snapshot, %DONE% filings scored (auto)" -- %FILES%
if errorlevel 1 (
    echo publish: commit failed
    exit /b 1
)
git push origin main
if errorlevel 1 (
    echo publish: push failed -- commit kept locally, will go up with the next successful push
    exit /b 1
)
echo publish: pushed %DONE% filings
