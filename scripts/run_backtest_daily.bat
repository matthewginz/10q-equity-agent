@echo off
REM Task Scheduler entrypoint -- daily resumable backtest chunk.
REM Gemini's free tier caps each model per day, so each run scores as many
REM filings as today's quota allows, then refreshes the Wall Street comparison.
REM backtest_accuracy.py holds a lock file, so an overlapping run exits at once.
REM Finally publishes the new results to GitHub so the live app updates itself.
cd /d "%~dp0\.."
echo ===== %date% %time% ===== >> logs\backtest.log
venv\Scripts\python.exe -u scripts\backtest_accuracy.py --require-backtest-key >> logs\backtest.log 2>&1
venv\Scripts\python.exe -u scripts\analyst_comparison.py >> logs\backtest.log 2>&1
call scripts\publish_backtest.bat >> logs\backtest.log 2>&1
call scripts\nightly_notes.bat >> logs\backtest.log 2>&1
