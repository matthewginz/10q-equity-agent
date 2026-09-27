@echo off
REM Publish the backtest data files so the Streamlit Cloud app (deploys from
REM GitHub main) shows the new results. All the logic, checks and the live
REM verification are in publish_backtest.py; this wrapper is what Task Scheduler
REM and run_backtest_daily.bat call.
cd /d "%~dp0\.."
venv\Scripts\python.exe -u scripts\publish_backtest.py
