@echo off
REM Nightly crossword scrape + solve — runs at 2am UTC via Task Scheduler
REM Logs to logs\nightly_YYYY-MM-DD.log

cd /d C:\Users\shute\PycharmProjects\cryptic_solver_V2

set PYTHON=C:\Users\shute\PycharmProjects\AI_Solver\.venv\Scripts\python.exe
set LOG_DATE=%date:~6,4%-%date:~3,2%-%date:~0,2%

%PYTHON% scripts\nightly_run.py >> logs\nightly_%LOG_DATE%.log 2>&1
