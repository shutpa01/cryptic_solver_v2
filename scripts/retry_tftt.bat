@echo off
REM Retry TFTT at 4am UTC — picks up Times puzzles not blogged by 2am
REM Logs to logs\tftt_retry_YYYY-MM-DD.log

cd /d C:\Users\shute\PycharmProjects\cryptic_solver_V2

set PYTHON=C:\Users\shute\PycharmProjects\AI_Solver\.venv\Scripts\python.exe
set LOG_DATE=%date:~6,4%-%date:~3,2%-%date:~0,2%

%PYTHON% scripts\retry_tftt.py >> logs\tftt_retry_%LOG_DATE%.log 2>&1
