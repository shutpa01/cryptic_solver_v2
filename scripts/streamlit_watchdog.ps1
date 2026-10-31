# Keeps the Streamlit dashboard (:8501) running so the Tailscale 10000 phone link always has a
# server behind it (user, 2026-10-31: found not running in Barcelona, nothing to restart it).
# Run every 5 minutes by the CrypticSolver_StreamlitDev scheduled task. Starts Streamlit only
# when nothing is listening on 8501, so it never disturbs a running dashboard. Launched from
# PowerShell, never Git Bash, so the dashboard's deploy ssh is Windows OpenSSH.
$repo = "C:\Users\shute\PycharmProjects\cryptic_solver_V2"
$streamlit = "C:\Users\shute\PycharmProjects\AI_Solver\.venv\Scripts\streamlit.exe"
$log = "$repo\logs\streamlit_dev.log"

if (Get-NetTCPConnection -State Listen -LocalPort 8501 -ErrorAction SilentlyContinue) { exit 0 }

Add-Content -Path "$repo\logs\streamlit_watchdog.log" -Value ("{0}  :8501 not listening - starting Streamlit" -f (Get-Date -Format s))
Set-Location $repo
& $streamlit run "dashboard/app.py" --server.port 8501 --server.headless true *>> $log
