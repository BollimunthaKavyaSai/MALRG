@echo off
setlocal enabledelayedexpansion

title Nexus-Reviews AI Launcher
echo =====================================================================
echo  NEXUS-REVIEWS AI - MULTI-AGENT SCHOLARLY SYSTEM LAUNCHER
echo =====================================================================
echo.

cd /d "%~dp0"

:: 1. Verify MongoDB connectivity or start MongoDB if service exists
echo [*] Checking MongoDB connection on port 27017...
powershell -Command "try { $client = New-Object System.Net.Sockets.TcpClient('127.0.0.1', 27017); $client.Close(); exit 0 } catch { exit 1 }"
if errorlevel 1 (
    echo [!] WARNING: MongoDB does not appear to be running on 127.0.0.1:27017.
    echo     Attempting to start MongoDB service or daemon...
    net start MongoDB >nul 2>&1
)

:: 2. Launch Python FastAPI AI Engine (Port 8000)
echo [*] Starting Python FastAPI AI Engine on port 8000...
start "Nexus AI - Python Engine (Port 8000)" cmd /k "cd /d "%~dp0" && python -m uvicorn app:app --host 127.0.0.1 --port 8000"

:: 3. Launch Node.js Express Orchestrator (Port 3000)
echo [*] Starting Node.js Express Orchestrator on port 3000...
start "Nexus AI - Node Server (Port 3000)" cmd /k "cd /d "%~dp0server" && node index.js"

:: 4. Launch React Vite Frontend (Port 5173)
echo [*] Starting React Vite Frontend on port 5173...
start "Nexus AI - Frontend (Port 5173)" cmd /k "cd /d "%~dp0frontend" && npm run dev -- --host 127.0.0.1 --port 5173"

:: 5. Wait for services to bind ports
echo.
echo [*] Initializing services, please wait 5 seconds...
timeout /t 5 /nobreak >nul

:: 6. Launch browser
echo [*] Launching user interface in default web browser...
start http://localhost:5173

echo.
echo =====================================================================
echo  [SYSTEM ONLINE] All 3 microservices are running:
echo.
echo   * React Frontend:     http://localhost:5173
echo   * Node.js Server:     http://127.0.0.1:3000
echo   * Python FastAPI:     http://127.0.0.1:8000
echo   * API Health Check:   http://127.0.0.1:3000/api/health
echo   * Corpus Stats:       http://127.0.0.1:3000/api/corpus/stats
echo   * Ablation Benchmark: http://127.0.0.1:3000/api/evaluation/benchmark
echo.
echo  To shut down all services cleanly, run 'stop.bat'.
echo =====================================================================
echo.
pause
