@echo off
echo =====================================================================
echo  Shutting down Nexus-Reviews AI services...
echo =====================================================================
echo.

:: Kill port 8000 (Python uvicorn)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo [*] Stopping Python FastAPI process %%a on port 8000...
    taskkill /F /PID %%a >nul 2>&1
)

:: Kill port 3000 (Node Express)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000" ^| findstr "LISTENING"') do (
    echo [*] Stopping Node.js Express process %%a on port 3000...
    taskkill /F /PID %%a >nul 2>&1
)

:: Kill port 5173 (Vite Frontend)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173" ^| findstr "LISTENING"') do (
    echo [*] Stopping React Vite process %%a on port 5173...
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo [OK] All Nexus-Reviews AI services stopped.
echo =====================================================================
pause
