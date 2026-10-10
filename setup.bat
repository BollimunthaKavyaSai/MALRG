@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo  NEXUS-REVIEWS AI - SYSTEM SETUP & DEPENDENCY INSTALLER
echo =====================================================================
echo.

:: 1. Check Python
echo [*] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [!] ERROR: Python is not installed or not added to PATH.
    echo     Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
python --version

:: 2. Check Node.js
echo [*] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo [!] ERROR: Node.js is not installed or not added to PATH.
    echo     Please install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)
node --version

:: 3. Check / Create .env
if not exist .env (
    echo [*] Creating .env from .env.example...
    copy .env.example .env
    echo [!] Created .env. Please check and set your GROQ_API_KEY in .env
)

:: 4. Install Python Dependencies
echo.
echo [*] Installing Python dependencies from requirements.txt...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [!] Warning: Some Python packages may have encountered warnings. Continuing...
) else (
    echo [OK] Python dependencies installed successfully.
)

:: 5. Install Server Dependencies
echo.
echo [*] Installing Node.js Express server dependencies...
cd /d "%~dp0server"
call npm install
if errorlevel 1 (
    echo [!] Server npm install failed.
) else (
    echo [OK] Server dependencies installed successfully.
)
cd /d "%~dp0"

:: 6. Install Frontend Dependencies
echo.
echo [*] Installing React Vite frontend dependencies...
cd /d "%~dp0frontend"
call npm install
if errorlevel 1 (
    echo [!] Frontend npm install failed.
) else (
    echo [OK] Frontend dependencies installed successfully.
)
cd /d "%~dp0"

:: 7. Seed Scholarly Corpus
echo.
echo [*] Checking and seeding scholarly database in MongoDB...
cd /d "%~dp0server"
call npm run seed
cd /d "%~dp0"

echo.
echo =====================================================================
echo  [SUCCESS] All dependencies and services configured!
echo  Run 'start.bat' or '.\start.ps1' to launch the entire system.
echo =====================================================================
pause
