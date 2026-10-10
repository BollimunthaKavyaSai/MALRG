# Nexus-Reviews AI - PowerShell Startup Script
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " NEXUS-REVIEWS AI - MULTI-AGENT SCHOLARLY SYSTEM LAUNCHER" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptPath

# 1. MongoDB check
Write-Host "[*] Checking MongoDB connection on port 27017..." -ForegroundColor Yellow
try {
    $tcp = New-Object System.Net.Sockets.TcpClient("127.0.0.1", 27017)
    $tcp.Close()
    Write-Host "[OK] MongoDB is active on port 27017." -ForegroundColor Green
} catch {
    Write-Host "[!] MongoDB not responding on 127.0.0.1:27017. Attempting start..." -ForegroundColor Red
    Start-Service -Name MongoDB -ErrorAction SilentlyContinue
}

# 2. Start Python FastAPI
Write-Host "[*] Starting Python FastAPI AI Engine on port 8000..." -ForegroundColor Yellow
Start-Process -FilePath "cmd.exe" -ArgumentList "/k cd /d `"$scriptPath`" && python -m uvicorn app:app --host 127.0.0.1 --port 8000"

# 3. Start Node.js Express
Write-Host "[*] Starting Node.js Express Orchestrator on port 3000..." -ForegroundColor Yellow
Start-Process -FilePath "cmd.exe" -ArgumentList "/k cd /d `"$scriptPath\server`" && node index.js"

# 4. Start React Vite
Write-Host "[*] Starting React Vite Frontend on port 5173..." -ForegroundColor Yellow
Start-Process -FilePath "cmd.exe" -ArgumentList "/k cd /d `"$scriptPath\frontend`" && npm run dev -- --host 127.0.0.1 --port 5173"

# 5. Wait for services
Start-Sleep -Seconds 5

# 6. Launch browser
Write-Host "[*] Launching user interface in browser..." -ForegroundColor Green
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " [SYSTEM ONLINE] All 3 microservices are running:" -ForegroundColor Green
Write-Host "   * React Frontend:     http://localhost:5173" -ForegroundColor White
Write-Host "   * Node.js Server:     http://127.0.0.1:3000" -ForegroundColor White
Write-Host "   * Python FastAPI:     http://127.0.0.1:8000" -ForegroundColor White
Write-Host "   * Health Check:       http://127.0.0.1:3000/api/health" -ForegroundColor White
Write-Host "=====================================================================" -ForegroundColor Cyan
