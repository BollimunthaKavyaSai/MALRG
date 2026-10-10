# Auto-resume driver for the MALRG ablation study.
# Re-runs the (fully cached, resumable) study until every topic x configuration is generated
# and evaluated, waiting for the Groq daily token quota to free up between attempts.
# Start:  powershell -ExecutionPolicy Bypass -File auto_resume.ps1
# Logs:   logs\auto_resume.log (driver)  and  logs\run_<n>.log (each attempt)

$ErrorActionPreference = "Continue"
$suite = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $suite
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force "$suite\logs" | Out-Null
$driverLog = "$suite\logs\auto_resume.log"
$expected = 140         # ablation 60 + baselines 20 + prompts 50 (5 prompts x 10 topics) + preferences 10
$waitMinutes = 30
$maxAttempts = 400      # ~8 days at 30-minute intervals

function Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg" | Out-File -Append -Encoding utf8 $driverLog }

function Count-Evaluated {
    (Get-ChildItem "$suite\runs" -Recurse -Filter "??_eval.json" -ErrorAction SilentlyContinue |
        Where-Object { (Get-Content $_.FullName -Raw) -match '"judge_done": true' }).Count
}

Log "auto-resume started (PID $PID)"
for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    $runLog = "$suite\logs\run_$attempt.log"
    Log "attempt $attempt -> $runLog"
    # cmd redirection keeps the log in UTF-8 (PowerShell 5.1 '*>' would write UTF-16)
    & cmd.exe /c "python -u run_all.py all > `"$runLog`" 2>&1"
    $code = $LASTEXITCODE
    $done = Count-Evaluated
    Log "attempt $attempt finished: exit=$code evaluated=$done/$expected"
    if ($code -eq 0 -and $done -ge $expected) {
        Log "study complete; tables, figures and report written to results\"
        break
    }
    Start-Sleep -Seconds ($waitMinutes * 60)
}
Log "auto-resume stopped"
