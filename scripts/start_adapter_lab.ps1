param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python environment not found: $pythonPath" }
$env:PYTHONPATH = Join-Path $projectRoot '.adapter_compare_deps'
$env:HF_HOME = Join-Path $env:USERPROFILE '.cache\huggingface'
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONUNBUFFERED = '1'
$studioUrl = 'http://127.0.0.1:8765'
$running = $false
try {
    $status = Invoke-RestMethod -Uri "$studioUrl/api/status" -TimeoutSec 3
    $running = $status.base_revision -eq 'c1899de289a04d12100db370d81485cdf75e47ca'
} catch { }
if (-not $running) {
    $logDirectory = Join-Path $projectRoot 'results'
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $serverPath = Join-Path $PSScriptRoot 'adapter_compare_server.py'
    $serverProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-u', ('"' + $serverPath + '"')) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory 'adapter_lab_server.log') -RedirectStandardError (Join-Path $logDirectory 'adapter_lab_server_errors.log')
    $serverProcess.Id | Set-Content -LiteralPath (Join-Path $logDirectory 'adapter_lab_server.pid')
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 1
        if ($serverProcess.HasExited) { throw 'Runner exited. See results\adapter_lab_server_errors.log.' }
        try {
            $status = Invoke-RestMethod -Uri "$studioUrl/api/status" -TimeoutSec 2
            if ($status.base_revision -eq 'c1899de289a04d12100db370d81485cdf75e47ca') { $running = $true; break }
        } catch { }
    }
    if (-not $running) { throw 'Runner did not respond. See results\adapter_lab_server_errors.log.' }
}
Write-Output "Adapter Lab ready: $studioUrl"
if (-not $NoBrowser) { Start-Process $studioUrl }
