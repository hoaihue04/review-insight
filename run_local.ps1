# Chay du an khong can Docker.
# .env dang tro toi ten may cua Docker ("postgres", "redis"). Script nay doi
# hai dia chi do sang localhost CHI cho lan chay nay, khong sua file .env,
# nen "docker compose up" van dung duoc nhu cu.
#
#   .\run_local.ps1            # http://127.0.0.1:8001
#   .\run_local.ps1 -Port 9000
#   .\run_local.ps1 -Online    # cho phep tai model tu huggingface.co (lan dau, hoac doi model)

param(
    [int]$Port = 8001,
    [string]$BindHost = "127.0.0.1",
    [switch]$Online
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Chua co .venv. Tao bang:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    Write-Host "  .\.venv\Scripts\python.exe -m pip install -e ."
    exit 1
}
if (-not (Test-Path ".env")) {
    Write-Host "Chua co file .env. Tao bang: copy .env.example .env" -ForegroundColor Yellow
    exit 1
}

function Get-DotEnvValue([string]$name) {
    $line = Get-Content ".env" -Encoding UTF8 | Where-Object { $_ -match "^\s*$name\s*=" } | Select-Object -First 1
    if ($null -eq $line) { return "" }
    return ($line -replace "^\s*$name\s*=\s*", "").Trim().Trim('"').Trim("'")
}

$dbUrl = Get-DotEnvValue "DATABASE_URL"
if ($dbUrl) { $env:DATABASE_URL = $dbUrl -replace "@postgres([:/])", '@localhost$1' }

$redisUrl = Get-DotEnvValue "REDIS_URL"
if ($redisUrl) { $env:REDIS_URL = $redisUrl -replace "//redis([:/])", '//localhost$1' }

$env:PYTHONUTF8 = "1"

# Model (bge-m3, bge-reranker, PhoBERT...) da co trong cache cua may. Moi lan nap model,
# thu vien van hoi huggingface.co de kiem tra phien ban; mang cham thi moi file cho 10s
# va thu lai 5 lan, lam phan tich vuot qua 5-10 phut. Che do offline chi doc tu cache.
if (-not $Online) {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    Write-Host "Nap model tu cache (offline). Dung -Online neu can tai model moi." -ForegroundColor DarkGray
}
Write-Host "Model embedding/reranker duoc nap san o nen, mat 1-2 phut sau khi server chay (xem dong [warmup])." -ForegroundColor DarkGray

Write-Host "Mo trinh duyet tai http://$($BindHost):$Port  (kiem tra ket noi: /api/health)" -ForegroundColor Green
& $python -m uvicorn app.main:app --host $BindHost --port $Port
