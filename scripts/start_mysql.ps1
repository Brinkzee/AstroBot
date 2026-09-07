# Start MySQL container for AstroBot
Write-Host "Starting AstroBot MySQL container..." -ForegroundColor Cyan

if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker compose up -d
} elseif (Get-Command wsl -ErrorAction SilentlyContinue) {
    wsl docker compose up -d
} else {
    Write-Error "Neither docker nor wsl command found on PATH."
    exit 1
}

Write-Host "MySQL container started successfully on port 3306." -ForegroundColor Green
