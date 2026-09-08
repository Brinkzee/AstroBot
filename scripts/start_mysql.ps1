# Start MySQL container for AstroBot
Write-Host "Starting AstroBot MySQL container..." -ForegroundColor Cyan

if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker compose up -d
} elseif (Get-Command wsl -ErrorAction SilentlyContinue) {
    wsl -d Ubuntu docker compose -f /mnt/d/PycharmProjects/AstroBot/docker-compose.yml up -d
} else {
    Write-Error "Neither docker nor wsl command found on PATH."
    exit 1
}

# Wait for MySQL port 3306
Write-Host "Waiting for MySQL to accept connections on port 3306..." -ForegroundColor Yellow
$maxRetries = 20
$ready = $false
for ($i = 0; $i -lt $maxRetries; $i++) {
    Start-Sleep -Seconds 1
    $tcp = Test-NetConnection -ComputerName 127.0.0.1 -Port 3306 -WarningAction SilentlyContinue
    if ($tcp.TcpTestSucceeded) {
        $ready = $true
        break
    }
}

if ($ready) {
    Write-Host "MySQL container started successfully on port 3306." -ForegroundColor Green
} else {
    Write-Warning "MySQL container started, but port 3306 is not responding yet. Please check container logs."
}
