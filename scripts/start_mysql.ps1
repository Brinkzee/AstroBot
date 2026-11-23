# Start MySQL container for AstroBot
Write-Host "Starting AstroBot MySQL container..." -ForegroundColor Cyan

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $RepoRoot "docker-compose.yml"

if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker compose -f $ComposeFile up -d
} elseif (Get-Command wsl -ErrorAction SilentlyContinue) {
    $drive = $ComposeFile.Substring(0, 1).ToLower()
    $rest = $ComposeFile.Substring(2) -replace '\\', '/'
    $wslComposePath = "/mnt/$drive$rest"
    wsl -d Ubuntu docker compose -f $wslComposePath up -d
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
