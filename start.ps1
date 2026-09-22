# AstroBot 一键启动快捷脚本 (PowerShell)
# 自动适配控制台编码并调用 run.py 执行环境检查与拉起主服务

[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "        🚀 AstroBot 智能电商客服系统一键启动程序 🚀" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python 环境：优先使用项目虚拟环境
$pythonExe = $null
if (Test-Path ".venv\Scripts\python.exe") {
    $pythonExe = ".venv\Scripts\python.exe"
    Write-Host "[环境] 优先使用当前项目虚拟环境: .venv" -ForegroundColor Green
} elseif (Test-Path "venv\Scripts\python.exe") {
    $pythonExe = "venv\Scripts\python.exe"
    Write-Host "[环境] 优先使用当前项目虚拟环境: venv" -ForegroundColor Green
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = "py"
}

if (-not $pythonExe) {
    Write-Host "[错误] 未在系统 PATH 或当前项目中检测到 Python 环境！" -ForegroundColor Red
    Write-Host "请先安装 Python 3.10+ 并勾选 'Add Python to PATH'，或创建虚拟环境 .venv。" -ForegroundColor Yellow
    Read-Host "按回车键退出..."
    exit 1
}

Write-Host "[启动] 正在启动 AstroBot 服务..." -ForegroundColor Cyan
Write-Host ""

if ($args.Count -gt 0) {
    & $pythonExe run.py @args
} else {
    & $pythonExe run.py
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[提示] 服务异常退出或启动失败，请检查上方日志输出。" -ForegroundColor Yellow
    Read-Host "按回车键退出..."
    exit $LASTEXITCODE
}
