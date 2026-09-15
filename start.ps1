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

# 检查 Python 环境
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[错误] 未在系统 PATH 中检测到 Python 环境！" -ForegroundColor Red
    Write-Host "请先安装 Python 3.10+ 并勾选 'Add Python to PATH'。" -ForegroundColor Yellow
    Read-Host "按回车键退出..."
    exit 1
}

python run.py $args
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[提示] 服务异常退出或启动失败，请检查上方日志输出。" -ForegroundColor Yellow
    Read-Host "按回车键退出..."
    exit $LASTEXITCODE
}
