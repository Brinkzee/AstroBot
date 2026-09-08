# AstroBot 一键启动快捷脚本 (PowerShell)
# 自动适配控制台编码并调用 run.py 执行环境检查与拉起主服务

[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host ">>> [AstroBot] 正在启动环境就绪检查与智能客服服务..." -ForegroundColor Cyan

python run.py $args
