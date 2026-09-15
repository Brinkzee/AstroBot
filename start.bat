@echo off
chcp 65001 >nul
title AstroBot 智能电商客服系统
cd /d "%~dp0"

echo =================================================================
echo         🚀 AstroBot 智能电商客服系统一键启动程序 🚀
echo =================================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未在系统 PATH 中检测到 Python 环境！
    echo 请先安装 Python 3.10+ 并勾选 "Add Python to PATH"。
    echo.
    pause
    exit /b 1
)

python run.py %*
if errorlevel 1 (
    echo.
    echo [提示] 服务异常退出或启动失败，请检查上方日志输出。
    echo.
    pause
    exit /b %errorlevel%
)
