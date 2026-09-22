@echo off
chcp 65001 >nul
title AstroBot 智能电商客服系统
cd /d "%~dp0"

echo =================================================================
echo         🚀 AstroBot 智能电商客服系统一键启动程序 🚀
echo =================================================================
echo.

set "PY_CMD="

:: 1. 优先检测当前项目目录下的虚拟环境
if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
    echo [环境] 优先使用当前项目虚拟环境: .venv
    goto :found_python
)
if exist "venv\Scripts\python.exe" (
    set "PY_CMD=venv\Scripts\python.exe"
    echo [环境] 优先使用当前项目虚拟环境: venv
    goto :found_python
)

:: 2. 检测系统 PATH 中的 python
where python >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
    goto :found_python
)

:: 3. 检测系统 Python Launcher (py)
where py >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py -3"
    goto :found_python
)

echo [错误] 未在系统 PATH 或当前项目中检测到 Python 环境！
echo 请先安装 Python 3.10+ 并勾选 "Add Python to PATH"，或创建虚拟环境 .venv。
echo.
pause
exit /b 1

:found_python
echo [启动] 正在启动 AstroBot 服务...
echo.

%PY_CMD% run.py %*
if errorlevel 1 (
    echo.
    echo [提示] 服务异常退出或启动失败，请检查上方日志输出。
    echo.
    pause
    exit /b %errorlevel%
)
