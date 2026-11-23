@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo 生成六面额识别演示图...
"%PY%" scripts\demo_denominations.py
echo.
echo 生成硬币计数演示图...
"%PY%" scripts\demo_coins.py
echo.

echo 结果：
echo   results\figures\demo_six_denominations.png
echo   results\figures\demo_coins.png
echo.
start "" "results\figures"
pause
endlocal
