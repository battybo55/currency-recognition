@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================================
echo   Banknote / Coin Labeling Tool
echo   Left click : add point
echo   Right click / Enter : close polygon
echo   r : rect mode      Backspace : undo point
echo   u : undo label     d : next    a : prev    q : quit
echo ============================================================
echo.

"%PY%" scripts\label_tool.py --folder data\raw\self --pattern "s*.jpg" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo [ERROR] exit code %RC% - see the message above.
  pause
)
endlocal
