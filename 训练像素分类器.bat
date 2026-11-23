@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

if not exist "data\raw\annotations\masks" (
  echo [ERROR] No masks found: data\raw\annotations\masks
  echo Run the labeling tool first: 标注工具.bat
  echo.
  pause
  exit /b 1
)

echo Training pixel classifier with manual masks...
echo.

"%PY%" scripts\train_pixelclf.py --masks data\raw\annotations\masks

echo.
echo Done. Press any key to close.
pause >nul
endlocal
