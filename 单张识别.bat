@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

if "%~1"=="" (
  echo Usage: drag an image file onto this bat, or:
  echo   单张识别.bat path\to\note.jpg
  echo.
  pause
  exit /b 1
)

echo Processing: %~1
echo.

"%PY%" scripts\single_note_demo.py --image "%~1"
set "RC=%ERRORLEVEL%"

echo.
echo Result image: data\raw\_evidence\single_demo.png
if not "%RC%"=="0" echo [ERROR] exit code %RC%
pause
endlocal
