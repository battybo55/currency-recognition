@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem GUI 需 tkinter，用系统 Python 3.14（路径由 LOCALAPPDATA 动态求出）
set "PY=%LOCALAPPDATA%\Python\bin\python.exe"
if not exist "%PY%" set "PY=%~dp0..\.venv\Scripts\python.exe"

start "" "%PY%" scripts\gui.py
endlocal
