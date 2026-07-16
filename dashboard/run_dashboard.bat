@echo off
REM Double-click to set up (first time) and launch the dashboard on Windows.
REM Needs Python installed. Keep this window open while you use the dashboard.
cd /d "%~dp0.."
python run.py
echo.
echo Dashboard stopped. Press any key to close this window.
pause >nul
