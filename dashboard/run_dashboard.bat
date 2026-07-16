@echo off
REM Double-click this to start the evaluation dashboard on Windows.
REM It runs the project venv's Python on app.py, which opens your browser
REM automatically. Keep this window open while you use the dashboard.
cd /d "%~dp0.."
".venv\Scripts\python.exe" "dashboard\app.py"
echo.
echo Dashboard stopped. Press a key to close this window.
pause >nul
