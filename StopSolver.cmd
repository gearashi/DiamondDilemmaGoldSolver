@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0StopSolver.ps1"
exit /b %errorlevel%
