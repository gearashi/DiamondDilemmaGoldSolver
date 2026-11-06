@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0OpenDashboard.ps1" %*
set "dashboard_exit=%errorlevel%"
if not "%dashboard_exit%"=="0" echo Dashboard failed. See the error above.
exit /b %dashboard_exit%
