@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Setup.ps1" %*
set "setup_exit=%errorlevel%"
if not "%setup_exit%"=="0" echo Setup failed. See the error above.
exit /b %setup_exit%
