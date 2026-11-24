@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0StartSolver.ps1" %*
set "solver_exit=%errorlevel%"
if not "%solver_exit%"=="0" echo Solver failed. See the error above.
exit /b %solver_exit%
