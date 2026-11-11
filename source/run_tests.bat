@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0..\runtime\python\python.exe"
if not exist "%PY%" (
  echo [ERROR] runtime python not found: %PY%
  echo         expected runtime\python\python.exe under the workspace root
  exit /b 1
)
echo [tests] python: %PY%
"%PY%" -u -X utf8 tests\run_tests.py
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" (echo [tests] ALL PASSED) else (echo [tests] FAILED, exit=%RC%)
exit /b %RC%