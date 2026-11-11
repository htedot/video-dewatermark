@echo off
setlocal
title Video Dewatermark Tool

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PY=%ROOT%..\runtime\python\python.exe"
set "PORT=8231"
set "LOGDIR=%ROOT%backend\logs"
set "PIDFILE=%LOGDIR%\server.pid"

if not exist "%PY%" (
  echo [ERROR] runtime python not found: %PY%
  echo         expected runtime\python\python.exe under the workspace root
  exit /b 1
)
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

rem ---- port check: must be free (8200-8299 range) ----
set "BUSY="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do set "BUSY=%%P"
if defined BUSY (
  echo [ERROR] port %PORT% is already in use by PID %BUSY%
  echo         change PORT in backend\.env to another free port in 8200-8299
  exit /b 1
)

echo [1/2] starting server on http://127.0.0.1:%PORT% ...
rem NOTE: never capture this powershell stdout with "for /f". The python child
rem inherits the pipe write handle, the pipe never reaches EOF, and cmd hangs.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Start-Process -FilePath '%PY%' -ArgumentList '-u','-X','utf8','backend\main.py' -WorkingDirectory '%ROOT%' -RedirectStandardOutput '%LOGDIR%\server.log' -RedirectStandardError '%LOGDIR%\server.err.log' -WindowStyle Hidden -PassThru; Set-Content -Path '%PIDFILE%' -Value $p.Id -Encoding ascii; $ok = $false; for ($i = 0; $i -lt 60; $i++) { try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:%PORT%/api/health' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { $ok = $true; break } } catch {} ; Start-Sleep -Milliseconds 500 }; if ($ok) { exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo [ERROR] server did not become healthy in time
  echo         see %LOGDIR%\server.err.log
  if exist "%PIDFILE%" del /q "%PIDFILE%"
  exit /b 1
)

set "SERVER_PID="
if exist "%PIDFILE%" set /p SERVER_PID=<"%PIDFILE%"
echo [2/2] healthy, pid=%SERVER_PID%
echo.
echo [OK] http://127.0.0.1:%PORT%/
echo      logs: backend\logs\server.log
start "" "http://127.0.0.1:%PORT%/"
endlocal