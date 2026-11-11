@echo off
setlocal enabledelayedexpansion
title Video Dewatermark Tool - stop

set "ROOT=%~dp0"
set "PORT=8231"
set "PIDFILE=%ROOT%backend\logs\server.pid"
set "KILLED=0"

if exist "%PIDFILE%" (
  set /p OLD=<"%PIDFILE%"
  if defined OLD (
    taskkill /PID !OLD! /T /F >nul 2>&1
    if not errorlevel 1 (
      echo [OK] stopped server pid !OLD!
      set "KILLED=1"
    )
  )
  del /q "%PIDFILE%" >nul 2>&1
)

rem ---- fallback: kill whatever is still listening on the port ----
set "BUSY="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do set "BUSY=%%P"
if defined BUSY (
  taskkill /PID !BUSY! /T /F >nul 2>&1
  echo [OK] stopped listener pid !BUSY!
  set "KILLED=1"
)

if "!KILLED!"=="0" echo [i] nothing was listening on port %PORT%
echo done.
endlocal
