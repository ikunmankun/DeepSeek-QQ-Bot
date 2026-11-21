@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
set "VERSION=20261116-DIARY-PET"
set "ENTRY=/usr/local/bin/xiaomachi-wsl-entry"
set "TASK_NAME=Xiaomachi WSL Runtime"
set "PET_TASK_NAME=Xiaomachi Pet Startup"
set "TASK_INSTALLER=%~dp0infra\wsl\scripts\install_windows_runtime_task.ps1"
set "PET_TASK_INSTALLER=%~dp0infra\wsl\scripts\install_windows_pet_task.ps1"
set "MONITOR_URL=http://127.0.0.1:8765/"

echo ==================================================
echo   Xiaomachi one-click start  (%VERSION%)
echo   napcat + bot + private + monitor + pet, in one go
echo ==================================================

echo.
echo [1/7] Booting WSL ...
wsl.exe --user root --exec true >nul 2>&1
if errorlevel 1 (
  echo Cannot boot the default WSL distro. Installed distros:
  wsl.exe -l -v
  goto :failed
)

echo [2/7] Ensuring the Windows runtime task ...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$task = Get-ScheduledTask -TaskName '%TASK_NAME%' -ErrorAction SilentlyContinue; if (-not $task) { & '%TASK_INSTALLER%' }" >nul 2>&1
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Get-ScheduledTask -TaskName '%TASK_NAME%' -ErrorAction SilentlyContinue | Start-ScheduledTask" >nul 2>&1
echo       task: %TASK_NAME%

echo [3/7] Ensuring Docker daemon ...
wsl.exe --user root --exec systemctl start docker >nul 2>&1

echo [4/7] Starting QQ platform + bot + private + monitor ...
wsl.exe --user root --exec %ENTRY% start
if errorlevel 1 (
  echo.
  echo       Start failed. Refreshing WSL to clear stale GPU driver mounts; retrying once ...
  wsl.exe --user root --exec true >nul 2>&1
  wsl.exe --shutdown
  timeout /t 8 /nobreak >nul
  wsl.exe --user root --exec %ENTRY% start
  if errorlevel 1 goto :failed
)

echo.
echo [5/7] Verifying readiness ...
wsl.exe --user root --exec %ENTRY% status
set "STATUS_CODE=%ERRORLEVEL%"
if "%STATUS_CODE%"=="75" goto :login
if not "%STATUS_CODE%"=="0" goto :failed

echo.
echo [6/7] Container overview ...
wsl.exe --user root --exec docker ps --format "{{.Names}}  {{.Status}}"

echo.
echo [7/7] Starting the desktop pet (logon task, hidden launcher) ...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$task = Get-ScheduledTask -TaskName '%PET_TASK_NAME%' -ErrorAction SilentlyContinue; if (-not $task) { & '%PET_TASK_INSTALLER%' } else { Start-ScheduledTask -TaskName '%PET_TASK_NAME%' }" >nul 2>&1
if errorlevel 1 echo       pet task failed; start manually with pet\start-pet.bat
timeout /t 3 /nobreak >nul
tasklist /FI "IMAGENAME eq electron.exe" 2>nul | find /I "electron.exe" >nul && echo       pet is up (hide/show via its tray icon) || echo       pet process not detected; start manually with pet\start-pet.bat

echo.
echo All services are up.
echo   Agent live  : %MONITOR_URL%
echo   Memory map  : %MONITOR_URL%memory
echo   Diary       : %MONITOR_URL%diary
echo   QQ WebUI    : http://127.0.0.1:6099/
echo.
echo Tip: set NO_BROWSER=1 before running to skip opening the browser.
if /I not "%NO_BROWSER%"=="1" start "" "%MONITOR_URL%"
exit /b 0

:login
echo.
echo The stack is running, but the QQ platform needs a login.
echo Open http://127.0.0.1:6099/ , finish login, then run:
echo   status-xiaomachi-wsl.bat
if /I not "%NO_BROWSER%"=="1" start "" "http://127.0.0.1:6099/"
pause
exit /b 75

:failed
echo.
echo Startup failed. See the output above.
echo Tip: if you see NvFBC.dll / mount errors, run:  wsl --shutdown
echo      then run this script again.
pause
exit /b 1
