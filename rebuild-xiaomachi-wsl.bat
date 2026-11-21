@echo off
setlocal
echo WSL BAT VERSION 20260912-REBUILD
set "ENTRY=/usr/local/bin/xiaomachi-wsl-entry"
echo Rebuilding Xiaomachi from the current source tree.
echo A new release is copied, the bot image is rebuilt, and xiaomachi-bot +
echo xiaomachi-private are recreated. The QQ platform container keeps running.
wsl.exe --user root --cd "%~dp0" --exec bash infra/wsl/scripts/xiaomachi-wsl-entry.sh install
if errorlevel 1 goto :failed
echo Verifying Xiaomachi readiness...
wsl.exe --user root --exec "%ENTRY%" status
set "STATUS_EXIT_CODE=%ERRORLEVEL%"
if "%STATUS_EXIT_CODE%"=="75" goto :recovering
if not "%STATUS_EXIT_CODE%"=="0" goto :failed
echo Rebuild complete. The new code, configs and persona are live.
rem 自改 V1：顺手拉起（幂等，单实例锁兜底）WSL 侧 selfmod worker，测试/发布提案
wsl.exe --user root --exec bash -c "mkdir -p /var/lib/docker/volumes/xiaomachi-bot-data/_data/selfmod && nohup python3 /mnt/e/QQ-Bot/scripts/selfmod_worker.py >> /var/lib/docker/volumes/xiaomachi-bot-data/_data/selfmod/worker.log 2>&1 &"
ping -n 3 127.0.0.1 >nul
exit /b 0

:recovering
echo QQ is temporarily offline. The stack and watchdog are still running.
echo Run status-xiaomachi-wsl.bat later to confirm it is back online.
pause
exit /b 75

:failed
echo Rebuild failed. Review the output above.
pause
exit /b 1
