@echo off
rem Dafeiyu pet launcher (webm animation pool build). Keep ASCII-only:
rem a GBK-misread Chinese rem line can swallow the newline and comment out
rem the next line. Tray icon hides/shows the pet; closing window = quit.
cd /d "%~dp0pet\shell"
if not exist node_modules\electron (
  echo First run: installing dependencies ...
  set ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/
  call npm install --no-audit --no-fund
)
start "" /b cmd /c "npx electron . > ..\pet.log 2>&1"
exit /b 0
