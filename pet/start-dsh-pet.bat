@echo off
rem dsh-pet launcher: fixed bridge token + anti-throttling flags
set "DSH_PET_TOKEN=KYHRwezNtLEaQOP_mdfbVBl9Wilmq6572Bgae8H6ob8"
cd /d "%~dp0dsh-pet\dsh-pet-0.4.8-win-x64"
start "" "dsh-pet.exe" ^
  --disable-background-timer-throttling ^
  --disable-backgrounding-occluded-windows ^
  --disable-renderer-backgrounding ^
  --disable-features=CalculateNativeWinOcclusion
exit /b 0
rem bridge: bot soul -> pet body (memory map whispers)
start "pet-bridge" /min cmd /c "python E:\QQ-Bot\pet\bridge\pet_bridge.py watch"
exit /b 0
