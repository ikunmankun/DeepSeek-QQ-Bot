[CmdletBinding()]
param(
    [ValidateSet("Install", "Remove")]
    [string]$Action = "Install",
    [string]$TaskName = "Xiaomachi Pet Startup",
    [string]$PetRoot = "E:\QQ-Bot\pet"
)

# NOTE: keep this file ASCII-only. PowerShell 5.1 reads BOM-less scripts in the
# ANSI codepage (GBK on zh-CN systems); a UTF-8 Chinese comment whose byte
# count lands odd can swallow the newline and merge the next code line into
# the comment (same failure class as Chinese rem lines in .bat files).

$ErrorActionPreference = "Stop"

if ($Action -eq "Remove") {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed scheduled task: $TaskName"
    exit 0
}

# The task action runs electron.exe directly through a bare cmd that first
# clears NODE_OPTIONS: this machine sets a machine-wide
# NODE_OPTIONS=--openssl-legacy-provider, and Electron refuses to start with
# it (exit code 9, "not allowed in NODE_OPTIONS"). Interactive shells do not
# inherit it, which is why manual launches worked while the task did not.
# No wscript/npx intermediates: their console chains die with the session.
$shellDir = Join-Path $PetRoot "shell"
$electronExe = Join-Path $shellDir "node_modules\electron\dist\electron.exe"
if (-not (Test-Path $electronExe)) {
    throw "Electron not found: $electronExe (run pet\start-pet.bat once to npm-install)"
}

$taskArguments = "/c set NODE_OPTIONS=& `"$electronExe`" `"$shellDir`""
$taskAction = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\cmd.exe" -Argument $taskArguments
$taskTrigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$taskPrincipal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited
$taskSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable

$task = New-ScheduledTask `
    -Action $taskAction `
    -Trigger $taskTrigger `
    -Principal $taskPrincipal `
    -Settings $taskSettings `
    -Description "Starts the Dafeiyu desktop pet shell alongside the Xiaomachi bot at logon."
Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName

Write-Host "Installed and started scheduled task: $TaskName"
Write-Host "The pet shell starts with the bot at logon; hide/show it from its tray icon."
