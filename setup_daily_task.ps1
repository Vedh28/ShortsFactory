param([string]$Time = "09:00")
$python = (Get-Command python).Source
$script = Join-Path $PSScriptRoot "shorts_maker.py"
$action = New-ScheduledTaskAction -Execute $python -Argument "`"$script`"" -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable
Register-ScheduledTask -TaskName "ShortsMakerDaily" -Action $action -Trigger $trigger -Settings $settings -Description "Generate and publish a historical YouTube Short" -Force | Out-Null
Write-Host "Scheduled daily at $Time. The PC must be awake and connected to the internet."
