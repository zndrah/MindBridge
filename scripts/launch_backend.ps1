param(
    [string]$ProjectRoot,
    [string]$Port = "8000"
)

$pidFile = Join-Path $ProjectRoot ".run\backend.pid"
$runDir  = Join-Path $ProjectRoot ".run"

if (-not (Test-Path $runDir)) { New-Item -ItemType Directory -Path $runDir -Force | Out-Null }

$driveLetter = "X:"
# Ensure drive letter is mapped in the user session
cmd.exe /c "subst $driveLetter /d 2>nul" | Out-Null
cmd.exe /c "subst $driveLetter `"$ProjectRoot`"" | Out-Null

$script = "${driveLetter}\scripts\start_backend.bat"

$proc = Start-Process -FilePath "cmd.exe" `
    -ArgumentList @("/k", "call `"$script`" $Port") `
    -WorkingDirectory "${driveLetter}\" `
    -PassThru

$proc.Id | Set-Content -Path $pidFile -Encoding ASCII
Write-Host "Backend launched with PID $($proc.Id)"
