param(
    [string]$ProjectRoot,
    [string]$Port = "5173"
)

$pidFile = Join-Path $ProjectRoot ".run\frontend.pid"
$runDir  = Join-Path $ProjectRoot ".run"

if (-not (Test-Path $runDir)) { New-Item -ItemType Directory -Path $runDir -Force | Out-Null }

$driveLetter = "X:"
# Ensure drive letter is mapped in the user session
cmd.exe /c "subst $driveLetter `"$ProjectRoot`" 2>nul" | Out-Null

$script = "${driveLetter}\scripts\start_frontend.bat"
$frontendDir = "${driveLetter}\frontend"

$proc = Start-Process -FilePath "cmd.exe" `
    -ArgumentList @("/k", "call `"$script`" $Port") `
    -WorkingDirectory $frontendDir `
    -PassThru

$proc.Id | Set-Content -Path $pidFile -Encoding ASCII
Write-Host "Frontend launched with PID $($proc.Id)"
