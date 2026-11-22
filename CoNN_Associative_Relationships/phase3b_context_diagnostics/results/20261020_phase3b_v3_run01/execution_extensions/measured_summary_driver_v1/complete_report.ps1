$ErrorActionPreference = 'Stop'
$summaryRoot = Split-Path -Parent $PSScriptRoot
$summaryRun = Join-Path $summaryRoot 'results/20261020_phase3b_v3_run01'
$summaryExtension = Join-Path $summaryRun 'execution_extensions/measured_summary_driver_v1'
New-Item -ItemType Directory -Path $summaryExtension | Out-Null
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $summaryExtension 'complete_report.ps1')
$summaryTarget = Join-Path $PSScriptRoot 'measured_summary.py'
$summaryHash = (Get-FileHash -LiteralPath $summaryTarget -Algorithm SHA256).Hash.ToLowerInvariant()
$summaryRecord = @{ source_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant(); target_sha256=$summaryHash; role='post-training measured reporting only' }
$summaryEncoding = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $summaryExtension 'manifest.json'),($summaryRecord | ConvertTo-Json),$summaryEncoding)
$summaryWatch = [Diagnostics.Stopwatch]::StartNew()
try {
    while (-not (Test-Path -LiteralPath (Join-Path $summaryRun 'execution_extensions/finalization_driver_v1/completion.json'))) {
        if ($summaryWatch.Elapsed.TotalSeconds -gt 21600) { throw 'Measured-report wait exceeded six hours' }
        $summaryFailures = @(Get-ChildItem -LiteralPath (Join-Path $summaryRun 'execution_extensions') -Recurse -Filter failure.json)
        if ($summaryFailures.Count -gt 0) { throw 'Earlier execution failure: inspect preserved failure.json files' }
        Start-Sleep -Seconds 10
    }
    if ((Get-FileHash -LiteralPath $summaryTarget -Algorithm SHA256).Hash.ToLowerInvariant() -ne $summaryHash) { throw 'Measured reporting source changed after seal' }
    $env:PYTHONDONTWRITEBYTECODE='1'
    $summaryPython = Join-Path (Split-Path -Parent $summaryRoot) '.venv_cuda23/Scripts/python.exe'
    & $summaryPython $summaryTarget
    if ($LASTEXITCODE -ne 0) { throw "Measured summary exited $LASTEXITCODE" }
    [IO.File]::WriteAllText((Join-Path $summaryExtension 'completion.json'),'{"completed":true,"optimizer_updates":0}',$summaryEncoding)
} catch {
    [IO.File]::WriteAllText((Join-Path $summaryExtension 'failure.json'),(@{error=$_.ToString()} | ConvertTo-Json),$summaryEncoding)
    throw
}
