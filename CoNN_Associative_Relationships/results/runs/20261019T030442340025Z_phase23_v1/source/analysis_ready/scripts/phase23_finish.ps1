$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path -Parent $PSScriptRoot
Set-Location $experimentRoot
$runPath = (Get-Content config/phase23_active_run.txt).Trim()
$pythonPath = Join-Path $experimentRoot '..\conn_experiment_01\.venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'src;..\conn_experiment_01\src'
$deadline = [DateTime]::UtcNow.AddHours(6)
while ($true) {
    $executionPath = Get-ChildItem ($runPath + '/execution') -Directory | Sort-Object Name | Select-Object -Last 1
    if (Test-Path ($executionPath.FullName + '/completion.json')) {
        $finished = Get-Content ($executionPath.FullName + '/completion.json') | ConvertFrom-Json
        if ($finished.completed -eq 1050 -and $finished.failed -eq 0) { break }
    }
    if ([DateTime]::UtcNow -gt $deadline) { throw 'Grid not completed within bounded postprocessing wait; no selection executed.' }
    Start-Sleep -Seconds 10
}

function Invoke-MeasurementStep([string]$label, [string[]]$taskArguments) {
    Write-Output ('Starting ' + $label)
    & $pythonPath @taskArguments *> ($runPath + '/' + $label + '.log')
    if ($LASTEXITCODE -ne 0) {
        Get-Content ($runPath + '/' + $label + '.log') -Tail 30
        throw ($label + ' failed; measurements and logs retained')
    }
    Write-Output ('Completed ' + $label)
}

Invoke-MeasurementStep 'analysis_full' @('-m','conn_assoc.phase23_analysis','full','--run',$runPath)
Invoke-MeasurementStep 'checkpoint_diagnostics' @('-m','conn_assoc.phase23_diagnostics','--run',$runPath)
Invoke-MeasurementStep 'independent_verification' @('-m','conn_assoc.phase23_verify','--run',$runPath)
Invoke-MeasurementStep 'research_report' @('scripts/phase23_report.py')
Invoke-MeasurementStep 'delivery_finalize' @('scripts/phase23_finalize.py')
Write-Output 'Phase 2/3 complete, verified and reported; official test remains embargoed.'
