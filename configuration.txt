<#
PowerShell equivalente di `run_all.sh` (Windows).

Uso:
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\run_all.ps1

Questo script prova ad attivare il virtualenv locale in `.venv\Scripts\Activate.ps1`.
Sostituisce il comando `uv run ...` con `python ...`. Se preferisci usare `uv`, modifica le voci nella variabile `$cmds`.
#>

Push-Location $PSScriptRoot

$activate = Join-Path $PSScriptRoot '.venv\Scripts\Activate.ps1'
if (Test-Path $activate) {
    Write-Host "Attivo virtualenv: $activate"
    & $activate
} else {
    Write-Warning "Virtualenv non trovato in $activate. Assicurati di attivarlo manualmente se necessario."
}

$python = 'python'

$cmds = @(
    @($python, '-m', 'Transformers', '--dataset', 'bace', '--model', 'chemberta'),
    @($python, '-m', 'Transformers', '--dataset', 'bace', '--model', 'chemberta2'),
    @($python, '-m', 'Transformers', '--dataset', 'bace', '--model', 'selformer'),
    @($python, '-m', 'Transformers', '--dataset', 'bace', '--model', 'smilesbert'),

    @($python, '-m', 'Transformers', '--dataset', 'cep', '--model', 'chemberta'),
    @($python, '-m', 'Transformers', '--dataset', 'cep', '--model', 'chemberta2'),
    @($python, '-m', 'Transformers', '--dataset', 'cep', '--model', 'selformer'),
    @($python, '-m', 'Transformers', '--dataset', 'cep', '--model', 'smilesbert'),

    @($python, '-m', 'Transformers', '--dataset', 'malaria', '--model', 'chemberta'),
    @($python, '-m', 'Transformers', '--dataset', 'malaria', '--model', 'chemberta2'),
    @($python, '-m', 'Transformers', '--dataset', 'malaria', '--model', 'selformer'),
    @($python, '-m', 'Transformers', '--dataset', 'malaria', '--model', 'smilesbert'),

    @($python, '-m', 'Transformers', '--dataset', 'lipophilicity', '--model', 'chemberta'),
    @($python, '-m', 'Transformers', '--dataset', 'lipophilicity', '--model', 'chemberta2'),
    @($python, '-m', 'Transformers', '--dataset', 'lipophilicity', '--model', 'selformer'),
    @($python, '-m', 'Transformers', '--dataset', 'lipophilicity', '--model', 'smilesbert')
)

foreach ($cmd in $cmds) {
    $exe = $cmd[0]
    $args = @()
    if ($cmd.Length -gt 1) { $args = $cmd[1..($cmd.Length - 1)] }
    Write-Host "Eseguo: $($exe) $($args -join ' ')"
    & $exe @args
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Comando terminato con codice $LASTEXITCODE. Interrompo l'esecuzione."
        Break
    }
}

Pop-Location

Write-Host "run_all.ps1 terminato."
