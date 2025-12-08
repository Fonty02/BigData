<#
PowerShell script per eseguire GraphMAE_finetune.py su tutti i dataset.

Uso:
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\run_graphmae.ps1
#>

Push-Location $PSScriptRoot

# Tenta di attivare il virtualenv nella root del progetto (un livello sopra 'prog')
$activate = Join-Path $PSScriptRoot '..\.venv\Scripts\Activate.ps1'
if (Test-Path $activate) {
    Write-Host "Attivo virtualenv: $activate"
    & $activate
} else {
    # Fallback: cerca nella cartella corrente (come in run_all.ps1 originale)
    $activateLocal = Join-Path $PSScriptRoot '.venv\Scripts\Activate.ps1'
    if (Test-Path $activateLocal) {
        Write-Host "Attivo virtualenv: $activateLocal"
        & $activateLocal
    } else {
        Write-Warning "Virtualenv non trovato. Assicurati di averlo attivato."
    }
}

$python = 'python'
$script = 'GraphMAE_finetune.py'

# Lista dei dataset presenti nella cartella data/dataset_finetuning
$datasets = @(
    "bace",
    "cep",
    "lipophilicity",
    "malaria"
)

# Parametri comuni
$epochs = 100
$batch_size = 32

foreach ($dataset in $datasets) {
    Write-Host "================================================================"
    Write-Host " AVVIO FINE-TUNING GRAPHMAE SU DATASET: $dataset "
    Write-Host "================================================================"
    
    $cmdArgs = @($script, '--dataset', $dataset, '--epochs', $epochs, '--batch_size', $batch_size)
    
    Write-Host "Eseguo: $python $cmdArgs"
    
    & $python @cmdArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Errore durante l'esecuzione del dataset $dataset. Codice uscita: $LASTEXITCODE"
        # Decommenta 'Break' se vuoi fermare l'esecuzione al primo errore
        # Break 
    }
    
    Write-Host "Completato $dataset.`n"
}

Pop-Location
Write-Host "run_graphmae.ps1 terminato."
