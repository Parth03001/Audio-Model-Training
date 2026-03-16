# start_label_studio.ps1
# Run this from the project root to start Label Studio with local file serving enabled.
# Usage:  .\start_label_studio.ps1

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT   = $ProjectRoot

Write-Host ""
Write-Host "Local file serving enabled."
Write-Host "  LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = true"
Write-Host "  LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT   = $ProjectRoot"
Write-Host ""
Write-Host "After Label Studio opens, get your API token:"
Write-Host "  http://localhost:8080/user/account"
Write-Host ""
Write-Host "Then run the setup script (in a NEW terminal):"
Write-Host "  python scripts\setup_label_studio.py --token YOUR_TOKEN --audio_dir $ProjectRoot\audio_dataset --action setup"
Write-Host ""

label-studio start
