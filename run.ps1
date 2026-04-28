# run.ps1 - Plain Text Version
if (Test-Path ".\venv\Scripts\python.exe") {
    $PYTHON = ".\venv\Scripts\python.exe"
} else {
    $PYTHON = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
}

Write-Host ""
Write-Host "--- Medicine Strip OCR Server ---"
Write-Host "Python: $PYTHON"
Write-Host "URL: http://localhost:8000"
Write-Host ""

if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" -Force | Out-Null
}

& $PYTHON -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
