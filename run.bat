@echo off
SETLOCAL

if exist venv\Scripts\python.exe (
    set PYTHON=venv\Scripts\python.exe
) else (
    set PYTHON=python
)

echo.
echo --- Medicine Strip OCR Server ---
echo Python: %PYTHON%
echo URL: http://localhost:8000
echo.

if not exist logs (
    mkdir logs
)

%PYTHON% -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

ENDLOCAL
