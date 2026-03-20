@echo off
setlocal

for %%I in ("%~dp0.") do set "ROOT=%%~fI"
set "VENV=%ROOT%\.venv_shared"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
  echo [Setup] Creating shared environment at %VENV%
  py -3 -m venv "%VENV%"
  if errorlevel 1 exit /b 1
)

echo [Setup] Installing/updating dependencies
"%PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%PY%" -m pip install -r "%ROOT%\srcnn\requirements.txt"
if errorlevel 1 exit /b 1

echo [Run] Start one-click retraining pipeline
"%PY%" "%ROOT%\one_click_retrain.py" --only all
if errorlevel 1 exit /b 1

echo [Done] Training pipeline completed.
exit /b 0
