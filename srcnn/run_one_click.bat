@echo off
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PY=%ROOT%\.venv_shared\Scripts\python.exe"
if not exist "%PY%" (
  call "%ROOT%\one_click_retrain.bat"
  exit /b %errorlevel%
)
"%PY%" "%ROOT%\one_click_retrain.py" --pipeline srcnn --only all
exit /b %errorlevel%
