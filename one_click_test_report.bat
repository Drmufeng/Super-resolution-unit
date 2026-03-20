@echo off
setlocal

for %%I in ("%~dp0.") do set "ROOT=%%~fI"
set "PY=%ROOT%\.venv_shared\Scripts\python.exe"

if not exist "%PY%" (
  echo [Fail] Shared environment missing: %PY%
  exit /b 1
)

"%PY%" "%ROOT%\one_click_test_report.py" --mode all --with-docx
exit /b %errorlevel%
