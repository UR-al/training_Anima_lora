@echo off
REM ===================================================================
REM  Anima LoRA - update from GitHub: git pull + re-sync dependencies.
REM  Your datasets / output / models are gitignored and never touched.
REM ===================================================================
setlocal
cd /d "%~dp0"

echo Pulling latest from GitHub ...
git pull
if errorlevel 1 (
  echo git pull failed ^(local changes? wrong remote?^). Resolve, then re-run.
  pause
  exit /b 1
)

echo.
echo Re-syncing dependencies ...
where uv >nul 2>nul
if errorlevel 1 (
  if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu132 --pre
  ) else (
    echo no uv and no .venv - run install.bat or install_pip.bat first.
  )
) else (
  uv sync
)

echo.
echo [OK] Updated.
pause
