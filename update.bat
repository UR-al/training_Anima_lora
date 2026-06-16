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
    ".venv\Scripts\python.exe" -m pip install -e ".[gradio]" --extra-index-url https://download.pytorch.org/whl/cu132 --pre
  ) else (
    echo no uv and no .venv - run install_uv.bat or install_pip.bat first.
  )
) else (
  REM --extra gradio: the GUI (gradio + fastapi/uvicorn/starlette) is an OPT-IN extra.
  REM Plain "uv sync" UNINSTALLS it, breaking run_gui.bat — always keep the extra here.
  uv sync --extra gradio
)

echo.
REM Rebuild the GUI's torch-free options cache now (optimizer zoo + arg schema), so
REM the first GUI launch after this update is fast instead of paying the ~5s rebuild.
REM Best-effort — a failure here never blocks the update.
echo Pre-warming GUI options cache ...
where uv >nul 2>nul
if errorlevel 1 (
  if exist ".venv\Scripts\python.exe" ".venv\Scripts\python.exe" -c "import gui.backend as b; b.build_options_cache()" 2>nul
) else (
  uv run python -c "import gui.backend as b; b.build_options_cache()" 2>nul
)

echo.
echo [OK] Updated.
pause
