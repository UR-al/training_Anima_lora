@echo off
REM ===================================================================
REM  Anima LoRA - install via uv (recommended: exact, locked deps).
REM  Double-click, or run from a terminal in the repo root.
REM ===================================================================
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv ^(https://astral.sh/uv^) ...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

where uv >nul 2>nul
if errorlevel 1 (
  echo error: uv not found after install. Open a NEW terminal and re-run this,
  echo        or use install_pip.bat instead.
  pause
  exit /b 1
)

echo.
echo Running "uv sync --extra gradio" - resolves torch ^(cu132^) + flash-attn + the
echo Gradio GUI. This can take a while...
uv sync --extra gradio
if errorlevel 1 (
  echo.
  echo uv sync failed. Most common cause on Windows: antivirus locking uv's
  echo trampoline .exe files. Add a Defender folder exclusion for:
  echo     %CD%
  echo     %LOCALAPPDATA%\uv
  echo then re-run this installer.
  pause
  exit /b 1
)

echo.
echo [OK] Installed. Start the GUI with  run_gui.bat   ^(or: uv run python tasks.py gradio-gui^)
echo.
echo NOTE: torch.compile / Triton needs the CUDA 13.2 toolkit ^(nvcc^). If training
echo       errors during compile, install it from:
echo       https://developer.nvidia.com/cuda-13-2-0-download-archive
echo.
pause
