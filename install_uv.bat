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
echo Running "uv sync --extra gui" - resolves torch ^(cu132^) + flash-attn + the
echo native PySide6 GUI. This can take a while...
uv sync --extra gui
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
echo Installing nvcc for torch.compile ^(nvidia-cuda-nvcc^) - side-loaded so it does
echo NOT perturb the locked CUDA stack. Best-effort; eager fallback covers a miss.
uv pip install nvidia-cuda-nvcc

echo.
echo [OK] Installed. Start the GUI with  run_gui.bat   ^(or: uv run python tasks.py native-gui^)
echo.
echo NOTE: nvcc (for torch.compile) is installed for you via the nvidia-cuda-nvcc
echo       wheel - no manual CUDA Toolkit needed. If nvcc is somehow missing,
echo       training auto-falls-back to eager; a full toolkit is at:
echo       https://developer.nvidia.com/cuda-13-2-0-download-archive
echo.
pause
