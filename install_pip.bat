@echo off
REM ===================================================================
REM  Anima LoRA - install via pip (alternative; uv via install_uv.bat is
REM  the validated path with exact pins). Needs Python 3.13 on PATH.
REM ===================================================================
setlocal
cd /d "%~dp0"

echo Creating virtual environment (.venv) with Python 3.13 ...
py -3.13 -m venv .venv 2>nul
if errorlevel 1 python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
  echo error: could not create .venv - is Python 3.13 installed?
  echo        Get it from https://www.python.org/downloads/  then re-run.
  pause
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip

echo.
echo Installing torch + torchvision from the CUDA 13.2 index ...
pip install --extra-index-url https://download.pytorch.org/whl/cu132 --pre "torch>=2.12.0,<2.13" "torchvision>=0.27.0,<0.28"
if errorlevel 1 ( echo torch install failed. & pause & exit /b 1 )

echo.
echo Installing libraries from requirements.txt ...
pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo pip install hit an error. The uv path ^(install_uv.bat^) is the validated one;
  echo consider using it. flash-attn / triton-windows can be the sticking points.
  pause
  exit /b 1
)

echo.
echo Installing the editable anima_lora package ^(deps already satisfied^) ...
pip install -e . --no-deps
if errorlevel 1 ( echo editable install failed. & pause & exit /b 1 )

echo.
echo Installing nvcc for torch.compile ^(nvidia-cuda-nvcc^) ...
pip install nvidia-cuda-nvcc

echo.
echo [OK] Installed (pip). Start the GUI with  run_gui.bat
echo NOTE: nvcc (for torch.compile) was just installed (nvidia-cuda-nvcc) - no
echo       manual CUDA Toolkit. If it's ever missing, training runs eager.
echo       Full toolkit: https://developer.nvidia.com/cuda-13-2-0-download-archive
echo.
pause
