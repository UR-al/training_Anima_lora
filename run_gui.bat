@echo off
REM Launch the Anima LoRA web control panel (configure -> launch -> monitor)
REM in your browser. Pass extra flags through, e.g.:  run_gui.bat --port 7900
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" tasks.py webgui %*
) else (
  uv run python tasks.py webgui %*
)
