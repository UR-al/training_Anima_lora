@echo off
REM Launch the Anima LoRA native (PySide6/Qt) desktop control panel.
REM Needs the optional `gui` extra (uv sync --extra gui).
cd /d "%~dp0"
REM Reduce CUDA fragmentation OOMs ("reserved but unallocated" kind) for the
REM train.py subprocess the panel spawns. setdefault-style: don't clobber a
REM value the user already exported.
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
REM Prefer `uv run --extra gui` when uv is present: it ensures the opt-in PySide6
REM extra is installed (self-heals if a plain `uv sync` ever removed it), then runs
REM in the synced .venv. Only fall back to the bare .venv python when uv isn't there.
where uv >nul 2>nul
if not errorlevel 1 (
  uv run --extra gui python tasks.py native-gui %*
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" tasks.py native-gui %*
) else (
  echo no uv and no .venv - run install_uv.bat or install_pip.bat first.
)
REM Keep the window open if it exited with an error, so the message is readable.
if errorlevel 1 pause
