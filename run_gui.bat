@echo off
REM Launch the Anima LoRA web control panel (configure -> launch -> monitor)
REM in your browser. Pass extra flags through, e.g.:  run_gui.bat --port 7900
cd /d "%~dp0"
REM Reduce CUDA fragmentation OOMs ("reserved but unallocated" kind) for the GUI
REM and any daemon it spawns. The daemon also sets this per-job, so this is a
REM belt-and-suspenders for direct launches. setdefault-style: don't clobber a
REM value the user already exported.
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" tasks.py webgui %*
) else (
  uv run python tasks.py webgui %*
)
REM Keep the window open if it exited with an error, so the message is readable
REM (a normal Ctrl-C stop exits 0 and closes cleanly).
if errorlevel 1 pause
