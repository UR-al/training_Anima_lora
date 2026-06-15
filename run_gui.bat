@echo off
REM Launch the Anima LoRA Gradio control panel (configure -> launch -> monitor)
REM in your browser. Needs the optional `gradio` extra (uv sync --extra gradio).
REM Pass extra flags through, e.g.:  run_gui.bat --port 7900
cd /d "%~dp0"
REM Reduce CUDA fragmentation OOMs ("reserved but unallocated" kind) for the GUI
REM and the train.py subprocess it spawns. setdefault-style: don't clobber a
REM value the user already exported.
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" tasks.py gradio-gui %*
) else (
  uv run python tasks.py gradio-gui %*
)
REM Keep the window open if it exited with an error, so the message is readable
REM (a normal Ctrl-C stop exits 0 and closes cleanly).
if errorlevel 1 pause
