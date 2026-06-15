@echo off
REM Launch the Anima LoRA Gradio control panel (configure -> launch -> monitor)
REM in your browser. Needs the optional `gradio` extra (uv sync --extra gradio).
REM Pass extra flags through, e.g.:  run_gui.bat --port 7900
cd /d "%~dp0"
REM Reduce CUDA fragmentation OOMs ("reserved but unallocated" kind) for the GUI
REM and the train.py subprocess it spawns. setdefault-style: don't clobber a
REM value the user already exported.
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
REM Prefer `uv run --extra gradio` when uv is present: it ensures the opt-in gradio
REM extra is installed (self-heals if a plain `uv sync` ever removed it), then runs in
REM the synced .venv. Only fall back to the bare .venv python when uv isn't available.
where uv >nul 2>nul
if not errorlevel 1 (
  uv run --extra gradio python tasks.py gradio-gui %*
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" tasks.py gradio-gui %*
) else (
  echo no uv and no .venv - run install_uv.bat or install_pip.bat first.
)
REM Keep the window open if it exited with an error, so the message is readable
REM (a normal Ctrl-C stop exits 0 and closes cleanly).
if errorlevel 1 pause
