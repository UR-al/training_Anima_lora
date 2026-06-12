"""Web training monitor (vendored from AnimaLoraToolkit) + the composing sink
that bridges it to anima_lora's ProgressSink.

- ``train_monitor``: the stdlib-only HTTP dashboard server + state store.
- ``monitor_smooth.html``: the Chart.js dashboard client (served by the server).
- ``sink.MonitorSink``: forwards ProgressSink events to the dashboard so the
  monitor attaches at the existing per-step hook with no edits to the hot loop.
"""
