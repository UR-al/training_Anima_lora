"""Composing ProgressSink that mirrors training metrics to the web dashboard.

``MonitorSink`` wraps an (optional) :class:`~library.training.progress.ProgressSink`
and forwards every event to it unchanged, while additionally pushing per-step
loss / lr to the vendored stdlib HTTP monitor (``train_monitor``). Because
``AnimaTrainer`` already calls ``progress_sink.log(...)`` at the optimizer-step
boundary (``log_dispatch.dispatch_logs`` / ``library/training/loop.py``), swapping
the trainer's single ``progress_sink`` for a ``MonitorSink`` lights up the
dashboard with **no edits to the hot loop or the model**.

Every call into the monitor is guarded: a monitor failure can never crash
training (mirrors the donor's invariant).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _first_lr(logs: dict) -> Optional[float]:
    """Pick a representative LR from a ``logs`` dict (keys look like ``lr/unet``).

    Prefer the unet group, else the first ``lr/*`` entry.
    """
    fallback = None
    for key, val in logs.items():
        if not key.startswith("lr/"):
            continue
        try:
            value = float(val)
        except (TypeError, ValueError):
            continue
        if key == "lr/unet":
            return value
        if fallback is None:
            fallback = value
    return fallback


class MonitorSink:
    """ProgressSink-compatible wrapper that also drives the web dashboard.

    Mirrors the ProgressSink surface actually used by the trainer: ``run_start``,
    ``log``, ``ckpt``, ``run_end``, ``close`` (plus ``sample`` for image previews).
    ``inner`` may be ``None`` (JSONL progress disabled) — the monitor still runs.
    """

    def __init__(
        self,
        inner,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        output_dir: Optional[str] = None,
        open_browser: bool = False,
        config: Optional[dict] = None,
        resume: bool = False,
    ) -> None:
        self._inner = inner
        self._host = host
        self._port = port
        self._output_dir = output_dir
        self._open_browser = open_browser
        self._config = dict(config or {})
        self._resume = resume
        self._total_steps: Optional[int] = None
        self._update = None  # bound train_monitor.update_monitor once started
        self._started = False

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        try:
            from library.monitoring.train_monitor import (
                start_monitor_server,
                update_monitor,
            )

            start_monitor_server(
                port=self._port,
                host=self._host,
                output_dir=self._output_dir,
                open_browser=self._open_browser,
            )
            self._update = update_monitor
            logger.info(
                "web monitor: http://%s:%d",
                "localhost" if self._host in ("0.0.0.0", "127.0.0.1") else self._host,
                self._port,
            )
            if self._config:
                self._update(config=self._config)
        except Exception as exc:  # never let the monitor break training
            logger.warning("web monitor disabled (failed to start): %s", exc)
            self._update = None

    # --- ProgressSink surface (forward to inner + mirror to dashboard) ---

    def run_start(
        self,
        *,
        total_steps: int,
        total_epochs: int,
        pid: int,
        log_dir: Optional[str] = None,
    ) -> None:
        self._total_steps = total_steps
        # Rehydrate the loss/lr curve from the persisted state.json BEFORE starting
        # the server / emitting config — otherwise the config emit's save_state()
        # would overwrite state.json (with the fresh, empty in-memory state) before
        # the resume load could read the prior curve back.
        if self._resume:
            try:
                from library.monitoring.train_monitor import load_persisted_state

                n = load_persisted_state(self._config.get("run"))
                if n:
                    logger.info(
                        "web monitor: rehydrated %d prior loss point(s) for resume", n
                    )
            except Exception as exc:
                logger.debug("monitor resume rehydrate failed: %s", exc)
        self._ensure_started()
        if self._update is not None:
            try:
                self._update(
                    step=0,
                    total_steps=total_steps,
                    config={**self._config, "total_epochs": total_epochs},
                )
            except Exception as exc:
                logger.debug("monitor run_start emit failed: %s", exc)
        if self._inner is not None:
            self._inner.run_start(
                total_steps=total_steps,
                total_epochs=total_epochs,
                pid=pid,
                log_dir=log_dir,
            )

    def log(
        self,
        logs: dict,
        *,
        global_step: int,
        epoch: int,
        val_step: Optional[int] = None,
    ) -> None:
        if self._inner is not None:
            self._inner.log(
                logs, global_step=global_step, epoch=epoch, val_step=val_step
            )
        # Only mirror training steps (skip validation passes) to the loss curve.
        if self._update is None or val_step is not None:
            return
        try:
            loss = logs.get("loss/average")
            if loss is None:
                loss = logs.get("loss/current")
            self._update(
                loss=loss,
                lr=_first_lr(logs),
                epoch=epoch,
                step=global_step,
                total_steps=self._total_steps,
            )
        except Exception as exc:
            logger.debug("monitor step emit failed: %s", exc)

    def ckpt(self, *, global_step: int, path: str) -> None:
        if self._inner is not None:
            self._inner.ckpt(global_step=global_step, path=path)

    def sample(self, path: str) -> None:
        """Push a freshly-saved sample image path to the dashboard gallery."""
        if self._update is None:
            return
        try:
            self._update(sample_path=path)
        except Exception as exc:
            logger.debug("monitor sample emit failed: %s", exc)

    def run_end(
        self, *, status: str, final_step: int, error: Optional[str] = None
    ) -> None:
        if self._inner is not None:
            self._inner.run_end(status=status, final_step=final_step, error=error)

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
