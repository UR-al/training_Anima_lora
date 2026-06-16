# -*- coding: utf-8 -*-
"""Regression tests for runtime LR control (library/training/loop._apply_runtime_lr).

The load-bearing invariant: the live LR multiplier must NOT compound. A plain
in-place `pg['lr'] *= scale` only avoids compounding when a real scheduler rewrites
`lr` every step; schedule-free optimizers run under a no-op DummyScheduler, so the
multiply would collapse (0.5ⁿ→0) or explode (2ⁿ) the LR. These tests stub both the
schedule-free case (lr never reset) and the real-scheduler case (lr reset each step).
No torch needed — a plain dict param-group is all `_apply_runtime_lr` touches.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import library.monitoring.mcp_data as mcp_data  # noqa: E402
import library.training.loop as loop  # noqa: E402


class _Opt:
    def __init__(self, *lrs):
        self.param_groups = [{"lr": float(x)} for x in lrs]


class _State:
    def __init__(self, opt, gs=0):
        self.optimizer = opt
        self.global_step = gs


class _Args:
    monitor = True


def _reset(monkeypatch, control):
    """Force `_apply_runtime_lr` to see `control` and start with fresh tracking."""
    monkeypatch.setattr(mcp_data, "read_control", lambda *a, **k: dict(control))
    loop._LR_CTRL.update(cache={}, read_step=-10_000, groups={}, scale=1.0)


def test_schedulefree_constant_scale_does_not_compound(monkeypatch):
    # DummyScheduler / schedule-free: nothing ever resets pg['lr'].
    _reset(monkeypatch, {"lr_scale": 0.5})
    opt = _Opt(1.0)
    st = _State(opt)
    for i in range(50):
        st.global_step = i
        loop._apply_runtime_lr(_Args(), st)
    # Must stay at base*scale = 0.5, NOT 0.5**50 (~9e-16).
    assert abs(opt.param_groups[0]["lr"] - 0.5) < 1e-9


def test_schedulefree_scale_up_does_not_explode(monkeypatch):
    _reset(monkeypatch, {"lr_scale": 2.0})
    opt = _Opt(1e-4)
    st = _State(opt)
    for i in range(50):
        st.global_step = i
        loop._apply_runtime_lr(_Args(), st)
    assert abs(opt.param_groups[0]["lr"] - 2e-4) < 1e-12  # 2*base, not 2**50*base


def test_real_scheduler_applies_fresh_base_each_step(monkeypatch):
    # Real scheduler: it rewrites pg['lr']=scheduled BEFORE _apply_runtime_lr runs.
    _reset(monkeypatch, {"lr_scale": 0.5})
    opt = _Opt(0.0)
    st = _State(opt)
    for i in range(20):
        base = 1.0 + i  # a (silly) per-step schedule
        opt.param_groups[0]["lr"] = base  # the "scheduler.step()" effect
        st.global_step = i
        loop._apply_runtime_lr(_Args(), st)
        assert abs(opt.param_groups[0]["lr"] - base * 0.5) < 1e-9  # realized, no compound


def test_reset_to_one_restores_base_for_schedulefree(monkeypatch):
    _reset(monkeypatch, {"lr_scale": 0.5})
    opt = _Opt(1.0)
    st = _State(opt)
    for i in range(10):
        st.global_step = i
        loop._apply_runtime_lr(_Args(), st)
    assert abs(opt.param_groups[0]["lr"] - 0.5) < 1e-9
    # User sets scale back to 1.0 → LR must return to the original base, not stay at 0.5.
    monkeypatch.setattr(mcp_data, "read_control", lambda *a, **k: {"lr_scale": 1.0})
    loop._LR_CTRL["read_step"] = -10_000  # force a re-read
    for i in range(10, 20):
        st.global_step = i
        loop._apply_runtime_lr(_Args(), st)
    assert abs(opt.param_groups[0]["lr"] - 1.0) < 1e-9


def test_monitor_off_is_inert(monkeypatch):
    _reset(monkeypatch, {"lr_scale": 0.5})

    class _Off:
        monitor = False

    opt = _Opt(1.0)
    st = _State(opt)
    loop._apply_runtime_lr(_Off(), st)
    assert opt.param_groups[0]["lr"] == 1.0  # untouched
    assert loop.current_lr_scale() == 1.0


def test_current_lr_scale_reflects_applied(monkeypatch):
    _reset(monkeypatch, {"lr_scale": 0.25})
    opt = _Opt(1.0)
    st = _State(opt)
    loop._apply_runtime_lr(_Args(), st)
    assert abs(loop.current_lr_scale() - 0.25) < 1e-12


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
