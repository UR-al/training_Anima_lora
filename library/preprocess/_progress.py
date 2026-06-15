"""Optional progress callback for the cache functions.

The cache loops live in ``library/`` and must run headless (GUI subprocess,
tests, embedding code), so they never create a progress bar themselves. A caller that
*does* want one passes a ``progress`` callback; the CLI wrappers pass
:func:`tqdm_progress`. The protocol is intentionally tiny:

    progress(advance, total=N, detail="…")

called once up front with ``total`` to size the bar, then once per processed
item with ``advance=1`` (and an optional ``detail`` postfix). Cache functions
guard on ``progress is None``, so omitting it is a clean no-op.

Structured side-channel (``ANIMA_PROGRESS_JSONL``): when that env var is set
(a consumer — e.g. the web GUI — points it at a preprocess run's
``progress.jsonl``), the same callback *also*
appends throttled ``{"ev":"preprocess","phase":desc,"done":k,"total":N}`` lines
so the web GUI can render a real progress bar instead of scraping tqdm text. It
deliberately never emits an ``"ev":"run_end"`` line, so any exit-code
finalization is untouched. Unset (plain CLI / tests) → pure tqdm, zero overhead.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

ProgressFn = Callable[..., None]

_JSONL_ENV = "ANIMA_PROGRESS_JSONL"


def tqdm_progress(desc: str) -> ProgressFn:
    """Return a :data:`ProgressFn` that drives a lazily-created ``tqdm`` bar.

    The bar is created on the first call that supplies ``total`` so the
    function controls when (and at what size) the bar appears. When
    ``ANIMA_PROGRESS_JSONL`` is set it additionally mirrors progress to that
    file as structured JSONL (throttled to ~3 writes/sec).
    """
    from tqdm import tqdm

    state: dict[str, object] = {"bar": None, "done": 0, "total": None, "emit": 0.0}
    jsonl_path = os.environ.get(_JSONL_ENV) or None

    def _emit(force: bool = False) -> None:
        if not jsonl_path:
            return
        now = time.time()
        # Throttle so a skip-heavy pass can't thrash the disk; always emit the
        # sizing tick (force) and the final 100% (force).
        if not force and now - float(state["emit"]) < 0.3:
            return
        state["emit"] = now
        rec = {
            "ev": "preprocess",  # never "run_end" — leaves exit-code finalization untouched
            "phase": desc,
            "done": int(state["done"]),
            "total": state["total"],
        }
        try:
            with open(jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def cb(advance: int = 0, *, total: Optional[int] = None, detail: str = "") -> None:
        bar = state["bar"]
        if total is not None and bar is None:
            bar = state["bar"] = tqdm(total=total, desc=desc)
            state["total"] = total
            state["done"] = 0
            _emit(force=True)
        if bar is None:
            return
        if detail:
            bar.set_postfix_str(detail)
        if advance:
            bar.update(advance)
            state["done"] = int(state["done"]) + advance
            total_n = state["total"]
            _emit(force=total_n is not None and int(state["done"]) >= int(total_n))

    return cb
