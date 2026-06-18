# -*- coding: utf-8 -*-
"""Make the pip-installed CUDA ``nvcc`` visible to torch.compile / inductor.

PyTorch's wheels bundle the CUDA *runtime* (+ cuDNN, cublas, nvrtc …) but **not** the
*compiler* — so ``torch.compile``'s inductor codegen needs an ``nvcc`` that the wheels
don't provide. Rather than make the user install the multi-GB CUDA Toolkit, we ship the
lightweight ``nvidia-cuda-nvcc`` pip wheel (nvcc + ptxas + nvlink, ~one folder) and point
``CUDA_HOME`` / ``PATH`` at it here. The wheel installs into the same ``nvidia/cu13/``
namespace tree the other ``nvidia-*`` runtime wheels populate, so that one dir is a
fairly complete CUDA_HOME (bin from this wheel, lib/include from the runtime wheels).

A real system toolkit (``CUDA_HOME`` set, or ``nvcc`` already on ``PATH``) always wins —
we only fill the gap. Torch-free; safe to import/call anywhere.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_NVCC_EXE = "nvcc.exe" if os.name == "nt" else "nvcc"


def _pip_nvcc_root() -> Path | None:
    """The bundled CUDA tree (``…/nvidia/cu13``) containing ``bin/nvcc``, or None."""
    try:
        import nvidia  # shared namespace package all nvidia-* wheels populate
    except ImportError:
        return None
    for entry in list(getattr(nvidia, "__path__", []) or []):
        for sub in ("cu13", "cu12"):
            root = Path(entry) / sub
            if (root / "bin" / _NVCC_EXE).is_file():
                return root
    return None


def _system_nvcc() -> bool:
    """True if a system nvcc is already discoverable (PATH or a real CUDA_HOME)."""
    if shutil.which("nvcc"):
        return True
    for var in ("CUDA_HOME", "CUDA_PATH"):
        ch = os.environ.get(var)
        if ch and (Path(ch) / "bin" / _NVCC_EXE).is_file():
            return True
    return False


def ensure_cuda_home() -> str | None:
    """Point ``CUDA_HOME`` / ``PATH`` at the pip ``nvcc`` when no system toolkit exists.

    No-op when a system nvcc is already found (returns its CUDA_HOME or ""). Returns
    the resolved CUDA_HOME on success, or None when no nvcc is available at all.
    """
    if _system_nvcc():
        return os.environ.get("CUDA_HOME", "")
    root = _pip_nvcc_root()
    if root is None:
        return None
    os.environ["CUDA_HOME"] = str(root)
    os.environ.setdefault("CUDA_PATH", str(root))
    bindir = str(root / "bin")
    path = os.environ.get("PATH", "")
    if bindir not in path.split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + path
    return str(root)


def nvcc_available() -> bool:
    """True if any nvcc is resolvable (system PATH/CUDA_HOME or the pip wheel)."""
    return _system_nvcc() or _pip_nvcc_root() is not None
