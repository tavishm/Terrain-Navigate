from __future__ import annotations
import numpy as np
from typing import Tuple


def load_synthetic_terrain(shape: Tuple[int, int] = (200, 200), seed: int = 42) -> np.ndarray:
    """Deterministic, smooth-ish synthetic height map for demos.

    Combines a few sinusoids and a gentle hill bump.
    Returns H with shape (m, n), dtype=float32.
    """
    rng = np.random.default_rng(seed)
    m, n = shape
    y = np.linspace(0, 2 * np.pi, m, dtype=np.float32)
    x = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    Z = (
        20.0 * np.sin(1.0 * X) +
        15.0 * np.cos(1.3 * Y) +
        5.0 * np.sin(2.1 * X + 0.7 * Y)
    ).astype(np.float32)

    # gentle hill in the middle
    cx, cy = 0.5 * (n - 1), 0.5 * (m - 1)
    DX = (np.arange(n) - cx)[None, :]
    DY = (np.arange(m) - cy)[:, None]
    R2 = (DX ** 2 + DY ** 2) / float(max(m, n))
    Z += (30.0 * np.exp(-R2 / 200.0)).astype(np.float32)

    # tiny noise for realism
    Z += (rng.normal(0, 0.25, size=(m, n))).astype(np.float32)
    return Z.astype(np.float32)


# Placeholder for real data path (not used in demo yet)
try:
    import laspy  # type: ignore
except Exception:  # laspy optional
    laspy = None


def load_laz(path: str):  # pragma: no cover - not used yet
    if laspy is None:
        raise RuntimeError("laspy not installed. Install 'laspy' to read LAZ.")
    # TODO: implement LAZ -> grid pipeline
    raise NotImplementedError
