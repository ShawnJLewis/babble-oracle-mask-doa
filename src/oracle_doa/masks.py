"""Oracle time-frequency masks.

Both masks are computed from the simulated direct-path signal, which a real
system never has.  They therefore describe an *upper bound* on what a mask can
do for the backend, not an achievable performance.

Note the denominator of the ratio mask is the mixture minus the direct path, so
it contains late reverberation as well as babble.  This is a direct-path
dominance mask, not a speech-versus-noise mask.
"""
from __future__ import annotations

import numpy as np

KINDS = ("irm", "psm")


def oracle_masks(Ymix: np.ndarray, Ydir: np.ndarray) -> tuple[tuple, tuple]:
    """IRM and PSM, one (frames, F) mask per microphone.

    ``irm = |Yd|^2 / (|Yd|^2 + |Ymix - Yd|^2)``
    ``psm = clip(irm * cos(angle(Ymix) - angle(Yd)), 0, 1)``
    """
    k = min(Ymix.shape[1], Ydir.shape[1])
    Ymix, Ydir = Ymix[:, :k], Ydir[:, :k]
    p = np.abs(Ydir) ** 2
    n = np.abs(Ymix - Ydir) ** 2
    irm = (p / (p + n + 1e-12)).astype(np.float32)
    psm = np.clip(irm * np.cos(np.angle(Ymix) - np.angle(Ydir)), 0, 1).astype(np.float32)
    return tuple(irm), tuple(psm)


def select(name: str, masks: tuple[tuple, tuple]) -> tuple:
    """Pick one mask family out of the pair returned by :func:`oracle_masks`."""
    irm, psm = masks
    if name == "irm":
        return irm
    if name == "psm":
        return psm
    raise ValueError(f"unknown mask {name!r}; choose from {KINDS}")
