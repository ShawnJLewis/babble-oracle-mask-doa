"""Microphone-array geometries and the direction grids used by the simulation.

Five arrays are registered.  Each one fixes the room, the element coordinates,
the source distances, the reverberation grid and the accuracy tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import C

# MIRD 4-4-4-8-4-4-4 cm spacing, centred on the array origin
BAR_SPACING = np.array([0.04, 0.04, 0.04, 0.08, 0.04, 0.04, 0.04])
_off = np.concatenate([[0.0], np.cumsum(BAR_SPACING)])
MIC_OFF_8 = np.stack([_off - _off.mean(), np.zeros(8), np.zeros(8)], 1)

# 37 test directions in 5 deg steps.  Array a2 draws its babble talkers from
# this grid; the other arrays use their full (distance x direction) grid.
TEST_GRID = np.arange(-90.0, 90.0 + 1e-9, 5.0)


@dataclass(frozen=True)
class Scenario:
    """One microphone array with its room, source grid and tolerance."""

    key: str
    room: tuple
    dists: tuple
    mic_off: np.ndarray                 # (n_mic, 3), centred on the array origin
    t60s: np.ndarray
    angles: np.ndarray                  # RIR grid: source and interferer directions
    targets: np.ndarray                 # directions a target may take
    tol: float                          # gross-accuracy tolerance, deg
    offset: float = 0.0
    step: float = 1.0                   # DOA search-grid step, deg
    babble_angles: np.ndarray | None = field(default=None, compare=False)

    @property
    def n_mic(self) -> int:
        return len(self.mic_off)

    @property
    def grid(self) -> np.ndarray:
        """Candidate directions of the DOA search, front half-plane only."""
        return np.arange(-90.0, 90.0 + 1e-9, self.step)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.room[0] / 2, self.room[1] / 2, self.room[2] / 2])

    @property
    def mic_pos(self) -> np.ndarray:
        return self.center + self.mic_off

    @property
    def _pair_lengths(self) -> np.ndarray:
        d = np.linalg.norm(self.mic_off[:, None, :] - self.mic_off[None, :, :], axis=2)
        return d[np.triu_indices(self.n_mic, k=1)]

    @property
    def min_spacing(self) -> float:
        return float(self._pair_lengths.min())

    @property
    def aperture(self) -> float:
        return float(self._pair_lengths.max())

    @property
    def alias_hz(self) -> float:
        """Highest frequency without spatial aliasing, c / (2 * min spacing)."""
        return C / (2.0 * self.min_spacing)

    def src_pos(self, theta_deg: float, dist: float) -> np.ndarray:
        th = np.deg2rad(theta_deg)
        return self.center + dist * np.array([np.sin(th), np.cos(th), 0.0])

    def pairs(self, n: int | None = None) -> list[tuple[int, int]]:
        m = self.n_mic if n is None else n
        return [(a, b) for a in range(m) for b in range(a + 1, m)]

    def tau(self, dist: float, idx=None) -> np.ndarray:
        """(n_pairs, n_grid) near-field TDOA of every candidate direction."""
        mic = self.mic_pos if idx is None else self.mic_pos[idx]
        cand = np.stack([self.src_pos(a, dist) for a in self.grid])
        d = np.linalg.norm(cand[:, None, :] - mic[None, :, :], axis=2).T
        return np.array([(d[b] - d[a]) / C
                         for a, b in self.pairs(None if idx is None else len(idx))])

    def describe(self) -> dict:
        return {
            "key": self.key,
            "n_mic": self.n_mic,
            "room": list(self.room),
            "aperture_m": round(self.aperture, 4),
            "min_spacing_m": round(self.min_spacing, 4),
            "alias_hz": round(self.alias_hz, 1),
            "dists_m": [float(d) for d in self.dists],
            "t60s": [float(t) for t in self.t60s],
            "tol_deg": float(self.tol),
            "search_step_deg": float(self.step),
        }


def ring(n: int, r: float, z: float = 0.0, phase: float = 0.0) -> np.ndarray:
    """n microphones on a circle of radius r (metres) at height offset z."""
    a = 2 * np.pi * np.arange(n) / n + phase
    return np.stack([r * np.cos(a), r * np.sin(a), np.full(n, z)], axis=1)


A2 = Scenario(
    key="a2", room=(8.0, 8.0, 3.0), dists=(1.5,),
    mic_off=np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
    t60s=np.round(np.arange(0.0, 1.01, 0.2), 2),
    angles=np.concatenate([np.arange(-87.5, 87.6, 5.0), TEST_GRID]),
    targets=TEST_GRID, tol=5.0, step=1.0, babble_angles=TEST_GRID)

B8 = Scenario(
    key="b8", room=(6.0, 6.0, 2.4), dists=(1.0, 2.0),
    mic_off=MIC_OFF_8,
    t60s=np.array([0.16, 0.36, 0.61]),
    angles=np.arange(-90.0, 90.1, 15.0),
    targets=np.arange(-75.0, 75.1, 15.0),      # interior directions, no endfire
    tol=7.5, step=1.0)

SPK6 = Scenario(                                # smart speaker, top ring
    key="spk6", room=(6.0, 6.0, 2.4), dists=(1.0, 2.0),
    mic_off=ring(6, 0.045),
    t60s=np.array([0.16, 0.36, 0.61]),
    angles=np.arange(-90.0, 90.1, 15.0),
    targets=np.arange(-75.0, 75.1, 15.0), tol=7.5, step=1.0)

ROB5 = Scenario(                                # robot vacuum, floor-level ring
    key="rob5", room=(6.0, 6.0, 2.4), dists=(1.0, 2.0),
    mic_off=ring(5, 0.11, z=-1.15),
    t60s=np.array([0.16, 0.36, 0.61]),
    angles=np.arange(-90.0, 90.1, 15.0),
    targets=np.arange(-75.0, 75.1, 15.0), tol=7.5, step=1.0)

CAR4 = Scenario(                                # car cabin, non-coplanar
    key="car4", room=(2.8, 1.6, 1.4), dists=(0.8,),
    mic_off=np.array([[-1.00, -0.55, 0.45], [1.00, -0.55, 0.45],
                      [-0.30, 0.55, 0.55], [0.30, 0.55, 0.55]]),
    t60s=np.array([0.16, 0.36, 0.61]),
    angles=np.arange(-90.0, 90.1, 15.0),
    targets=np.arange(-75.0, 75.1, 15.0), tol=10.0, step=1.0)

SCENARIOS = {s.key: s for s in (A2, B8, SPK6, ROB5, CAR4)}


def get(key: str) -> Scenario:
    try:
        return SCENARIOS[key]
    except KeyError:
        raise KeyError(f"unknown array {key!r}; choose from {sorted(SCENARIOS)}") from None
