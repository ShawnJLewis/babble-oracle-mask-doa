"""Classical DOA estimators for a general N-microphone array.

Every estimator scores candidate directions with

    sum_{pairs,f} [ C[p,f] cos(A[p,k,f]) + S[p,f] sin(A[p,k,f]) ]

where ``A`` is the phase delay predicted by the array geometry and ``C``/``S``
are the weighted sums of the observed inter-channel phase.  This factorisation
turns the whole direction grid into one einsum, so the cost does not grow with
the number of candidate directions.

The steering phases come from the microphone coordinates and the source
distance only.  No learned quantity enters the search.
"""
from __future__ import annotations

import functools

import numpy as np

from . import config
from .arrays import Scenario, get


@functools.lru_cache(maxsize=None)
def _phase(sc_key: str, dist: float, idx: tuple | None):
    """(COS_A, SIN_A), each (n_pairs, n_grid, F)."""
    sc = get(sc_key)
    tau = sc.tau(dist, None if idx is None else list(idx))          # (P, K)
    A = (2 * np.pi * (np.arange(config.F) / config.NFFT) * config.FS)[None, None, :] * tau[:, :, None]
    return np.cos(A), np.sin(A)


def _pairs(Y: np.ndarray) -> list[tuple[int, int]]:
    return [(a, b) for a in range(Y.shape[0]) for b in range(a + 1, Y.shape[0])]


def _dphi(Y: np.ndarray, pairs) -> np.ndarray:
    return np.stack([np.angle(Y[a]) - np.angle(Y[b]) for a, b in pairs])


def _cos_sin(Y, w, pairs, cosA, sinA) -> np.ndarray:
    ph = _dphi(Y, pairs)                                            # (P, T, F)
    C_ = np.einsum("ptf,ptf->pf", w, np.cos(ph))
    S_ = np.einsum("ptf,ptf->pf", w, np.sin(ph))
    return np.einsum("pkf,pf->k", cosA, C_) + np.einsum("pkf,pf->k", sinA, S_)


def _locate(score: np.ndarray, sc: Scenario) -> float:
    return float(sc.grid[np.argmax(score)])


def _prep(Y, idx, sc: Scenario, dist: float):
    pairs = _pairs(Y)
    cosA, sinA = _phase(sc.key, float(dist), None if idx is None else tuple(idx))
    return pairs, cosA, sinA


def _mask_pair_weight(M, pairs) -> np.ndarray:
    """Per-pair mask weight M_p * M_q, (P, T, F)."""
    return np.stack([M[a] * M[b] for a, b in pairs])


def _cov2(Y, w, pairs) -> np.ndarray:
    """Per-pair 2x2 covariance, (P, F, 2, 2)."""
    out = np.empty((len(pairs), config.F, 2, 2), np.complex128)
    for i, (a, b) in enumerate(pairs):
        Yab = np.stack([Y[a], Y[b]])
        Z = Yab * w[i][None, :, :]
        out[i] = np.einsum("atf,btf->fab", Z, np.conj(Yab))
    return out / np.maximum(w.sum(1), 1e-12)[:, :, None, None]


def _steer_pair(cosA, sinA) -> np.ndarray:
    """(P, K, F, 2) steering vectors, channel 0 as the reference."""
    c = np.stack([np.ones_like(cosA), cosA - 1j * sinA], -1)
    return c / np.linalg.norm(c, axis=-1, keepdims=True)


def gcc_phat(Y, sc: Scenario, dist: float, *, idx=None) -> float:
    """Classic GCC-PHAT: uniform weight over every time-frequency unit."""
    pairs, cosA, sinA = _prep(Y, idx, sc, dist)
    w = np.ones((len(pairs),) + Y.shape[1:], np.float32)
    return _locate(_cos_sin(Y, w, pairs, cosA, sinA), sc)


def mgcc(Y, sc: Scenario, dist: float, M, *, idx=None) -> float:
    """Mask-weighted GCC-PHAT, weight M_p * M_q per microphone pair."""
    pairs, cosA, sinA = _prep(Y, idx, sc, dist)
    w = _mask_pair_weight(M, pairs)
    return _locate(_cos_sin(Y, w, pairs, cosA, sinA), sc)


def sv(Y, sc: Scenario, dist: float, M, *, idx=None) -> float:
    """Steering-vector DOA from the principal eigenvector of each 2x2 covariance."""
    pairs, cosA, sinA = _prep(Y, idx, sc, dist)
    w = _mask_pair_weight(M, pairs)
    _, vec = np.linalg.eigh(_cov2(Y, w, pairs))
    d = np.angle(vec[:, :, 0, -1]) - np.angle(vec[:, :, 1, -1])      # (P, F)
    wf = w.sum(1)                                                    # (P, F)
    return _locate(np.einsum("pkf,pf->k", cosA, wf * np.cos(d)) +
                   np.einsum("pkf,pf->k", sinA, wf * np.sin(d)), sc)


def sr_snr(Y, sc: Scenario, dist: float, M, *, idx=None) -> float:
    """Mask-weighted steered-response SNR.

    With MVDR weights ``w = G c / (c^H G c)`` and ``G = Phi_n^-1``, the output
    SNR reduces to ``c^H G Phi_s G c / (c^H G Phi_s G c + c^H G c)``.
    """
    pairs, cosA, sinA = _prep(Y, idx, sc, dist)
    ws = _mask_pair_weight(M, pairs)
    wn = np.stack([(1 - M[a]) * (1 - M[b]) for a, b in pairs])
    Phis, Phin = _cov2(Y, ws, pairs), _cov2(Y, wn, pairs)
    tr = np.trace(Phin, axis1=2, axis2=3).real[:, :, None, None]
    G = np.linalg.inv(Phin + 1e-4 * tr * np.eye(2))
    c = _steer_pair(cosA, sinA)                                      # (P, K, F, 2)
    Gc = np.einsum("pfab,pkfb->pkfa", G, c)
    cGc = np.einsum("pkfa,pkfa->pkf", np.conj(c), Gc).real
    num = np.einsum("pkfa,pkfa->pkf", np.conj(c),
                    np.einsum("pfab,pkfb->pkfa", G,
                              np.einsum("pfab,pkfb->pkfa", Phis, Gc))).real
    snr = num / (num + np.maximum(cGc, 1e-12))                       # (P, K, F)
    wf = ws.sum(1)                                                   # (P, F)
    return _locate(np.einsum("pkf,pf->k", snr, wf), sc)


def music(Y, sc: Scenario, dist: float, M=None, *, idx=None) -> float:
    """Narrowband MUSIC per pair; every band is bounded to [0, 1] before summation."""
    pairs, cosA, sinA = _prep(Y, idx, sc, dist)
    if M is None:
        Phi = _cov2(Y, np.ones((len(pairs),) + Y.shape[1:], np.float32), pairs)
        wf = np.ones((len(pairs), config.F))
    else:
        wf = _mask_pair_weight(M, pairs).sum(1)
        Phi = _cov2(Y, np.stack([(1 - M[a]) * (1 - M[b]) for a, b in pairs]), pairs)
    _, vec = np.linalg.eigh(Phi)
    Un = vec[:, :, :, 0]                                             # (P, F, 2)
    c = _steer_pair(cosA, sinA)                                      # (P, K, F, 2)
    proj = np.einsum("pfm,pkfm->pkf", np.conj(Un), c)
    sp = 1.0 / (np.abs(proj) ** 2 + 1e-12)
    sp = sp / sp.max(1, keepdims=True)
    return _locate(np.einsum("pkf,pf->k", sp, wf), sc)


# name -> (callable, needs a mask)
BACKENDS = {
    "gcc_phat": (gcc_phat, False),
    "mgcc": (mgcc, True),
    "sr_snr": (sr_snr, True),
    "sv": (sv, True),
    "music": (music, False),
}


def estimate(name: str, Y, sc: Scenario, dist: float, M=None, *, idx=None) -> float:
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    fn, needs_mask = BACKENDS[name]
    if needs_mask and M is None:
        raise ValueError(f"backend {name!r} requires a mask")
    return fn(Y, sc, dist, M, idx=idx) if needs_mask else fn(Y, sc, dist, idx=idx)
