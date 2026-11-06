"""Corpus loading, STFT, diffuse-babble synthesis and mixture generation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .arrays import Scenario
from .rooms import all_rirs

_POOL: dict = {}


def speech(spk: str, n: int = config.SPEECH_POOL, data=None) -> list[np.ndarray]:
    """First ``n`` utterances of one CMU ARCTIC voice, in filename order."""
    import soundfile as sf

    root = config.arctic_dir(data) / spk
    key = (str(root), n)
    if key not in _POOL:
        files = sorted(root.glob("*.wav"))[:n]
        if not files:
            raise FileNotFoundError(
                f"no wav files under {root}. Run `oracle-doa download` first "
                f"or point --data at an existing corpus.")
        _POOL[key] = [sf.read(str(p), dtype="float32")[0] for p in files]
    return _POOL[key]


def fit(x: np.ndarray, dur: float) -> np.ndarray:
    n = int(dur * config.FS)
    return np.pad(x, (0, max(0, n - len(x))))[:n]


_WINDOW = None


def stft(x: np.ndarray) -> np.ndarray:
    """Complex STFT, shape (..., frames, F), periodic Hann window."""
    global _WINDOW
    if _WINDOW is None:
        _WINDOW = np.hanning(config.NFFT + 1)[:-1].astype(np.float32)
    d = 1 + (x.shape[-1] - config.NFFT) // config.HOP
    idx = np.arange(config.NFFT)[None, :] + config.HOP * np.arange(d)[:, None]
    return np.fft.rfft(x[..., idx] * _WINDOW, config.NFFT, axis=-1)


def logspec(Y: np.ndarray) -> np.ndarray:
    return np.log(np.abs(Y) ** 2 + 1e-8).astype(np.float32)


class BabbleBank:
    """Pre-convolved speech streams, one per (distance, T60, direction).

    A fresh diffuse-babble realisation then costs a random-window sum instead of
    one convolution per interferer position.  Each direction gets its own
    talker, which is what makes the babble diffuse rather than a single
    coherent interferer.
    """

    def __init__(self, sc: Scenario, rirs: dict | None = None, stream_len: float = 6.0,
                 seed: int = 0, data=None):
        from scipy.signal import fftconvolve

        self.sc = sc
        self.L = int(stream_len * config.FS)
        rirs = all_rirs(sc, data=data) if rirs is None else rirs
        rng = np.random.default_rng(seed)
        spans = [len(speech(s, data=data)) for s in config.BABBLE_SPEAKERS]
        pool_len = min(spans)                      # every talker must be reachable
        need = int(np.ceil(self.L / config.FS)) + 2

        self.bank = {}
        for dist in sc.dists:
            for t60 in sc.t60s:
                for ang in sc.angles:
                    h = rirs[(float(dist), float(t60), float(ang), "rev")]
                    segs = np.concatenate([
                        fit(speech(config.BABBLE_SPEAKERS[i % len(config.BABBLE_SPEAKERS)],
                                   data=data)[rng.integers(pool_len)], 1.0)
                        for i in range(need)])[:self.L]
                    self.bank[(float(dist), float(t60), float(ang))] = np.stack(
                        [fftconvolve(segs, h[m])[:self.L] for m in range(sc.n_mic)])

    def n_talkers(self, angles=None) -> int:
        sc = self.sc
        if angles is None:
            return len(sc.dists) * len(sc.angles)
        return len(angles)

    def sample(self, rng, t60: float, dur: float, dist: float | None = None,
               angles=None) -> np.ndarray:
        """Sum one random window per interferer position.

        ``angles=None`` uses every (distance, direction) grid point of the
        array; passing ``angles`` restricts to one distance and those
        directions.
        """
        sc = self.sc
        if angles is None:
            keys = [(d, float(a)) for d in sc.dists for a in sc.angles]
        else:
            d = sc.dists[0] if dist is None else dist
            keys = [(d, float(a)) for a in angles]
        L = int(dur * config.FS)
        out = np.zeros((sc.n_mic, L), np.float32)
        for k in keys:
            s = self.bank[(k[0], float(round(t60, 2)), k[1])]
            t0 = int(rng.integers(0, max(1, self.L - L)))
            out += s[:, t0:t0 + L]
        return out


def make_mixture(sc: Scenario, t60: float, angle: float, rng, bank: BabbleBank,
                 dist: float | None = None, spk: str = config.TARGET_SPK,
                 utt_idx: int = 0, angles=None, dur: float = config.DUR,
                 snr_db: float = -6.0, data=None) -> tuple:
    """(mixture, direct-path target, reverberant target), each (n_mic, N).

    The target is convolved with the reverberant and with the order-0 RIR; the
    babble is a fresh diffuse realisation scaled so that the reverberant target
    to babble ratio equals ``snr_db``.
    """
    from scipy.signal import fftconvolve

    r = all_rirs(sc, data=data)
    dist = sc.dists[0] if dist is None else dist
    t60 = float(round(t60, 2))
    h = r[(dist, t60, float(angle), "rev")]
    hd = r[(dist, t60, float(angle), "dir")]

    pool = speech(spk, data=data)
    tgt = pool[utt_idx % len(pool)]
    tgt = fit(np.roll(tgt, int(rng.integers(0, max(1, len(tgt) // 2)))), dur)

    n = int(dur * config.FS)
    rev = np.stack([fftconvolve(tgt, h[m])[:n] for m in range(sc.n_mic)])
    direct = np.stack([fftconvolve(tgt, hd[m])[:n] for m in range(sc.n_mic)])

    bab = bank.sample(rng, t60, dur, dist, angles)
    p_s, p_n = np.mean(rev ** 2), np.mean(bab ** 2)
    # scale so that 10*log10(P_s / P_n) == snr_db
    bab = bab * (np.sqrt(p_s / max(p_n, 1e-20)) * 10 ** (-snr_db / 20.0))
    y = rev + bab
    return y.astype(np.float32), direct.astype(np.float32), rev.astype(np.float32)
