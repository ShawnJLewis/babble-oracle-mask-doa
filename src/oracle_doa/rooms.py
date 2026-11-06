"""Room impulse responses from the image-source method.

Wall absorption follows the Sabine relation used by the reference generator,
``alpha = 0.161 V / (T60 S)``.  Direct-path RIRs are a separate order-0 run, so
the direct sound and the reverberant tail can be separated exactly.

RIRs are cached on disk as ``rirs_<array>_o<order>.npz`` keyed by
``"<dist>|<t60>|<angle>|<kind>"`` with ``kind`` in ``{"rev", "dir"}``.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from . import config
from .arrays import Scenario


def alpha_of(t60: float, room: tuple) -> float:
    """Sabine wall absorption for a given reverberation time."""
    if t60 <= 0:
        return 0.0
    V = float(np.prod(room))
    S = 2.0 * (room[0] * room[1] + room[1] * room[2] + room[0] * room[2])
    return float(min(0.161 * V / (t60 * S), 0.999))


def _rir(sc: Scenario, alpha: float, dist: float, ang: float, kind: str,
         order: int = config.RIR_ORDER) -> np.ndarray:
    import pyroomacoustics as pra

    if kind == "dir" or alpha <= 1e-6:
        room = pra.ShoeBox(list(sc.room), fs=config.FS, max_order=0)
    else:
        room = pra.ShoeBox(list(sc.room), fs=config.FS,
                           materials=pra.Material(alpha), max_order=order)
    room.add_source(sc.src_pos(ang, dist).tolist())
    room.add_microphone_array(pra.MicrophoneArray(sc.mic_pos.T, config.FS))
    room.compute_rir()
    rs = [np.asarray(room.rir[m][0]) for m in range(sc.n_mic)]
    L = max(len(x) for x in rs)
    return np.stack([np.pad(x, (0, L - len(x))) for x in rs]).astype(np.float32)


def cache_path(sc: Scenario, order: int = config.RIR_ORDER, data=None) -> Path:
    return config.data_dir(data) / f"rirs_{sc.key}_o{order}.npz"


_CACHE: dict = {}


def all_rirs(sc: Scenario, order: int = config.RIR_ORDER, data=None,
             progress: bool = False) -> dict:
    """``{(dist, t60, angle, kind): rir[n_mic, L]}`` for one array.

    Reads the disk cache when present, otherwise builds every RIR declared by
    the scenario and writes the cache atomically.
    """
    path = cache_path(sc, order, data)
    ck = (sc.key, order, str(path))
    if ck in _CACHE:
        return _CACHE[ck]

    if path.exists():
        z = np.load(path)
        out = {}
        for k in z.files:
            d, t, a, kind = k.split("|")
            out[(float(d), float(t), float(a), kind)] = z[k]
        if progress:
            print(f"  loaded {len(out)} RIRs from {path}")
    else:
        out = {}
        total = len(sc.dists) * len(sc.t60s) * len(sc.angles) * 2
        n = 0
        for dist in sc.dists:
            for t60 in sc.t60s:
                alpha = alpha_of(float(t60), sc.room)
                for ang in sc.angles:
                    for kind in ("rev", "dir"):
                        out[(float(dist), float(t60), float(ang), kind)] = _rir(
                            sc, alpha, float(dist), float(ang), kind, order)
                        n += 1
                if progress:
                    print(f"  {sc.key}: {n}/{total} RIRs", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(
            tmp, **{f"{d}|{t}|{a}|{k}": v for (d, t, a, k), v in out.items()})
        os.replace(tmp, path)
        if progress:
            print(f"  wrote {len(out)} RIRs to {path}")

    _CACHE[ck] = out
    return out


def drr(sc: Scenario, t60: float, dist: float | None = None, angle: float = 0.0,
        data=None) -> float:
    """Direct-to-reverberant ratio, dB, computed from the first microphone."""
    r = all_rirs(sc, data=data)
    dist = sc.dists[0] if dist is None else dist
    t60 = float(round(t60, 2))
    h = r[(dist, t60, float(angle), "rev")][0]
    hd = r[(dist, t60, float(angle), "dir")][0]
    pd = float(np.sum(hd ** 2))
    return 10 * np.log10(max(pd, 1e-15) / max(float(np.sum(h ** 2)) - pd, 1e-15))


def cache_info(sc: Scenario, order: int = config.RIR_ORDER, data=None) -> dict:
    """Size and digest of the RIR cache, for the result metadata.

    Only the file name is recorded, so a result file does not carry a local
    path from the machine that produced it.
    """
    path = cache_path(sc, order, data)
    info = {"file": path.name, "order": order, "n_rirs": None, "sha256": None}
    if not path.exists():
        return info
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    info["n_rirs"] = len(all_rirs(sc, order, data))
    info["sha256"] = h.hexdigest()
    return info
