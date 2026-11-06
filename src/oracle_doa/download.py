"""Fetch CMU ARCTIC utterances.

CMU ARCTIC is free for any purpose.  Only the voices this repository needs are
downloaded: one target voice and six babble voices.  Each file is fetched
individually, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config

BASE = ("http://festvox.org/cmu_arctic/cmu_arctic/"
        "cmu_us_{spk}_arctic/wav/arctic_{s}{i:04d}.wav")
SET_A = 450                     # utterances in set a; the rest live in set b

DEFAULT_QUOTA = {
    "slt": 200,                 # target voice
    "clb": 60,
    "bdl": 40, "rms": 40, "jmk": 40, "awb": 40, "ksp": 40,
}


def _url(spk: str, j: int) -> str:
    s, i = ("a", j) if j <= SET_A else ("b", j - SET_A)
    return BASE.format(spk=spk, s=s, i=i)


def _one(spk: str, j: int, dst_dir: Path, retries: int = 3) -> bool:
    s, i = ("a", j) if j <= SET_A else ("b", j - SET_A)
    dst = dst_dir / f"arctic_{s}{i:04d}.wav"
    if dst.exists() and dst.stat().st_size > 1000:
        return True
    for _ in range(retries):
        try:
            with urllib.request.urlopen(_url(spk, j), timeout=40) as r:
                data = r.read()
            if len(data) <= 1000:
                continue
            dst.write_bytes(data)
            return True
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    return False


def fetch(quota: dict | None = None, data=None, workers: int = 12,
          quiet: bool = False) -> dict:
    """Download the requested utterances. Returns ``{speaker: (ok, missing)}``."""
    quota = DEFAULT_QUOTA if quota is None else quota
    root = config.arctic_dir(data)
    jobs = []
    for spk, n in quota.items():
        d = root / spk
        d.mkdir(parents=True, exist_ok=True)
        jobs += [(spk, j, d) for j in range(1, n + 1)]

    results: dict[str, list[bool]] = {spk: [] for spk in quota}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [(spk, pool.submit(_one, spk, j, d)) for spk, j, d in jobs]
        for spk, fut in futures:
            results[spk].append(bool(fut.result()))

    out = {}
    for spk, ok in results.items():
        missing = int(len(ok) - sum(ok))
        out[spk] = (int(sum(ok)), missing)
        if not quiet:
            print(f"  {spk}: {sum(ok)} ok, {missing} missing")
    if not quiet:
        print(f"  corpus root: {root}")
    return out
