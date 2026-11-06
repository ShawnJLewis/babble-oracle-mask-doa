"""Global constants and path resolution.

Every module reads the signal-chain constants from here, so there is a single
place to change the sampling rate or the STFT geometry.
"""
from __future__ import annotations

import os
from pathlib import Path

FS = 16000                      # sampling rate, Hz
DUR = 2.4                       # mixture length, s
NFFT = 512
HOP = 128
F = NFFT // 2 + 1               # 257 frequency bins
C = 343.0                       # speed of sound, m/s

RIR_ORDER = 60                  # image-source reflection order
TARGET_SPK = "slt"              # CMU ARCTIC voice used as the target talker
BABBLE_SPEAKERS = ("bdl", "rms", "jmk", "awb", "ksp", "clb")
SPEECH_POOL = 400               # utterances kept in memory per speaker

ENV_DATA = "ORACLE_DOA_DATA"


def data_dir(cli: str | os.PathLike | None = None) -> Path:
    """Resolve the data directory.

    Priority: explicit argument, then ``$ORACLE_DOA_DATA``, then ``./data``.
    """
    if cli is not None:
        return Path(cli).expanduser().resolve()
    env = os.environ.get(ENV_DATA)
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / "data").resolve()


def arctic_dir(cli: str | os.PathLike | None = None) -> Path:
    return data_dir(cli) / "arctic"
