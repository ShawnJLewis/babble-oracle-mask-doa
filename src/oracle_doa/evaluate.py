"""Experiment driver: build conditions, run estimators, score and serialise."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import __version__, arrays, audio, backends, config, masks, rooms


@dataclass(frozen=True)
class Condition:
    label: str
    dist: float
    t60: float
    snr_db: float

    def key(self) -> str:
        return f"d{self.dist:g}_t{self.t60:.2f}_s{self.snr_db:g}"


def conditions(sc: arrays.Scenario, dists=None, t60s=None,
               snrs: tuple = (-6.0,)) -> list[Condition]:
    dists = tuple(sc.dists) if dists is None else tuple(dists)
    t60s = tuple(float(t) for t in sc.t60s) if t60s is None else tuple(t60s)
    out = []
    for dist in dists:
        for t60 in t60s:
            for snr in snrs:
                c = Condition("", float(dist), float(t60), float(snr))
                out.append(Condition(c.key(), c.dist, c.t60, c.snr_db))
    return out


@dataclass(frozen=True)
class RunSpec:
    array: str = "a2"
    mask_names: tuple = ("irm", "psm")
    backend_names: tuple = tuple(backends.BACKENDS)
    n_utt: int = 200
    spk: str = config.TARGET_SPK
    seed: int = 7
    dists: tuple | None = None
    t60s: tuple | None = None
    snrs: tuple = (-6.0,)
    rir_order: int = config.RIR_ORDER
    data: str | None = None
    out: str | None = None
    quiet: bool = False


def circular_error(est_deg: float, ref_deg: float) -> float:
    """Signed angular error folded into [-180, 180]."""
    d = (est_deg - ref_deg + 180.0) % 360.0 - 180.0
    return float(d)


def _systems(spec: RunSpec) -> list[tuple[str, str]]:
    """(mask name or 'none', backend name) pairs to evaluate."""
    out = [("none", "gcc_phat"), ("none", "music")]
    for m in spec.mask_names:
        for b in spec.backend_names:
            if backends.BACKENDS[b][1]:
                out.append((m, b))
            elif b == "music":
                out.append((m, b))
    seen, uniq = set(), []
    for pair in out:
        if pair not in seen:
            seen.add(pair)
            uniq.append(pair)
    return uniq


NOTES = [
    "Oracle IRM = |Ydir|^2 / (|Ydir|^2 + |Ymix - Ydir|^2); the denominator contains "
    "late reverberation and babble, so it is a direct-path dominance mask rather "
    "than a speech-versus-noise mask.",
    "Oracle masks are built from the simulated direct-path signal and are not "
    "available to a real system; the numbers are upper bounds, not achievable "
    "accuracy.",
    "All microphones of the array are used. The DOA search covers the front "
    "half-plane [-90, 90] deg with a 1 deg step.",
    "One RNG drives the whole run (target direction, target offset, babble "
    "windows), so results depend on n_utt and on the condition order.",
]


def run(spec: RunSpec) -> dict:
    sc = arrays.get(spec.array)
    conds = conditions(sc, spec.dists, spec.t60s, spec.snrs)
    systems = _systems(spec)

    rirs = rooms.all_rirs(sc, spec.rir_order, spec.data)
    bank = audio.BabbleBank(sc, rirs, data=spec.data)
    babble_angles = sc.babble_angles
    n_talkers = bank.n_talkers(babble_angles)

    errs: dict[tuple[str, str], list[float]] = {s: [] for s in systems}
    per_cond = []
    rng = np.random.default_rng(spec.seed)

    for cond in conds:
        cond_errs = {s: [] for s in systems}
        for i in range(spec.n_utt):
            ang = float(sc.targets[rng.integers(len(sc.targets))])
            y, direct, _ = audio.make_mixture(
                sc, cond.t60, ang, rng, bank, dist=cond.dist, spk=spec.spk,
                utt_idx=i, angles=babble_angles, snr_db=cond.snr_db, data=spec.data)
            Y, Yd = audio.stft(y), audio.stft(direct)
            k = min(Y.shape[1], Yd.shape[1])
            Y, Yd = Y[:, :k], Yd[:, :k]
            M = dict(zip(masks.KINDS, masks.oracle_masks(Y, Yd)))

            for mask_name, backend_name in systems:
                est = backends.estimate(backend_name, Y, sc, cond.dist,
                                        M.get(mask_name))
                cond_errs[(mask_name, backend_name)].append(
                    circular_error(est, ang))

        metrics = {}
        for s, vals in cond_errs.items():
            e = np.abs(np.array(vals))
            metrics[f"{s[0]}/{s[1]}"] = {
                "acc_tol": round(100.0 * float(np.mean(e <= sc.tol)), 1),
                "acc_5": round(100.0 * float(np.mean(e <= 5.0)), 1),
                "acc_7p5": round(100.0 * float(np.mean(e <= 7.5)), 1),
                "acc_10": round(100.0 * float(np.mean(e <= 10.0)), 1),
                "mae_deg": round(float(e.mean()), 2),
                "rmse_deg": round(float(np.sqrt((e ** 2).mean())), 2),
                "n": len(vals),
            }
        for s, vals in cond_errs.items():
            errs[s].extend(vals)

        per_cond.append({
            "label": cond.label,
            "dist_m": cond.dist,
            "t60_s": cond.t60,
            "snr_db": cond.snr_db,
            "drr_db": round(rooms.drr(sc, cond.t60, cond.dist, data=spec.data), 2),
            "n_utt": spec.n_utt,
            "metrics": metrics,
        })
        if not spec.quiet:
            masked = [v["acc_tol"] for k, v in metrics.items() if not k.startswith("none/")]
            best = max(masked) if masked else metrics["none/gcc_phat"]["acc_tol"]
            print(f"  {cond.label}: {n_talkers} talkers, best masked ACC@tol "
                  f"{best:.1f}% ({spec.n_utt} utt)", flush=True)

    summary = {}
    for s, vals in errs.items():
        e = np.abs(np.array(vals))
        summary[f"{s[0]}/{s[1]}"] = {
            "acc_tol": round(100.0 * float(np.mean(e <= sc.tol)), 1),
            "acc_5": round(100.0 * float(np.mean(e <= 5.0)), 1),
            "acc_7p5": round(100.0 * float(np.mean(e <= 7.5)), 1),
            "acc_10": round(100.0 * float(np.mean(e <= 10.0)), 1),
            "mae_deg": round(float(e.mean()), 2),
            "rmse_deg": round(float(np.sqrt((e ** 2).mean())), 2),
            "n": len(vals),
        }

    out = {
        "schema_version": 1,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "package_version": __version__,
        "array": sc.describe(),
        "setup": {
            "spk": spec.spk,
            "n_utt": spec.n_utt,
            "seed": spec.seed,
            "snr_db": [float(s) for s in spec.snrs],
            "masks": list(spec.mask_names),
            "backends": list(spec.backend_names),
            "rir_order": spec.rir_order,
            "rir_cache": rooms.cache_info(sc, spec.rir_order, spec.data),
            "babble": {
                "n_talkers": n_talkers,
                "mode": "test-grid" if babble_angles is not None else "full-grid",
            },
        },
        "conditions": per_cond,
        "summary": summary,
        "notes": NOTES,
    }

    path = Path(spec.out) if spec.out else Path("results") / f"{sc.key}_oracle.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    if not spec.quiet:
        print(f"  wrote {path}")
    return out
