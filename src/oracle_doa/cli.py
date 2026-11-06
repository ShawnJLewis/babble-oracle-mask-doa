"""Command-line interface."""
from __future__ import annotations

import argparse
import json
import sys

from . import arrays, backends, config, rooms
from .download import DEFAULT_QUOTA, fetch
from .evaluate import RunSpec, conditions, run


def _list(text: str) -> tuple:
    return tuple(x.strip() for x in str(text).split(",") if x.strip())


def _floats(text: str) -> tuple:
    return tuple(float(v) for v in _list(text))


def _check_grid(values, allowed, what: str) -> tuple:
    """Map requested values onto the declared grid, refusing anything else."""
    if values is None:
        return tuple(float(v) for v in allowed)
    out = []
    for v in values:
        match = [a for a in allowed if abs(float(a) - float(v)) < 1e-9]
        if not match:
            raise SystemExit(
                f"error: {what}={v:g} is not in the declared grid "
                f"{[float(a) for a in allowed]}. Either pick a declared value, or "
                f"edit the scenario in src/oracle_doa/arrays.py and delete the "
                f"stale RIR cache before rebuilding it.")
        out.append(float(match[0]))
    return tuple(out)


def cmd_download(a) -> int:
    quota = dict(DEFAULT_QUOTA)
    if a.speakers:
        quota = {}
        for part in _list(a.speakers):
            spk, _, n = part.partition(":")
            quota[spk] = int(n) if n else DEFAULT_QUOTA.get(spk, 40)
    print("downloading CMU ARCTIC utterances")
    fetch(quota, data=a.data, workers=a.workers)
    return 0


def cmd_rirs(a) -> int:
    keys = list(arrays.SCENARIOS) if a.array == "all" else [a.array]
    for key in keys:
        sc = arrays.get(key)
        rirs = rooms.all_rirs(sc, a.order, a.data, progress=not a.quiet)
        print(f"{key}: {len(rirs)} RIRs, cache {rooms.cache_path(sc, a.order, a.data)}")
        if a.verify:
            for t60 in sc.t60s:
                for dist in sc.dists:
                    print(f"  d={dist:g} m, T60={t60:.2f} s: "
                          f"DRR {rooms.drr(sc, t60, dist, data=a.data):+.2f} dB")
    return 0


def cmd_arrays(a) -> int:
    rows = [arrays.get(k).describe() for k in arrays.SCENARIOS]
    if a.json:
        print(json.dumps(rows, indent=1))
        return 0
    head = ("array", "mics", "min d (m)", "aperture", "f< c/2d (Hz)",
            "room (m)", "dists (m)", "T60 grid (s)", "tol")
    print("{:<7}{:>5}{:>10}{:>9}{:>13}  {:<12}{:<11}{:<30}{}".format(*head))
    for r in rows:
        print("{:<7}{:>5}{:>10.3f}{:>9.3f}{:>13.0f}  {:<12}{:<11}{:<30}{:.1f}".format(
            r["key"], r["n_mic"], r["min_spacing_m"], r["aperture_m"], r["alias_hz"],
            "x".join(f"{v:g}" for v in r["room"]),
            ",".join(f"{v:g}" for v in r["dists_m"]),
            ",".join(f"{v:.2f}" for v in r["t60s"]), r["tol_deg"]))
    return 0


def cmd_run(a) -> int:
    keys = list(arrays.SCENARIOS) if a.array == "all" else [a.array]
    if a.array == "all" and a.out:
        raise SystemExit("error: --out is only valid for a single --array")

    if a.mask in ("none", "None", ""):
        mask_names: tuple = ()
    else:
        mask_names = _list(a.mask)
        for name in mask_names:
            if name not in ("irm", "psm"):
                raise SystemExit(f"error: unknown mask {name!r}; choose irm, psm or none")

    backend_names = tuple(backends.BACKENDS) if a.backend == "all" else _list(a.backend)
    for name in backend_names:
        if name not in backends.BACKENDS:
            raise SystemExit(f"error: unknown backend {name!r}; "
                             f"choose from {sorted(backends.BACKENDS)}")

    for key in keys:
        sc = arrays.get(key)
        spec = RunSpec(
            array=key,
            mask_names=mask_names,
            backend_names=backend_names,
            n_utt=a.utt,
            spk=a.spk,
            seed=a.seed,
            dists=_check_grid(_floats(a.dist), sc.dists, "--dist") if a.dist else None,
            t60s=_check_grid(_floats(a.t60), sc.t60s, "--t60") if a.t60 else None,
            snrs=_floats(a.snr),
            rir_order=a.rir_order,
            data=a.data,
            out=a.out if len(keys) == 1 else None,
            quiet=a.quiet,
        )
        print(f"[{key}] {sc.n_mic} mics, "
              f"{len(conditions(sc, spec.dists, spec.t60s, spec.snrs))} conditions, "
              f"{a.utt} utterances, masks {spec.mask_names or ('none',)}")
        out = run(spec)
        if not a.quiet:
            print(f"[{key}] summary (ACC@{sc.tol:g} deg over all conditions)")
            for name, m in sorted(out["summary"].items(), key=lambda kv: -kv[1]["acc_tol"]):
                print(f"    {name:<14} {m['acc_tol']:5.1f}   MAE {m['mae_deg']:5.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="oracle-doa",
        description="Oracle time-frequency mask upper bounds for classical DOA "
                    "estimators on five microphone arrays in diffuse babble.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch CMU ARCTIC utterances")
    d.add_argument("--speakers", default=None,
                   help="per-speaker quota, e.g. slt:200,clb:60,bdl:40")
    d.add_argument("--data", default=None, help="data directory")
    d.add_argument("--workers", type=int, default=12, help="parallel downloads")
    d.set_defaults(func=cmd_download)

    r = sub.add_parser("rirs", help="build or inspect the RIR cache")
    r.add_argument("--array", default="all", help="array key or 'all'")
    r.add_argument("--order", type=int, default=config.RIR_ORDER,
                   help="image-source reflection order")
    r.add_argument("--data", default=None, help="data directory")
    r.add_argument("--verify", action="store_true",
                   help="print the DRR of every condition")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_rirs)

    g = sub.add_parser("arrays", help="print the array table")
    g.add_argument("--json", action="store_true", help="machine-readable output")
    g.set_defaults(func=cmd_arrays)

    p = sub.add_parser("run", help="run the evaluation")
    p.add_argument("--array", default="a2", help="array key or 'all'")
    p.add_argument("--mask", default="irm,psm", help="irm, psm, irm,psm or none")
    p.add_argument("--backend", default="all",
                   help="comma list of gcc_phat, mgcc, sr_snr, sv, music, or all")
    p.add_argument("--utt", type=int, default=200, help="utterances per condition")
    p.add_argument("--spk", default=config.TARGET_SPK, help="ARCTIC target speaker")
    p.add_argument("--snr", default="-6", help="babble SNR in dB, comma list")
    p.add_argument("--t60", default=None, help="subset of the declared T60 grid")
    p.add_argument("--dist", default=None, help="subset of the declared distances")
    p.add_argument("--seed", type=int, default=7,
                   help="one RNG drives the whole run")
    p.add_argument("--rir-order", dest="rir_order", type=int, default=config.RIR_ORDER,
                   help="image-source reflection order")
    p.add_argument("--data", default=None, help="data directory")
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_run)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
