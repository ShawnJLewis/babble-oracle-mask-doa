import numpy as np
import pytest

from oracle_doa import arrays, backends, config


def _unit_mask(sc, Y, value=0.5):
    """A mask strictly inside (0, 1), so the noise covariance is non-singular."""
    return tuple(np.full((Y.shape[1], config.F), value, np.float32)
                 for _ in range(sc.n_mic))


def _direct_path_stft(sc, theta_deg, dist=1.5, frames=40, seed=0):
    """Synthetic array STFT of a single plane-wave-like source at theta_deg."""
    rng = np.random.default_rng(seed)
    src = sc.src_pos(theta_deg, dist)
    r = np.linalg.norm(sc.mic_pos - src, axis=1)
    tau = (r - r.min()) / config.C
    S = rng.standard_normal((frames, config.F)) + 1j * rng.standard_normal((frames, config.F))
    w = 2 * np.pi * (np.arange(config.F) / config.NFFT) * config.FS
    Y = np.stack([S * np.exp(-1j * w * tau[m]) for m in range(sc.n_mic)])
    return Y, tau


@pytest.mark.parametrize("name", list(backends.BACKENDS))
@pytest.mark.parametrize("theta", [-40.0, 0.0, 25.0])
def test_single_source_is_localised(name, theta):
    sc = arrays.get("a2")
    Y, _ = _direct_path_stft(sc, theta)
    est = backends.estimate(name, Y, sc, 1.5, _unit_mask(sc, Y))
    assert est in set(np.round(sc.grid, 6).tolist())
    assert abs(est - theta) <= 1.0


def test_masked_backends_need_a_mask():
    sc = arrays.get("a2")
    Y, _ = _direct_path_stft(sc, 10.0)
    for name, (_, needs_mask) in backends.BACKENDS.items():
        if needs_mask:
            with pytest.raises(ValueError):
                backends.estimate(name, Y, sc, 1.5, None)


def test_unknown_backend():
    with pytest.raises(KeyError):
        backends.estimate("esprit", np.zeros((2, 4, config.F), complex), arrays.get("a2"), 1.5)


def test_six_mic_ring_is_sharper_than_two_mics():
    """A larger aperture should return a peak closer to the truth on average."""
    errs = {}
    for key in ("a2", "spk6"):
        sc = arrays.get(key)
        e = []
        for theta in (-60.0, -30.0, 0.0, 30.0, 60.0):
            Y, _ = _direct_path_stft(sc, theta)
            e.append(abs(backends.estimate("mgcc", Y, sc, 1.5, _unit_mask(sc, Y)) - theta))
        errs[key] = float(np.mean(e))
    assert errs["spk6"] <= errs["a2"] + 1e-9
