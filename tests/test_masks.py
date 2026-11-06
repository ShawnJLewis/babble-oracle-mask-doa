import numpy as np

from oracle_doa import masks


def _stft_like(shape=(20, 257), seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_irm_is_one_when_the_direct_path_equals_the_mixture():
    Yd = _stft_like()
    irm, psm = masks.oracle_masks(Yd[None], Yd[None])
    assert len(irm) == 1 and len(psm) == 1
    assert np.allclose(irm[0], 1.0, atol=1e-5)
    assert np.allclose(psm[0], 1.0, atol=1e-5)


def test_irm_matches_the_closed_form():
    Ymix = _stft_like(seed=1)
    Yd = _stft_like(seed=2)
    irm, _ = masks.oracle_masks(Ymix[None], Yd[None])
    p = np.abs(Yd) ** 2
    n = np.abs(Ymix - Yd) ** 2
    assert np.allclose(irm[0], p / (p + n + 1e-12), atol=1e-6)


def test_masks_are_bounded_and_per_microphone():
    Ymix = _stft_like()
    Yd = _stft_like(seed=3)
    irm, psm = masks.oracle_masks(np.stack([Ymix, Ymix]), Yd[None])
    assert len(irm) == 2 and irm[0].shape == Ymix.shape
    for m in (irm, psm):
        for arr in m:
            assert arr.dtype == np.float32
            assert arr.min() >= 0.0 and arr.max() <= 1.0 + 1e-6


def test_select():
    irm, psm = masks.oracle_masks(_stft_like()[None], _stft_like(seed=4)[None])
    assert masks.select("irm", (irm, psm)) is irm
    assert masks.select("psm", (irm, psm)) is psm
