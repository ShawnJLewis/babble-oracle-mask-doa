import numpy as np
import pytest

from oracle_doa import arrays, config


@pytest.mark.parametrize("key", list(arrays.SCENARIOS))
def test_geometry_is_consistent(key):
    sc = arrays.get(key)
    assert sc.n_mic == len(sc.mic_off)
    assert sc.mic_off.shape[1] == 3
    assert sc.min_spacing > 0
    assert sc.aperture >= sc.min_spacing
    assert abs(sc.alias_hz - config.C / (2 * sc.min_spacing)) < 1e-9
    assert sc.tol > 0


def test_search_grid_and_tolerances():
    a2 = arrays.get("a2")
    assert a2.grid[0] == -90.0 and a2.grid[-1] == 90.0
    assert len(a2.grid) == 181
    assert a2.tol == 5.0
    assert arrays.get("car4").tol == 10.0


def test_known_spacings():
    assert arrays.get("a2").min_spacing == pytest.approx(0.2, abs=1e-9)
    assert arrays.get("b8").min_spacing == pytest.approx(0.04, abs=1e-9)
    assert arrays.get("spk6").min_spacing == pytest.approx(0.045, abs=1e-6)
    assert arrays.get("car4").min_spacing == pytest.approx(0.6, abs=1e-9)
    assert arrays.get("a2").alias_hz == pytest.approx(857.5, abs=0.1)


def test_tau_matches_plane_wave_in_the_far_field():
    sc = arrays.get("a2")
    tau = sc.tau(1e6, None)                     # far field, broadside
    assert tau.shape == (1, len(sc.grid))
    i = int(np.argmin(np.abs(sc.grid)))         # theta = 0
    assert abs(tau[0, i]) < 1e-4


def test_babble_grid_selection():
    assert arrays.get("a2").babble_angles is not None
    assert len(arrays.get("a2").babble_angles) == 37
    assert arrays.get("b8").babble_angles is None
