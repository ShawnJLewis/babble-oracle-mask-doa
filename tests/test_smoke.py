import subprocess
import sys

import numpy as np
import pytest

from oracle_doa import arrays, audio, config, masks, rooms


def _mini_scenario():
    """Two directions, one anechoic condition: cheap enough for a test."""
    return arrays.Scenario(
        key="_test", room=(4.0, 4.0, 2.5), dists=(1.0,),
        mic_off=np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        t60s=np.array([0.0]), angles=np.array([-30.0, 0.0, 30.0]),
        targets=np.array([-30.0, 0.0, 30.0]), tol=5.0, step=1.0)


def _has_corpus():
    try:
        audio.speech(config.TARGET_SPK, n=5)
        return True
    except FileNotFoundError:
        return False


def _pyroomacoustics_works():
    """pyroomacoustics can abort the process, so probe it in a subprocess."""
    code = ("import numpy as np, pyroomacoustics as pra;"
            "r=pra.ShoeBox([4,4,2.5],fs=16000,max_order=0);"
            "r.add_source([1.0,1.0,1.0]);"
            "r.add_microphone_array(pra.MicrophoneArray(np.array([[2.0,2.0,1.0]]).T,16000));"
            "r.compute_rir();"
            "print(int(bool(r.rir[0][0].size)))")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return out.returncode == 0 and out.stdout.strip().endswith(b"1")


pytestmark = [
    pytest.mark.skipif(not _has_corpus(),
                       reason="CMU ARCTIC corpus not present; run `oracle-doa download`"),
    pytest.mark.skipif(not _pyroomacoustics_works(),
                       reason="pyroomacoustics cannot compute a RIR on this machine"),
]


def test_end_to_end_on_a_toy_array(tmp_path):
    sc = _mini_scenario()
    rirs = rooms.all_rirs(sc, order=0, data=tmp_path)
    assert len(rirs) == 6                      # 3 angles x 2 kinds
    bank = audio.BabbleBank(sc, rirs, data=None)
    rng = np.random.default_rng(0)
    y, direct, rev = audio.make_mixture(sc, 0.0, 0.0, rng, bank, dist=1.0,
                                        snr_db=-6.0)
    assert y.shape == direct.shape == rev.shape == (sc.n_mic, int(config.DUR * config.FS))

    Y, Yd = audio.stft(y), audio.stft(direct)
    k = min(Y.shape[1], Yd.shape[1])
    M = dict(zip(masks.KINDS, masks.oracle_masks(Y[:, :k], Yd[:, :k])))
    assert M["irm"][0].shape == (k, config.F)
