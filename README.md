# Oracle-mask DOA upper bounds on five microphone arrays

This repository simulates diffuse babble in reverberant rooms, derives oracle
time-frequency masks from the direct-path signal, and feeds both into classical
direction-of-arrival (DOA) estimators.  It answers one question per array:

> how far can a time-frequency mask carry a classical DOA estimator, when the
> mask is perfect and the babble is diffuse?

The answer is an **upper bound**.  The masks use the direct-path signal, which a
real system never has, so the numbers describe what a mask-based front end can
do at best, not what any deployed system achieves.

Pipeline:

```
CMU ARCTIC wav  ->  image-source RIR  ->  reverberant target + diffuse babble
                                         |
                                         +-> mixture, direct path
                                                   |
                                    oracle IRM / PSM masks
                                                   |
                       GCC-PHAT | MGCC | SR-SNR | SV | MUSIC
                                                   |
                                    angle estimate -> ACC, MAE
```

No training is involved.  Five arrays are registered, from a 2-microphone
reference to a car cabin with non-coplanar elements.

## Install

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

Requires Python 3.10 or newer, `numpy`, `scipy`, `soundfile` and
`pyroomacoustics`.  Everything runs on CPU; a full a2 evaluation takes a few
minutes on one core.

Without installing, run from the source tree with `python -m oracle_doa`, adding
`PYTHONPATH=src` (`$env:PYTHONPATH="src"` in PowerShell) so the package is found.

## Quick start

```bash
oracle-doa download                    # CMU ARCTIC, about 50 MB, 460 files
oracle-doa rirs --array a2             # build the RIR cache (slow, see Cost)
oracle-doa run --array a2 --utt 200    # evaluate, writes results/a2_oracle.json
```

A single condition is much faster, once the cache exists:

```bash
oracle-doa run --array a2 --t60 0.2 --utt 20 --mask psm --backend mgcc,sr_snr
```

```
[a2] 2 mics, 1 conditions, 20 utterances, masks ('psm',)
  d1.5_t0.20_s-6: 37 talkers, best masked ACC@tol 100.0% (20 utt)
```

`--array all` walks the five arrays in order.  `oracle-doa arrays` prints the
geometry table used in the next section.

## Arrays

| key | mics | geometry | min spacing | aperture | alias-free `f < c/2d` | room (m) | distances | T60 grid (s) | tolerance |
|---|---|---|---|---|---|---|---|---|---|
| `a2` | 2 | linear along x, ±0.1 m | 0.20 m | 0.20 m | < 858 Hz | 8x8x3 | 1.5 m | 0, 0.2, 0.4, 0.6, 0.8, 1.0 | 5 deg |
| `b8` | 8 | linear, MIRD 4-4-4-8-4-4-4 cm | 0.04 m | 0.32 m | < 4288 Hz | 6x6x2.4 | 1.0, 2.0 m | 0.16, 0.36, 0.61 | 7.5 deg |
| `spk6` | 6 | ring, r = 45 mm, ceiling height | 0.045 m | 0.09 m | < 3811 Hz | 6x6x2.4 | 1.0, 2.0 m | 0.16, 0.36, 0.61 | 7.5 deg |
| `rob5` | 5 | ring, r = 110 mm, 1.15 m below source | 0.129 m | 0.209 m | < 1326 Hz | 6x6x2.4 | 1.0, 2.0 m | 0.16, 0.36, 0.61 | 7.5 deg |
| `car4` | 4 | non-coplanar, 2 pillars + 2 roof lights | 0.600 m | 2.00 m | < 286 Hz | 2.8x1.6x1.4 | 0.8 m | 0.16, 0.36, 0.61 | 10 deg |

The DOA search covers the front half-plane, -90 to +90 degrees, in 1 degree
steps.  A linear or ring array cannot distinguish a source from its mirror image
behind it, so back-half directions are out of scope.

The alias-free band matters: `car4` keeps only about 9 of the 257 bins free of
phase wrapping, `a2` about 27, `spk6` about 122.  Its 10 degree tolerance is a
consequence of that, not a relaxed criterion.

## Conditions and babble

| key | conditions | how they are built | babble talkers |
|---|---|---|---|
| `a2` | 6 | 1 distance x 6 T60 | 37 (5 degree grid at 1.5 m) |
| `b8` | 6 | 2 distances x 3 T60 | 26 (2 distances x 13 directions) |
| `spk6` | 6 | 2 distances x 3 T60 | 26 |
| `rob5` | 6 | 2 distances x 3 T60 | 26 |
| `car4` | 3 | 1 distance x 3 T60 | 13 |

Each interferer position gets its own talker, which is what makes the babble
diffuse.  Per condition, utterances are drawn at a fixed babble SNR, -6 dB by
default (`--snr`), measured against the reverberant target.

RIRs come from the image-source method.  Wall absorption is set by the Sabine
relation `alpha = 0.161 V / (T60 S)`, so T60 is a controlled input rather than a
measured output.  Direct-path RIRs are a separate order-0 run, which is what
makes the oracle masks exact.

## Oracle masks

For every microphone,

```
IRM = |Yd|^2 / (|Yd|^2 + |Ymix - Yd|^2)
PSM = clip( IRM * cos( angle(Ymix) - angle(Yd) ), 0, 1 )
```

`Yd` is the STFT of the direct-path target, `Ymix` the STFT of the mixture.  The
denominator of the IRM therefore contains late reverberation as well as babble:
this is a **direct-path dominance** mask, not a speech-versus-noise mask.
Section 8 of the paper cited below uses the same construction.

The masks are computed independently per microphone, so a backend combines them
as `M_p * M_q` for a pair `(p, q)`.

## Backends

| name | mask | statistic | note |
|---|---|---|---|
| `gcc_phat` | no | inter-channel phase, uniform weight | reference, no mask |
| `mgcc` | yes | inter-channel phase weighted by `M_p M_q` | mask-weighted GCC-PHAT |
| `sr_snr` | yes | MVDR output SNR per pair, signal and noise covariance from `M` and `(1-M)` | steered-response SNR |
| `sv` | yes | principal eigenvector phase of each 2x2 mask-weighted covariance | steering-vector DOA |
| `music` | optional | noise subspace of the complementary covariance | per-band normalised before summation |

Every estimator scores the whole direction grid with one einsum over
`(pairs, directions, bins)`, so the cost does not grow with the number of
candidate directions.  The steering phases come from the microphone coordinates
and the source distance only; nothing learned enters the search.

## CLI reference

| command / flag | default | meaning |
|---|---|---|
| `oracle-doa download` | slt:200, clb:60, five others:40 | fetch CMU ARCTIC |
| `oracle-doa rirs --array` | `all` | build or inspect the RIR cache |
| `oracle-doa rirs --verify` | off | print the DRR of every condition |
| `oracle-doa arrays [--json]` | table | print the geometry table |
| `oracle-doa run --array` | `a2` | array key or `all` |
| `--mask` | `irm,psm` | `irm`, `psm`, both, or `none` |
| `--backend` | `all` | comma list of the five names |
| `--utt` | 200 | utterances per condition |
| `--snr` | -6 | babble SNR in dB, comma list |
| `--t60`, `--dist` | declared grid | subsets of the scenario grid |
| `--seed` | 7 | one RNG drives the whole run |
| `--rir-order` | 60 | image-source reflection order |
| `--data` | `$ORACLE_DOA_DATA` or `./data` | corpus and RIR cache |
| `--out` | `results/<array>_oracle.json` | output path |

`--t60` and `--dist` must lie on the declared grid.  The RIR cache is built for
the whole grid at once, so an off-grid value has no impulse response to read.

Results depend on `--utt` and on the condition order, because one RNG feeds the
whole run.  Compare settings at equal `--utt` and `--seed`, or compare only the
aggregated statistics.

## Output

`results/<array>_oracle.json` holds the array description, the setup, one block
per condition, and a summary over all conditions.  Each metric block is keyed
`"<mask>/<backend>"` and carries:

| field | meaning |
|---|---|
| `acc_tol` | percentage within the array's own tolerance |
| `acc_5`, `acc_7p5`, `acc_10` | percentage within a fixed tolerance, comparable across arrays |
| `mae_deg`, `rmse_deg` | circular mean and RMS of the absolute angular error |
| `n` | utterances behind the number |

`examples/results/` holds one short run and one full a2 run, for reference.

### Example: array `a2`, 200 utterances per condition

Accuracy within the 5 degree tolerance, percent.  DRR is the direct-to-reverberant
ratio of the first microphone; at T60 = 0 there is no reverberation and the DRR is
formally unbounded.

| system | 0.00 s | 0.20 s | 0.40 s | 0.60 s | 0.80 s | 1.00 s | mean | MAE |
|---|---|---|---|---|---|---|---|---|
| DRR (dB) | +146.4 | +4.0 | -2.3 | -4.8 | -6.5 | -7.7 | | |
| `gcc_phat`, no mask | 54.5 | 37.5 | 37.5 | 16.0 | 13.0 | 14.5 | 28.8 | 32.86 |
| `music`, no mask | 74.0 | 63.5 | 45.0 | 16.0 | 13.5 | 16.0 | 38.0 | 30.32 |
| `mgcc`, IRM | 100.0 | 99.0 | 99.5 | 98.5 | 97.0 | 98.0 | 98.7 | 0.36 |
| `sr_snr`, IRM | 100.0 | 99.5 | 100.0 | 100.0 | 99.5 | 99.0 | 99.7 | 0.34 |
| `sv`, IRM | 100.0 | 98.5 | 99.5 | 97.5 | 95.0 | 97.5 | 98.0 | 0.45 |
| `mgcc`, PSM | 100.0 | 99.0 | 100.0 | 100.0 | 99.5 | 100.0 | 99.8 | 0.20 |
| `sr_snr`, PSM | 100.0 | 100.0 | 100.0 | 100.0 | 99.5 | 100.0 | 99.9 | 0.23 |
| `sv`, PSM | 100.0 | 99.0 | 100.0 | 100.0 | 99.0 | 99.5 | 99.6 | 0.29 |

The gap between the masked and the unmasked rows is the size of the upper bound.
Masked and unmasked `music` come out identical on every array: MUSIC is evaluated
pair by pair, each pair carries a one-dimensional signal subspace, and a positive
mask weight reweights bands without moving the per-band peak.

## Data

Nothing but code is committed here.  `data/` and every audio, cache or weight
file are ignored by `.gitignore`.

**Target and babble speech: CMU ARCTIC.** Free for any purpose, 16 kHz,
multi-speaker.  Project page <http://www.festvox.org/cmu_arctic/>.  Individual
utterances are fetched from

```
http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_{spk}_arctic/wav/arctic_{set}{i:04d}.wav
```

with `set` in `{a, b}` and `i` starting at 1.  `oracle-doa download` requests 200
utterances of `slt` and 40 to 60 of the six babble voices; that quota decides the
pool size, and the pool size decides how many distinct utterances a condition
can draw (`--utt` must not exceed it).

Expected layout:

```
data/
  arctic/slt/arctic_a0001.wav ...
  arctic/{bdl,rms,jmk,awb,ksp,clb}/arctic_a0001.wav ...
  rirs_a2_o60.npz           # generated by `oracle-doa rirs`
```

**The IEEE corpus is not distributed here.**  The 720-sentence recordings
(IEEE Recommended Practice for Speech Quality Measurements, IEEE 297-1969) have
no public download; they are obtained through IEEE or an authorised channel.
This repository needs no specific corpus, so any 16 kHz single-speaker set works.
Two openly licensed sets of the same sentence material:

- ARU Speech Corpus, University of Liverpool: 12 speakers x 720 IEEE sentences,
  anechoic, CC-BY.  <https://datacat.liverpool.ac.uk/681/1/ARU_Speech_Corpus_v1_0.zip>
- Open Speech Repository: Harvard sentences, 16 kHz and 8 kHz, several speakers.
  <https://www.voiptroubleshooter.com/open_speech/american.html>

Resample any replacement to 16 kHz and put it under `data/arctic/<speaker>/`;
`--data` can point at a corpus kept elsewhere.  Changing the corpus changes the
alias-free band and the numbers.

## Cost

| step | scale | note |
|---|---|---|
| RIR cache, `a2` | 876 impulse responses | 6 T60 x 73 directions x 2 kinds |
| RIR cache, `b8`/`spk6`/`rob5` | 156 each | 2 distances x 3 T60 x 13 directions x 2 |
| RIR cache, `car4` | 78 | 1 distance x 3 T60 x 13 directions x 2 |
| babble bank, `a2` | 438 streams, about 340 MB resident | 1 distance x 6 T60 x 73 directions, 37 of which carry talkers |
| babble bank, others | 78 streams or fewer | proportional to grid size |
| a2 full run, 200 utt | 7 min 33 s on one core | 6 conditions, including the babble bank build |

The RIR cache is written once and read afterwards.  Build it per array rather
than all at once on a small machine.

## Layout and tests

```
src/oracle_doa/
  config.py      constants and data-directory resolution
  arrays.py      the five geometries and the direction grids
  rooms.py       Sabine absorption, image-source RIRs, disk cache, DRR
  audio.py       corpus pool, STFT, babble bank, mixture synthesis
  masks.py       oracle IRM and PSM
  backends.py    scoring core and the five estimators
  evaluate.py    conditions, evaluation loop, metrics, JSON
  download.py    CMU ARCTIC fetcher
  cli.py         command line
tests/           geometry, masks, backends on synthetic input, smoke test
examples/results/  one short run, one full a2 run
```

```bash
pytest -q
```

The smoke test skips itself when the corpus is absent or when
`pyroomacoustics` cannot compute an impulse response on the machine.

## Citing

The mask-guided localisation framework and the mask definitions follow

> Z.-Q. Wang, X. Zhang, and D. Wang, "Robust Speaker Localization Guided by
> Deep Learning-Based Time-Frequency Masking," *IEEE/ACM Trans. Audio, Speech,
> Lang. Process.*, vol. 27, no. 1, pp. 178-188, 2019.

The two-stage CASA line that motivates estimating these masks from a single
channel rather than from the array is

> Y. Liu and D. L. Wang, "Divide and Conquer: A Deep CASA Approach to
> Talker-Independent Monaural Speaker Separation," *IEEE/ACM Trans. Audio,
> Speech, Lang. Process.*, vol. 27, no. 12, pp. 2092-2102, 2019.

## License

MIT for the code in this repository.  The corpora keep their own licences; none
of them is redistributed here.
