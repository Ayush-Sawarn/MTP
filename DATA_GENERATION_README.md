# Data generation: EPA OFDM channel-estimation dataset

This document describes how the fixed training, validation, and test data in
this repository are generated for the AttRNet reproduction. The implementation
is [simulate_data.py](simulate_data.py) and the channel code is
[attrnet_ofdm/data.py](attrnet_ofdm/data.py).

## Goal

Each sample provides a noisy least-squares (LS) channel estimate at pilot
locations and the corresponding complete frequency-domain channel response:

```text
input x : [2, pilot_frequency_positions, pilot_time_positions]
target y: [2, 72 subcarriers, 14 OFDM symbols]
```

Axis 0 stores real and imaginary parts, respectively. The learning task is to
reconstruct `y` from `x`.

## Parameters taken from the paper

| Parameter | Value |
| --- | --- |
| Propagation model | 3GPP Extended Pedestrian A (EPA), 7 paths |
| EPA delays | 0, 30, 70, 90, 110, 190, 410 ns |
| EPA relative powers | 0, -1, -2, -3, -8, -17.2, -20.8 dB |
| Carrier frequency | 2.5 GHz |
| Receiver speed | 50 km/h |
| Subcarrier spacing | 15 kHz |
| Active subcarriers | 72 |
| OFDM symbols per subframe | 14 |
| Modulation | QPSK |
| Cyclic prefix length | 16 samples |
| Pilot counts from the paper | 24 and 48 |
| Additional reduced-overhead counts supported | 12 and 16 |
| Dataset size | 20,000 samples |
| Split | 12,000 train / 4,000 validation / 4,000 test |

The derived sampling rate is `72 × 15 kHz = 1.08 MHz`. Including the cyclic
prefix, an OFDM symbol lasts `88 / 1.08 MHz = 81.48 µs`. At 2.5 GHz and
50 km/h, the maximum Doppler frequency is approximately `115.82 Hz`.

## Channel generation procedure

For each sample, the generator:

1. Creates one independent seven-tap EPA channel realization.
2. Generates each tap as a zero-mean complex Gaussian fading process with
   Clarke/Jakes time covariance:

   ```text
   R(Δt) = J0(2π fD Δt)
   ```

   where `fD` is the maximum Doppler frequency above. This jointly generates
   the 14 time samples, rather than approximating temporal fading with a
   first-order AR process.
3. Scales each fading tap by the normalized EPA path power.
4. Converts tap gains to the frequency response at the 72 subcarrier
   frequencies using `exp(-j 2π f τ)`.
5. Generates unit-energy QPSK pilot symbols.
6. Samples the true frequency response at pilot positions, adds complex AWGN,
   and forms the LS observation:

   ```text
   Yp = Hp Xp + Wp
   H_LS = Yp / Xp
   ```

7. Stores real/imaginary LS pilot estimates as `x` and the full real/imaginary
   channel response as `y`.

## Pilot arrangement and SNR

The paper reports 24 and 48 pilots but does not disclose pilot coordinates.
For those counts, and for the additional 12- and 16-pilot experiments, this
project uses evenly distributed rectangular grids:

| Pilot count | Input shape | Frequency positions | Time positions |
| --- | --- | --- | --- |
| 12 | `[2, 3, 4]` | 3 evenly spaced locations over 72 subcarriers | 4 evenly spaced locations over 14 symbols |
| 16 | `[2, 4, 4]` | 4 evenly spaced locations over 72 subcarriers | 4 evenly spaced locations over 14 symbols |
| 24 | `[2, 6, 4]` | 6 evenly spaced locations over 72 subcarriers | 4 evenly spaced locations over 14 symbols |
| 48 | `[2, 12, 4]` | 12 evenly spaced locations over 72 subcarriers | 4 evenly spaced locations over 14 symbols |

The 12- and 16-pilot configurations are additional project experiments, not pilot counts evaluated in the AttRNet paper. All four counts use rectangular grids with four pilot-time positions; arbitrary pilot arrangements are not supported by this generator.

The training-data SNR distribution is also absent from the paper. The default
dataset cycles evenly through `0, 5, 10, 15, 20 dB`, giving 2,400 examples of
each SNR in the 12,000-sample training set and 800 in each 4,000-sample split.
Both choices are recorded under `unpublished_assumptions` in the generated
manifest.

## Generate the full datasets

From the workspace root:

```powershell
# 24-pilot experiment
python simulate_data.py --output data/epa7_24pilots --pilots 24

# 48-pilot experiment
python simulate_data.py --output data/epa7_48pilots --pilots 48

# Additional reduced-overhead experiments
python simulate_data.py --output data/epa7_12pilots --pilots 12
python simulate_data.py --output data/epa7_16pilots --pilots 16
```

The commands use the same seed by default. Consequently, their target channel
realizations are identical; only the pilot observation grids differ. This makes
pilot-count comparisons fair.

To use a different deterministic realization, choose a different seed:

```powershell
python simulate_data.py --output data/epa7_24pilots_seed7 --pilots 24 --seed 7
```

## Generated files

For each dataset directory, the generator writes:

```text
manifest.json
train_x.npy              train_y.npy              train_snr_db.npy
validation_x.npy         validation_y.npy         validation_snr_db.npy
test_x.npy               test_y.npy               test_snr_db.npy
```

The `.npy` arrays are memory-mapped by training and evaluation. They are not
loaded in full into RAM. `manifest.json` is the source of truth for the
configuration, derived Doppler/sampling values, pilot indices, SNR values,
seed, shapes, EPA profile, and every non-published assumption.

## Verify a dataset

```powershell
python -c "import json, numpy as np; p='data/epa7_24pilots'; m=json.load(open(p+'/manifest.json')); print(m['published_parameters']); print(np.load(p+'/train_x.npy', mmap_mode='r').shape); print(np.load(p+'/train_y.npy', mmap_mode='r').shape)"
```

Expected 24-pilot training shapes are:

```text
(12000, 2, 6, 4)
(12000, 2, 72, 14)
```

For 12, 16, and 48 pilots, the first shapes are `(12000, 2, 3, 4)`, `(12000, 2, 4, 4)`, and `(12000, 2, 12, 4)`, respectively.

## Reproducibility boundary

This is a standards-based reconstruction of the published setup, not the
authors' original MATLAB dataset. Exact sample-level replication is impossible
without their unpublished pilot coordinates, random seeds, SNR allocation,
channel-toolbox options, and preprocessing. The manifest makes every such
choice visible so results can be reproduced and compared honestly.
