# MATLAB EPA dataset generator

Open `generate_epa_ofdm_dataset.m` in MATLAB Online and run:

```matlab
generate_epa_ofdm_dataset(12, 'epa7_12pilots_matlab.mat')
generate_epa_ofdm_dataset(16, 'epa7_16pilots_matlab.mat')
generate_epa_ofdm_dataset(24, 'epa7_24pilots_matlab.mat')
generate_epa_ofdm_dataset(48, 'epa7_48pilots_matlab.mat')
```

The paper reports 24 and 48 pilots. The 12- and 16-pilot datasets are additional reduced-overhead experiments. The script places four pilot-time positions and uses 3, 4, 6, or 12 evenly spaced pilot-frequency positions, respectively.

The script requires **LTE Toolbox** and `lteFadingChannel`. It uses MATLAB's
sample-level EPA waveform filtering with fractional-delay compensation, GMEDS
fading, a 1.08 MHz sample rate, and the 115.82 Hz Doppler implied by the
paper's 2.5 GHz carrier and 50 km/h speed.

The `.mat` output contains `X_LS`, `H_TRUE`, `SNR_DB`, `split`, and `metadata`.
Array order is `[sample, real_imag, ...]`, intentionally matching the Python
project's layout. The script records the exact pilot indices, seeds, SNR
allocation, and explicit assumptions in `metadata`.

The paper does not publish its pilot indices, MATLAB seed, or SNR allocation.
The choices in the script are therefore reproducible defaults, not claims about
the authors' unpublished implementation.

## Import and verify in Python

After uploading the `.mat` file into the project workspace, convert it without
altering any samples:

```powershell
python import_matlab_dataset.py --mat-file epa7_12pilots_matlab.mat --output data/matlab_epa7_12pilots
python import_matlab_dataset.py --mat-file epa7_16pilots_matlab.mat --output data/matlab_epa7_16pilots
python import_matlab_dataset.py --mat-file epa7_24pilots_matlab.mat --output data/matlab_epa7_24pilots
python import_matlab_dataset.py --mat-file epa7_48pilots_matlab.mat --output data/matlab_epa7_48pilots
```

The importer converts MATLAB's v7.3 HDF5 dimension order to PyTorch's
`[sample, real_imag, frequency, time]` order and stores a SHA-256 checksum of
the source `.mat` file in each manifest. The imported files in this workspace
were verified to contain the expected 12,000/4,000/4,000 split and matching
target channels across the two pilot-count experiments.
