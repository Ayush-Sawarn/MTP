# Deep Learning Based Channel Estimation for OFDM

This project reproduces the AttRNet-Conv and AttRNet-FC channel estimators from Gao et al., *Deep Residual Learning With Attention Mechanism for OFDM Channel Estimation*. It builds MATLAB EPA-OFDM datasets, trains PyTorch models, and compares test-set MSE against approximate values digitized from the paper's plots.

## Progress so far

### AttRNet baseline

The baseline has been trained and evaluated for both attention variants at both pilot counts. The runs use the MATLAB-generated EPA datasets, 100 epochs, Adam with learning rate 0.001, batch size 128, and eight feature channels.

| Pilots | Model | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 24 | AttRNet-FC | 0.0655015 | 0.0255193 | 0.0114817 | 0.00633771 | 0.00449757 |
| 24 | AttRNet-Conv | 0.0655018 | 0.0255239 | 0.0114855 | 0.00634923 | 0.00451102 |
| 48 | AttRNet-FC | 0.0389337 | 0.0161118 | 0.00757006 | 0.00442765 | 0.00313416 |
| 48 | AttRNet-Conv | 0.0389175 | 0.0161130 | 0.00756809 | 0.00442008 | 0.00312695 |

These are test MSE values from the saved runs in `runs/corrected_*`. The expected trend is present: MSE decreases as SNR increases, and the 48-pilot models perform better than the corresponding 24-pilot models. FC and Conv attention perform similarly for a fixed pilot count.

The values are below the paper's approximate digitized values at 0–15 dB and above them at 20 dB for all four configurations. The paper values are readings from Figures 5–6, not tabulated source data. This remaining mismatch is a replication discrepancy, not evidence of improved performance. The original authors' dataset and exact pilot locations are unavailable, and simulation, noise, target, and preprocessing details may differ. See [`paper_vs_simulation_metrics.csv`](paper_vs_simulation_metrics.csv) for the point-by-point reference, relative differences, and checkpoint paths.

#### MSE-versus-SNR plots

Each plot compares the paper's digitized MSE values (dashed) with this project's test-set MSE (solid) at each SNR.

**24 pilots — AttRNet-FC**

![24 pilots, AttRNet-FC: paper and simulation MSE versus SNR](runs/corrected_fc_24/comparison.png)

**24 pilots — AttRNet-Conv**

![24 pilots, AttRNet-Conv: paper and simulation MSE versus SNR](runs/corrected_conv_24/comparison.png)

**48 pilots — AttRNet-FC**

![48 pilots, AttRNet-FC: paper and simulation MSE versus SNR](runs/corrected_fc_48/comparison.png)

**48 pilots — AttRNet-Conv**

![48 pilots, AttRNet-Conv: paper and simulation MSE versus SNR](runs/corrected_conv_48/comparison.png)

### Pilot-overhead study

The October 2026 mid-semester presentation reports preliminary comparisons of comb, uniform, block, staggered, and lattice pilot arrangements. Its reported observations are that uniform placement gives the lowest MSE among the tested 24-pilot patterns over most SNRs, and 16 uniformly placed pilots outperform the tested 24-pilot block and comb patterns in the shown results. These are preliminary presentation results; their per-SNR data and generation scripts are not currently included in this repository.

The generator and training scripts support uniform rectangular grids with 12, 16, 24, or 48 pilots, each using four pilot-time positions. The ConcreteAE selector and AttRNet comparison have now been run for 12 and 16 pilots using FC attention.

The selector needs noisy LS candidates at all 72 Ã— 14 resource elements. The MATLAB files store noisy LS values only at the original uniform pilot positions, along with the clean full channel. `pilot_design/pilot_utils.py` therefore reconstructs candidate observations on demand from each saved clean channel using the generator's unit-energy QPSK and AWGN equation and that sample's SNR. This leaves the imported datasets unchanged. It is an explicit observation-generation assumption: the added pilot patterns are not measurements recorded in the original MATLAB file.

Pilot indices use C-order flattening, `index = subcarrier * 14 + OFDM_symbol`. The selector fits only the training split, chooses checkpoints by discrete-selection validation MSE, and does not load test data. The AttRNet scripts keep the existing estimator architecture and train separate models per pattern and budget. Both methods use identical synthesized noisy observations for a given sample/seed and zero-filled `[2,72,14]` input grids.

Current FC results are mixed. At 12 pilots, ConcreteAE has lower NMSE at 10, 15, and 20 dB, but higher NMSE at 0 and 5 dB; its mean NMSE over the five SNRs is slightly higher (0.1642 vs 0.1636). At 16 pilots, ConcreteAE is better at 0 dB but worse at 5–20 dB, with a pronounced gap at higher SNRs. These are single-run results. In the 12-pilot FC runs, uniform AttRNet trained for 100 epochs while ConcreteAE AttRNet trained for 50, so that pair should be rerun with matched settings. The 16-pilot ConcreteAE selector's best MLP-decoder validation epoch was 1; that suggests the selected pattern may be unstable or mismatched to AttRNet's objective, but does not establish the cause. Test NMSE should not be used to choose selectors or tune training settings.

For future FC experiments, use each budget's matching MATLAB import. Train selectors for each budget; train uniform AttRNet as well when a matched uniform model does not already exist under this full-grid sparse-input protocol. The 12- and 16-pilot imports have identical clean-channel targets and SNR assignments, as do the 24- and 48-pilot imports.

To rerun the complete FC comparison from 12 pilots, use the PowerShell automation script from this project directory. It sequentially trains all four selectors, then uniform and ConcreteAE FC AttRNet models for each budget, evaluates all eight models on the held-out test splits, and regenerates the FC plots. It stops if a step fails. Activate the desired Python environment first so `python` resolves to that environment.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_fc_pilot_experiments.ps1
```

The default is 100 AttRNet epochs and seed 2025. Optional parameters can change those values, for example `-Epochs 50 -Seed 2026`. The script overwrites the matching selector indices/checkpoints, FC run checkpoints/configs, and FC metric rows for these four budgets. It does not delete historical Conv metrics.

```powershell
python train_pilot_selector.py --pilots 48 --data-dir data/matlab_epa7_24pilots
python train_pilot_selector.py --pilots 24 --data-dir data/matlab_epa7_24pilots
python train_pilot_selector.py --pilots 16 --data-dir data/matlab_epa7_16pilots
python train_pilot_selector.py --pilots 12 --data-dir data/matlab_epa7_12pilots
```

For 12 and 16 pilots, train matching uniform and ConcreteAE FC estimators. Example for 16 pilots:

```powershell
python train_pilot_attnet.py --model fc --method uniform --pilots 16 --data-dir data/matlab_epa7_24pilots
python train_pilot_attnet.py --model fc --method concrete --pilots 16 --pilot-indices results/pilot_indices/concrete_16.npy --data-dir data/matlab_epa7_24pilots
```

For 24 and 48 pilots, ConcreteAE FC training commands are:

```powershell
python train_pilot_attnet.py --model fc --method concrete --pilots 24 --pilot-indices results/pilot_indices/concrete_24.npy --data-dir data/matlab_epa7_24pilots --epochs 100
python train_pilot_attnet.py --model fc --method concrete --pilots 48 --pilot-indices results/pilot_indices/concrete_48.npy --data-dir data/matlab_epa7_48pilots --epochs 100
```

The existing `runs/corrected_fc_24` and `runs/corrected_fc_48` uniform checkpoints take compact pilot-grid inputs (`[2, P_f, 4]`). The new pilot-pattern training pipeline takes zero-filled full-grid inputs (`[2,72,14]`). Although the estimator architecture and weights have compatible tensor operations, these input/training protocols differ, so their metrics are not a strict controlled comparison. For a fair uniform-versus-ConcreteAE result at 24 and 48 pilots, retrain the uniform FC estimator using `train_pilot_attnet.py` and the matching data directory, or clearly label the existing baseline comparison as protocol-mismatched.

Evaluate each resulting `best.pt` checkpoint on test data:

```powershell
python evaluate_pilot_patterns.py --checkpoint runs/pilot_design/fc_uniform_16/best.pt --data-dir data/matlab_epa7_24pilots
python evaluate_pilot_patterns.py --checkpoint runs/pilot_design/fc_concrete_16/best.pt --data-dir data/matlab_epa7_24pilots
```

Evaluation appends/replaces the matching run rows in `results/metrics.csv` and saves example channel arrays. After evaluating the FC runs, create aggregate NMSE-versus-SNR, NMSE-versus-pilot-count, overhead-versus-NMSE, and selected-location plots with:

```powershell
python plot_pilot_results.py --model AttRNet-FC
```

The selector's synthetic observations use the same AWGN variance convention as `matlab/generate_epa_ofdm_dataset.m`. Since the stored dataset does not retain received symbols/noise at omitted positions, learned pilot selection is evaluated under this controlled regenerated observation model. Record that limitation when interpreting results.

### Next work

- Integrate and validate the different pilot counts and placements in the codebase, then report MSE/NMSE versus SNR and pilot overhead.
- Design and benchmark a lightweight residual Transformer with time-frequency position information against AttRNet using matched conditions.
- Explore the proposed IRS-assisted OFDM extension, including cascaded-channel estimation and IRS phase configuration.
- Compare estimation accuracy alongside parameter count, FLOPs, and inference time.

These are planned directions, not features implemented by the current scripts.

## Model and data

The model takes real and imaginary LS pilot estimates as two input channels and predicts the full 72 × 14 complex channel grid. Each attention block uses global average pooling to summarize learned convolution feature maps and produces one attention weight per feature channel; these weights do not correspond directly to individual pilot positions.

The implementation uses a stem convolution, three attention residual blocks, a 3 × 3 feature convolution, bilinear upsampling, and a 36 × 7 output convolution. The paper does not specify padding for the final layer, so this implementation uses same-size asymmetric padding to preserve the 72 × 14 output. AttRNet-FC uses fully connected attention excitation layers; AttRNet-Conv uses 1 × 1 convolutional excitation layers.

The current eight-feature implementation has 10,254 trainable parameters, including the post-block 3 × 3 feature convolution. The mid-semester presentation lists 9,670; reconcile the parameter-count convention and whether that feature convolution was included before using that figure as a paper-matched count.

The MATLAB dataset generator uses `lteFadingChannel` with the 3GPP EPA 7-path profile, 2.5 GHz carrier, 50 km/h motion, and derived maximum Doppler of about 115.82 Hz. Each pilot-count dataset has 20,000 samples split into 12,000 training, 4,000 validation, and 4,000 test samples. SNRs 0, 5, 10, 15, and 20 dB are equally allocated. Pilot coordinates and equal SNR allocation are explicit reproduction choices because the paper does not disclose them.

For details of the generated arrays and assumptions, see [`DATA_GENERATION_README.md`](DATA_GENERATION_README.md) and [`matlab/README.md`](matlab/README.md).

## Run the baseline

From this directory, activate a Python environment with the packages in `requirements.txt`. MATLAB datasets and imported NumPy splits for the 24- and 48-pilot baseline are already present in this workspace. To generate 12- or 16-pilot MATLAB datasets, run `generate_epa_ofdm_dataset(12, 'epa7_12pilots_matlab.mat')` or `generate_epa_ofdm_dataset(16, 'epa7_16pilots_matlab.mat')` in MATLAB Online, then import them with:

```powershell
python import_matlab_dataset.py --mat-file epa7_12pilots_matlab.mat --output data/matlab_epa7_12pilots
python import_matlab_dataset.py --mat-file epa7_16pilots_matlab.mat --output data/matlab_epa7_16pilots
```

Alternatively, generate them with the Python simulator:

```powershell
python simulate_data.py --output data/epa7_12pilots --pilots 12
python simulate_data.py --output data/epa7_16pilots --pilots 16
```

To train on either dataset, choose its pilot count and data directory, for example `python train.py --model fc --pilots 12 --data-dir data/matlab_epa7_12pilots --epochs 100 --output runs/fc_12pilots`. The existing baseline commands are:

Train each baseline configuration:

```powershell
python train.py --model fc --pilots 24 --data-dir data/matlab_epa7_24pilots --epochs 100 --output runs/corrected_fc_24
python train.py --model conv --pilots 24 --data-dir data/matlab_epa7_24pilots --epochs 100 --output runs/corrected_conv_24
python train.py --model fc --pilots 48 --data-dir data/matlab_epa7_48pilots --epochs 100 --output runs/corrected_fc_48
python train.py --model conv --pilots 48 --data-dir data/matlab_epa7_48pilots --epochs 100 --output runs/corrected_conv_48
```

Evaluate a checkpoint on the held-out test set and update the CSV. Repeat with each checkpoint and its matching data directory:

```powershell
python evaluate.py --checkpoint runs/corrected_fc_24/best.pt --data-dir data/matlab_epa7_24pilots --split test --comparison-csv paper_vs_simulation_metrics.csv
```

Evaluation prints overall and per-SNR MSE and saves a paper-style log-MSE comparison plot beside the checkpoint. Use `--plot path/to/plot.png` to choose a plot path.

## Repository map

- `attrnet_ofdm/model.py`: AttRNet-FC and AttRNet-Conv architectures.
- `attrnet_ofdm/data.py`: EPA channel/pilot utilities and saved-dataset loader.
- `matlab/generate_epa_ofdm_dataset.m`: MATLAB waveform-level EPA data generator.
- `import_matlab_dataset.py`: converts MATLAB v7.3 files into NumPy splits.
- `train.py` and `evaluate.py`: training, per-SNR test metrics, and paper comparisons.
- `paper_vs_simulation_metrics.csv`: approximate paper references and current baseline test results.
- `runs/corrected_*`: baseline checkpoints, configurations, and comparison plots.
