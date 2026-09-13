# AttRNet reproduction for OFDM channel estimation

This is a runnable PyTorch reproduction of *Deep Residual Learning With
Attention Mechanism for OFDM Channel Estimation* (Gao et al.).  It implements
the two proposed estimators, `AttRNet-Conv` and `AttRNet-FC`, together with an
EPA multipath OFDM simulator, LS estimates at pilot locations, and an MSE
evaluation loop.

For the complete dataset procedure and all reproducibility assumptions, see
[DATA_GENERATION_README.md](DATA_GENERATION_README.md).

## Paper settings represented here

| Item | Value |
| --- | --- |
| Channel | 3GPP EPA, 7 paths |
| Carrier / speed | 2.5 GHz / 50 km/h |
| OFDM grid | 72 subcarriers x 14 symbols |
| Modulation / CP | QPSK / 16 samples |
| Pilots | 24 (6 x 4) or 48 (12 x 4) |
| Network | 8 feature maps, three attention residual blocks |
| Dataset | 20,000 samples: 12,000 train / 4,000 validation / 4,000 test |
| Training | Adam, lr=0.001, batch=128, 100 epochs |

The PDF describes its last layer as a `36 x 7 x 2` convolution after
upsampling.  Since the padding/stride are not specified (and a valid 36x7
convolution cannot retain a 72x14 output), this implementation uses a
same-padded `(36, 7)` output convolution by default.  Pass `--output-kernel
3 3` for a compact conventional reconstruction head.  The figures/text also
swap the Conv/FC attention labels; model names here follow the abstract:
`AttRNet-Conv` uses 1x1 convolutions and `AttRNet-FC` uses linear layers.

## Install and run

```powershell
python -m pip install -r requirements.txt

# First generate the paper-scale data and its reproducibility manifest.
# It writes about 330 MB for 24 pilots.
python simulate_data.py --output data/epa7_24pilots --pilots 24

# Import MATLAB Online's waveform-level EPA data for training in this project.
python import_matlab_dataset.py --mat-file epa7_24pilots_matlab.mat --output data/matlab_epa7_24pilots
python import_matlab_dataset.py --mat-file epa7_48pilots_matlab.mat --output data/matlab_epa7_48pilots

# A fast smoke training run (works on CPU; CUDA is selected automatically)
python train.py --model fc --epochs 2 --data-dir data/epa7_24pilots --output runs/smoke

# Paper-scale training with the imported MATLAB lteFadingChannel data
python train.py --model fc --pilots 24 --epochs 100 --data-dir data/matlab_epa7_24pilots --output runs/matlab_attnet_fc_24
python train.py --model conv --pilots 48 --epochs 100 --data-dir data/matlab_epa7_48pilots --output runs/matlab_attnet_conv_48

# Evaluate the best checkpoint on the fixed 4,000-sample test split.
python evaluate.py --checkpoint runs/matlab_attnet_fc_24/best.pt --data-dir data/matlab_epa7_24pilots
python evaluate.py --checkpoint runs/matlab_attnet_conv_48/best.pt --data-dir data/matlab_epa7_48pilots

# Update the supplied comparison table with a real test result.
python evaluate.py --checkpoint runs/matlab_attnet_fc_24/best.pt --data-dir data/matlab_epa7_24pilots --comparison-csv paper_vs_simulation_metrics.csv
```

Each evaluation automatically creates a paper-style log-MSE versus SNR PNG
beside the checkpoint. Use `--plot results/fc_24_comparison.png` to choose a
different location. The paper line is digitized from Figures 5–6 and the solid
line is calculated from the saved 4,000-sample test split.

The legacy `data/epa7_*` directories and `runs/attnet_*` checkpoints use the
earlier Python-only simulator and a superseded 10,254-parameter model. Use the
`data/matlab_epa7_*` directories above for the closest MATLAB/EPA reproduction.

`simulate_data.py` records every parameter, derived Doppler quantity, pilot
coordinate, seed, and unavoidable assumption in `manifest.json`. Its EPA tap
processes have Clarke/Jakes temporal covariance at the paper's stated maximum
Doppler. Reported results will not be numerically identical to the MATLAB paper
without its unpublished pilot map, fading realization, and preprocessing
choices, but the published system/data parameters are all represented.
