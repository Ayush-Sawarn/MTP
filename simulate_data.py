"""Generate the 20,000-sample EPA dataset described in Gao et al.'s paper.

The resulting directory contains independent NumPy arrays for the paper's
12,000/4,000/4,000 training, validation, and test splits.  Inputs are noisy LS
estimates at pilot positions; targets are complete complex channel gains, with
real and imaginary parts stored along axis one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from attrnet_ofdm.data import EPAConfig, OFDMDataset, make_pilot_grid


PUBLISHED = {
    "channel_model": "3GPP Extended Pedestrian A (EPA), 7 paths",
    "carrier_frequency_hz": 2.5e9,
    "subcarrier_spacing_hz": 15_000.0,
    "receiver_speed_kmh": 50.0,
    "number_of_subcarriers": 72,
    "number_of_ofdm_symbols": 14,
    "modulation": "QPSK",
    "cyclic_prefix_length_samples": 16,
    "total_samples": 20_000,
    "split": {"train": 12_000, "validation": 4_000, "test": 4_000},
}


def write_split(output: Path, name: str, start: int, size: int, pilots: int, snrs: tuple[float, ...], seed: int, config: EPAConfig) -> None:
    probe = OFDMDataset(1, pilots, snr_db=snrs[0], seed=seed, config=config)
    x0, y0 = probe[0]
    x_file = np.lib.format.open_memmap(output / f"{name}_x.npy", mode="w+", dtype=np.float32, shape=(size, *x0.shape))
    y_file = np.lib.format.open_memmap(output / f"{name}_y.npy", mode="w+", dtype=np.float32, shape=(size, *y0.shape))
    snr_file = np.lib.format.open_memmap(output / f"{name}_snr_db.npy", mode="w+", dtype=np.float32, shape=(size,))
    for local_index in range(size):
        global_index = start + local_index
        snr = snrs[global_index % len(snrs)]
        dataset = OFDMDataset(1, pilots, snr_db=snr, seed=seed + global_index, config=config)
        x, y = dataset[0]
        x_file[local_index], y_file[local_index], snr_file[local_index] = x.numpy(), y.numpy(), snr
    x_file.flush()
    y_file.flush()
    snr_file.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/epa7_24pilots"))
    parser.add_argument("--pilots", type=int, choices=[24, 48], default=24)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--train-size", type=int, default=12_000)
    parser.add_argument("--validation-size", type=int, default=4_000)
    parser.add_argument("--test-size", type=int, default=4_000)
    args = parser.parse_args()
    sizes = (args.train_size, args.validation_size, args.test_size)
    if min(sizes) <= 0:
        parser.error("Every split must contain at least one sample.")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = EPAConfig()
    offset = 0
    for name, size in zip(("train", "validation", "test"), sizes):
        print(f"Writing {name}: {size:,} samples")
        write_split(output, name, offset, size, args.pilots, tuple(args.snr_db), args.seed, config)
        offset += size
    rows, cols = make_pilot_grid(config.n_subcarriers, config.n_symbols, args.pilots)
    manifest = {
        "published_parameters": {**PUBLISHED, "total_samples": sum(sizes), "split": dict(zip(("train", "validation", "test"), sizes))},
        "derived_parameters": {
            "sample_rate_hz": config.sample_rate_hz,
            "ofdm_symbol_period_s": config.ofdm_symbol_period_s,
            "maximum_doppler_hz": config.maximum_doppler_hz,
            "epa_delays_ns": [0, 30, 70, 90, 110, 190, 410],
            "epa_powers_db": [0, -1, -2, -3, -8, -17.2, -20.8],
        },
        "dataset_format": {"x": "[sample, real_imag, pilot_frequency, pilot_time]", "y": "[sample, real_imag, subcarrier, symbol]", "snr_db": "[sample]"},
        "pilot_configuration": {"count": args.pilots, "frequency_indices": rows.tolist(), "time_indices": cols.tolist()},
        "snr_db_values": args.snr_db,
        "seed": args.seed,
        "unpublished_assumptions": [
            "The paper specifies only the total pilot count, not pilot coordinates; a uniform 6x4 grid for 24 pilots or 12x4 grid for 48 pilots is used.",
            "The paper does not state the SNR distribution of its generated training data; samples cycle equally through the specified SNR values.",
            "The paper does not publish random seeds or its MATLAB fading-object options; each EPA tap uses Clarke/Jakes temporal covariance at the stated maximum Doppler.",
            "A 72-point FFT with all 72 stated subcarriers active is used because no guard-band or FFT size is reported.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done: {output.resolve()}")


if __name__ == "__main__":
    main()
