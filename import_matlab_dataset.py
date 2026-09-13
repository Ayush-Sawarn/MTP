"""Convert the MATLAB v7.3 EPA dataset into memory-mapped NumPy splits.

MATLAB stores v7.3 arrays as HDF5 with reversed dimension order.  This tool
preserves every sample while writing the layout consumed by train.py:
``[sample, real_imag, frequency, time]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def read_split_indices(h5: h5py.File, name: str) -> np.ndarray:
    indices = np.asarray(h5["split"][name]).reshape(-1).astype(np.int64) - 1
    if np.any(indices < 0):
        raise ValueError(f"MATLAB split '{name}' contains invalid (non-positive) indices")
    return indices


def take_samples(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read final-axis HDF5 samples, using a slice when the split is contiguous."""
    if len(indices) and np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices))):
        return dataset[..., int(indices[0]) : int(indices[-1]) + 1]
    return dataset[..., indices]


def convert_split(h5: h5py.File, name: str, output: Path, batch_size: int) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    indices = read_split_indices(h5, name)
    x_source, y_source, snr_source = h5["X_LS"], h5["H_TRUE"], h5["SNR_DB"]
    # HDF5 order from MATLAB: X=[time,pilot_freq,complex,sample],
    # H=[symbol,subcarrier,complex,sample].
    if x_source.ndim != 4 or y_source.shape[:3] != (14, 72, 2):
        raise ValueError("Unexpected MATLAB array layout; expected X_LS and H_TRUE from generate_epa_ofdm_dataset.m")
    n = len(indices)
    x_shape = (n, 2, x_source.shape[1], x_source.shape[0])
    y_shape = (n, 2, y_source.shape[1], y_source.shape[0])
    x_out = np.lib.format.open_memmap(output / f"{name}_x.npy", mode="w+", dtype=np.float32, shape=x_shape)
    y_out = np.lib.format.open_memmap(output / f"{name}_y.npy", mode="w+", dtype=np.float32, shape=y_shape)
    snr_out = np.lib.format.open_memmap(output / f"{name}_snr_db.npy", mode="w+", dtype=np.float32, shape=(n,))
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        source_indices = indices[start:end]
        x_out[start:end] = np.transpose(take_samples(x_source, source_indices), (3, 2, 1, 0))
        y_out[start:end] = np.transpose(take_samples(y_source, source_indices), (3, 2, 1, 0))
        # SNR_DB is [1,sample] in the MATLAB HDF5 layout.
        snr_out[start:end] = take_samples(snr_source, source_indices).reshape(-1)
    x_out.flush()
    y_out.flush()
    snr_out.flush()
    return n, x_shape, y_shape


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if not args.mat_file.is_file():
        parser.error(f"MATLAB file not found: {args.mat_file}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty: {args.output}. Choose a new directory to avoid overwriting data.")
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.mat_file, "r") as h5:
        required = {"X_LS", "H_TRUE", "SNR_DB", "split", "metadata"}
        missing = required - set(h5.keys())
        if missing:
            raise ValueError(f"Not a supported MATLAB EPA dataset; missing {sorted(missing)}")
        pilot_rows = np.asarray(h5["metadata/pilotRowsMatlabOneBased"]).reshape(-1).astype(int)
        pilot_columns = np.asarray(h5["metadata/pilotColumnsMatlabOneBased"]).reshape(-1).astype(int)
        results = {}
        for split in ("train", "validation", "test"):
            print(f"Converting {split} split...")
            results[split] = convert_split(h5, split, args.output, args.batch_size)
        sample_rate = float(np.asarray(h5["metadata/derivedParameters/sampleRateHz"]).squeeze())
        doppler = float(np.asarray(h5["metadata/derivedParameters/maximumDopplerHz"]).squeeze())
        seed = int(np.asarray(h5["metadata/baseSeed"]).squeeze())
        snr_values = np.asarray(h5["metadata/snrValuesDb"]).reshape(-1).astype(float).tolist()
    manifest = {
        "source": "MATLAB lteFadingChannel EPA v7.3 import",
        "mat_file": args.mat_file.name,
        "mat_file_sha256": sha256(args.mat_file),
        "matlab_hdf5_layout": {"X_LS": "[pilot_time,pilot_frequency,real_imag,sample]", "H_TRUE": "[symbol,subcarrier,real_imag,sample]"},
        "dataset_format": {"x": "[sample,real_imag,pilot_frequency,pilot_time]", "y": "[sample,real_imag,subcarrier,symbol]", "snr_db": "[sample]"},
        "pilot_configuration": {"count": int(len(pilot_rows) * len(pilot_columns)), "frequency_indices_matlab_one_based": pilot_rows.tolist(), "time_indices_matlab_one_based": pilot_columns.tolist()},
        "split": {name: result[0] for name, result in results.items()},
        "shapes": {name: {"x": result[1], "y": result[2]} for name, result in results.items()},
        "snr_db_values": snr_values,
        "derived_parameters": {"sample_rate_hz": sample_rate, "maximum_doppler_hz": doppler},
        "seed": seed,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done: {args.output.resolve()}")


if __name__ == "__main__":
    main()
