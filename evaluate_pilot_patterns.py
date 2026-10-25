"""Evaluate one learned/uniform AttRNet pilot-pattern checkpoint on test data."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from attrnet_ofdm.model import AttRNet, parameter_count
from pilot_design.pilot_utils import PILOT_BUDGETS, PilotPatternDataset, validate_pilot_indices


FIELDS = [
    "pilot_method", "model", "pilot_count", "pilot_overhead_percent", "snr_db",
    "mse", "nmse", "nmse_db", "nmse_averaging", "training_time_seconds",
    "inference_latency_ms_per_sample", "num_parameters", "test_samples", "checkpoint",
]


def append_or_replace(path: Path, new_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old_rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as stream:
            old_rows = list(csv.DictReader(stream))
    keys = {(r["pilot_method"], r["model"], int(r["pilot_count"]), float(r["snr_db"])) for r in new_rows}
    kept = [
        row for row in old_rows
        if (row["pilot_method"], row["model"], int(row["pilot_count"]), float(row["snr_db"])) not in keys
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows(new_rows)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/matlab_epa7_24pilots"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--results-csv", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--examples-dir", type=Path, default=Path("results/examples"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    budget = int(checkpoint["pilots"])
    if budget not in PILOT_BUDGETS:
        parser.error(f"Unsupported pilot budget in checkpoint: {budget}")
    indices = validate_pilot_indices(checkpoint["pilot_indices"], budget)
    dataset = PilotPatternDataset(args.data_dir, args.split, indices, budget, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = AttRNet(
        checkpoint["model"], checkpoint["features"],
        output_kernel=tuple(checkpoint["output_kernel"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    snr_values = np.asarray(dataset.snr_db)
    accum: dict[float, dict[str, float]] = {}
    inference_seconds = 0.0
    n_inference = 0
    examples: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for x, y, snr_batch in loader:
        x_device, y_device = x.to(device), y.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        prediction = model(x_device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - start
        n_inference += x.shape[0]

        squared_error = (prediction - y_device).square().sum(dim=1)
        squared_target = y_device.square().sum(dim=1)
        sample_mse = squared_error.mean(dim=(1, 2))
        sample_nmse = squared_error.sum(dim=(1, 2)) / (squared_target.sum(dim=(1, 2)) + 1e-12)
        for i, snr_tensor in enumerate(snr_batch):
            snr = float(snr_tensor.item())
            bucket = accum.setdefault(snr, {"mse": 0.0, "nmse": 0.0, "count": 0.0})
            bucket["mse"] += float(sample_mse[i].item())
            bucket["nmse"] += float(sample_nmse[i].item())
            bucket["count"] += 1
            if snr not in examples:
                examples[snr] = (
                    prediction[i].detach().cpu().numpy(),
                    y_device[i].detach().cpu().numpy(),
                )

    config_path = args.checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    train_seconds = float(config.get("training_seconds", checkpoint.get("training_seconds", 0.0)))
    parameter_total = parameter_count(model)
    latency_ms = 1000.0 * inference_seconds / max(n_inference, 1)
    method = checkpoint["pilot_method"]
    model_label = f"AttRNet-{checkpoint['model'].upper()}"
    overhead_pct = 100.0 * budget / (72 * 14)
    rows: list[dict[str, object]] = []

    print(f"checkpoint: {args.checkpoint}")
    print(f"data: {args.data_dir} ({args.split}, {len(dataset):,} samples); device={device}")
    print(f"pilot_method={method}; pilot_count={budget}; overhead={overhead_pct:.4f}%")
    print("snr_db,mse,nmse,nmse_db,samples")
    for snr in sorted(accum):
        count = int(accum[snr]["count"])
        mse_value = accum[snr]["mse"] / count
        nmse_value = accum[snr]["nmse"] / count
        nmse_db = 10.0 * np.log10(nmse_value + 1e-12)
        print(f"{snr:g},{mse_value:.8g},{nmse_value:.8g},{nmse_db:.6g},{count}")
        rows.append({
            "pilot_method": method,
            "model": model_label,
            "pilot_count": budget,
            "pilot_overhead_percent": f"{overhead_pct:.6f}",
            "snr_db": snr,
            "mse": f"{mse_value:.10g}",
            "nmse": f"{nmse_value:.10g}",
            "nmse_db": f"{nmse_db:.8g}",
            "nmse_averaging": "per_sample_then_mean",
            "training_time_seconds": f"{train_seconds:.6f}",
            "inference_latency_ms_per_sample": f"{latency_ms:.8g}",
            "num_parameters": parameter_total,
            "test_samples": count,
            "checkpoint": str(args.checkpoint),
        })
    append_or_replace(args.results_csv, rows)
    print(f"metrics CSV: {args.results_csv}")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    snrs = [float(row["snr_db"]) for row in rows]
    nmse_values = [float(row["nmse"]) for row in rows]
    figure, axis = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    axis.semilogy(snrs, nmse_values, marker="o", color="black", linewidth=2)
    axis.set_xticks(snrs)
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("NMSE (per-sample average)")
    axis.set_title(f"{model_label}, {method} pilots, budget {budget}")
    axis.grid(True, which="both", linestyle=":", alpha=0.65)
    figure.savefig(args.figures_dir / f"{checkpoint['model']}_{method}_{budget}_nmse_snr.png", dpi=180)
    plt.close(figure)

    args.examples_dir.mkdir(parents=True, exist_ok=True)
    example_snr = sorted(examples)[0]
    estimate, target = examples[example_snr]
    np.savez_compressed(
        args.examples_dir / f"{checkpoint['model']}_{method}_{budget}_snr_{example_snr:g}.npz",
        estimate=estimate,
        target=target,
        pilot_indices=indices,
        snr_db=example_snr,
    )


if __name__ == "__main__":
    main()
