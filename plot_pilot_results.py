"""Create aggregate plots after evaluating the pilot-pattern checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pilot_design.pilot_utils import PILOT_BUDGETS, pilot_coordinates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--indices-dir", type=Path, default=Path("results/pilot_indices"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--model", choices=["AttRNet-FC", "AttRNet-CONV", "all"], default="AttRNet-FC")
    args = parser.parse_args()
    with args.results_csv.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if args.model != "all":
        rows = [row for row in rows if row["model"] == args.model]
    if not rows:
        parser.error(f"No evaluated runs for model={args.model} found in {args.results_csv}")
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    series = sorted({(row["pilot_method"], row["model"]) for row in rows})
    budgets = sorted({int(row["pilot_count"]) for row in rows})
    snrs = sorted({float(row["snr_db"]) for row in rows})

    for budget in budgets:
        fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
        for method, model in series:
            selected = sorted(
                (row for row in rows if int(row["pilot_count"]) == budget and row["pilot_method"] == method and row["model"] == model),
                key=lambda row: float(row["snr_db"]),
            )
            if selected:
                ax.semilogy([float(r["snr_db"]) for r in selected], [float(r["nmse"]) for r in selected], marker="o", label=f"{method}, {model}")
        ax.set_xticks(snrs)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("NMSE (per-sample average)")
        ax.set_title(f"NMSE vs SNR, {budget} pilots")
        ax.grid(True, which="both", linestyle=":", alpha=0.65)
        ax.legend()
        fig.savefig(args.figures_dir / f"aggregate_nmse_snr_{budget}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    for method, model in series:
        points: list[tuple[int, float]] = []
        for budget in budgets:
            values = [float(r["nmse"]) for r in rows if int(r["pilot_count"]) == budget and r["pilot_method"] == method and r["model"] == model]
            if values:
                points.append((budget, float(np.mean(values))))
        if points:
            points.sort()
            ax.semilogy([p[0] for p in points], [p[1] for p in points], marker="o", label=f"{method}, {model}")
    ax.set_xticks(sorted(budgets))
    ax.set_xlabel("Number of pilots")
    ax.set_ylabel("Mean NMSE across evaluated SNRs")
    ax.set_title("NMSE vs pilot count")
    ax.grid(True, which="both", linestyle=":", alpha=0.65)
    ax.legend()
    fig.savefig(args.figures_dir / "aggregate_nmse_pilot_count.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    for method, model in series:
        selected = [r for r in rows if r["pilot_method"] == method and r["model"] == model]
        if selected:
            ax.scatter([float(r["pilot_overhead_percent"]) for r in selected], [float(r["nmse"]) for r in selected], label=f"{method}, {model}", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Pilot overhead (%)")
    ax.set_ylabel("NMSE")
    ax.set_title("Pilot overhead vs NMSE")
    ax.grid(True, which="both", linestyle=":", alpha=0.65)
    ax.legend()
    fig.savefig(args.figures_dir / "aggregate_overhead_nmse.png", dpi=180)
    plt.close(fig)

    for budget in PILOT_BUDGETS:
        path = args.indices_dir / f"concrete_{budget}.npy"
        if not path.exists():
            continue
        coordinates = pilot_coordinates(np.load(path, allow_pickle=False))
        fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
        ax.scatter(coordinates[:, 1], coordinates[:, 0], marker="x", s=36)
        ax.set_xlim(-0.5, 13.5)
        ax.set_ylim(71.5, -0.5)
        ax.set_xticks(range(14))
        ax.set_xlabel("OFDM symbol")
        ax.set_ylabel("Subcarrier")
        ax.set_title(f"ConcreteAE pilot locations ({budget} pilots)")
        ax.grid(True, alpha=0.25)
        fig.savefig(args.figures_dir / f"concrete_pilot_locations_{budget}.png", dpi=180)
        plt.close(fig)

    print(f"Aggregate figures for {args.model} saved in {args.figures_dir}")


if __name__ == "__main__":
    main()
