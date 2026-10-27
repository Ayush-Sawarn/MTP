"""Plot fresh FC estimator comparisons and the evaluated pilot locations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pilot_design.pilot_utils import PILOT_BUDGETS, uniform_pilot_indices, validate_pilot_indices


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/reproducibility_all_budgets.csv"))
    parser.add_argument("--indices-dir", type=Path, default=Path("results/pilot_indices"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--joint16-indices", type=Path, default=Path("results/candidate_joint_16/pilot_indices.npy"))
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in read_rows(args.metrics)
            if r.get("model") == "AttRNet-FC" and r.get("pilot_method") in {"uniform", "joint_frozen"}]
    if not rows:
        parser.error(f"No Uniform/joint_frozen FC metrics found in {args.metrics}")

    # For each available budget show seed means and sample-standard-deviation bands.
    available_budgets = sorted({int(r["pilot_count"]) for r in rows})
    for budget in available_budgets:
        budget_rows = [r for r in rows if int(r["pilot_count"]) == budget]
        fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
        styles = {"uniform": ("Uniform pilots", "#2878B5"),
                  "joint_frozen": ("Joint learned pilots + fresh FC AttRNet", "#D95319")}
        for method, (label, color) in styles.items():
            selected = [r for r in budget_rows if r["pilot_method"] == method]
            snrs = sorted({float(r["snr_db"]) for r in selected})
            means, stds = [], []
            for snr in snrs:
                vals = [float(r["nmse"]) for r in selected if float(r["snr_db"]) == snr]
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            if snrs:
                ax.errorbar(snrs, means, yerr=stds, marker="o", capsize=3,
                            linewidth=2, color=color, label=label)
        ax.set_yscale("log")
        ax.set_xticks(sorted({float(r["snr_db"]) for r in budget_rows}))
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("NMSE (mean across test samples)")
        ax.set_title(f"{budget} pilots: Uniform vs joint mask with freshly trained FC AttRNet")
        ax.grid(True, which="both", linestyle=":", alpha=0.65)
        ax.legend()
        fig.savefig(args.figures_dir / f"nmse_snr_{budget}_uniform_vs_joint_fresh_fc.png", dpi=180)
        plt.close(fig)

    # A single aggregate plot over budgets for which both methods have test results.
    paired_budgets = sorted({int(r["pilot_count"]) for r in rows
        if any(int(q["pilot_count"]) == int(r["pilot_count"]) and q["pilot_method"] != r["pilot_method"] for q in rows)})
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    styles = {"uniform": ("Uniform", "#2878B5"), "joint_frozen": ("Joint learned + fresh FC", "#D95319")}
    plotted = False
    for method, (label, color) in styles.items():
        xs, ys, es = [], [], []
        for budget in paired_budgets:
            by_seed: dict[str, list[float]] = {}
            for row in rows:
                if int(row["pilot_count"]) == budget and row["pilot_method"] == method:
                    by_seed.setdefault(row.get("seed", "0"), []).append(float(row["nmse"]))
            seed_means = [float(np.mean(values)) for values in by_seed.values()]
            if seed_means:
                xs.append(budget)
                ys.append(float(np.mean(seed_means)))
                es.append(float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, linewidth=2, color=color, label=label)
            plotted = True
    if not plotted:
        plt.close(fig)
        parser.error("No pilot budget has both Uniform and joint_frozen FC test results")
    ax.set_xticks(paired_budgets)
    ax.set_xlabel("Number of pilots")
    ax.set_ylabel("Mean NMSE across SNRs (mean ± seed standard deviation)")
    ax.set_title("Aggregate FC AttRNet performance by pilot budget")
    ax.grid(True, which="both", linestyle=":", alpha=0.65)
    ax.legend()
    fig.savefig(args.figures_dir / "aggregate_nmse_pilot_count_uniform_vs_joint_fresh_fc.png", dpi=180)
    plt.close(fig)

    # Pilot maps are emitted only when a learned mask exists; this avoids
    # presenting a missing/partial selector run as a completed result.
    for budget in PILOT_BUDGETS:
        uniform_path = args.indices_dir / f"uniform_{budget}.npy"
        joint_path = args.indices_dir / f"joint_{budget}.npy"
        if budget == 16 and args.joint16_indices.exists():
            joint_path = args.joint16_indices
        if not joint_path.exists():
            continue
        uniform = validate_pilot_indices(np.load(uniform_path, allow_pickle=False), budget)
        joint = validate_pilot_indices(np.load(joint_path, allow_pickle=False), budget)
        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        ax.scatter(uniform % 14, uniform // 14, marker="s", s=58, facecolors="none",
                   edgecolors="#2878B5", linewidths=1.7, label="Uniform")
        ax.scatter(joint % 14, joint // 14, marker="x", s=70,
                   color="#D95319", linewidths=2.0, label="Joint learned")
        ax.set_xlim(-0.5, 13.5)
        ax.set_ylim(71.5, -0.5)
        ax.set_xticks(range(14))
        ax.set_xlabel("OFDM symbol")
        ax.set_ylabel("Subcarrier")
        ax.set_title(f"Pilot locations on 72 × 14 grid ({budget} pilots)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.savefig(args.figures_dir / f"pilot_locations_uniform_vs_joint_{budget}.png", dpi=180)
        plt.close(fig)

    print(f"Final comparison figures saved to {args.figures_dir}; paired budgets: {paired_budgets}")


if __name__ == "__main__":
    main()
