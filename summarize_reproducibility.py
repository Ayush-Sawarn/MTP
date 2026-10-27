"""Summarize repeated-seed per-SNR NMSE results with sample standard deviation."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/reproducibility_16pilots.csv"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    grouped: dict[tuple[str, int, float], list[tuple[int, float]]] = defaultdict(list)
    with args.metrics.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("seed") in (None, ""):
                continue
            grouped[(row["pilot_method"], int(row["pilot_count"]), float(row["snr_db"]))].append(
                (int(row["seed"]), float(row["nmse"]))
            )

    summary_rows: list[dict[str, object]] = []
    for (method, budget, snr), values in sorted(grouped.items()):
        seeds, nmse = zip(*sorted(values))
        summary_rows.append({
            "pilot_method": method,
            "pilot_count": budget,
            "snr_db": snr,
            "seed_count": len(seeds),
            "seeds": ";".join(str(seed) for seed in seeds),
            "mean_nmse": float(np.mean(nmse)),
            "std_nmse_sample": float(np.std(nmse, ddof=1)) if len(nmse) > 1 else float("nan"),
        })

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]) if summary_rows else [])
            writer.writeheader()
            writer.writerows(summary_rows)

    print("method,pilot_count,snr_db,seeds,mean_nmse,std_nmse_sample")
    for row in summary_rows:
        print(
            f"{row['pilot_method']},{row['pilot_count']},{row['snr_db']:g},{row['seeds']},"
            f"{row['mean_nmse']:.8g},{row['std_nmse_sample']:.8g}"
        )


if __name__ == "__main__":
    main()
