"""Train unchanged AttRNet on a fixed learned or uniform pilot pattern."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from attrnet_ofdm.model import AttRNet, parameter_count
from pilot_design.pilot_utils import (
    PILOT_BUDGETS,
    PilotPatternDataset,
    uniform_pilot_indices,
    validate_pilot_indices,
)


@torch.no_grad()
def evaluate_mse(model: AttRNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y, _ in loader:
        err = (model(x.to(device)) - y.to(device)).square().sum(dim=1).mean()
        total += err.item() * x.shape[0]
        count += x.shape[0]
    return total / count


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["fc", "conv"], default="fc")
    parser.add_argument("--method", choices=["uniform", "concrete"], required=True)
    parser.add_argument("--pilots", type=int, choices=PILOT_BUDGETS, required=True)
    parser.add_argument("--pilot-indices", type=Path, default=None, help="Required for ConcreteAE; uniform indices are generated automatically.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/matlab_epa7_24pilots"), help="Common target/split dataset; its manifest pilot count is not used.")
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--output-kernel", type=int, nargs=2, default=(36, 7))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.method == "uniform":
        indices = uniform_pilot_indices(args.pilots)
        index_path = Path("results/pilot_indices") / f"uniform_{args.pilots}.npy"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(index_path, indices)
    else:
        index_path = args.pilot_indices or Path("results/pilot_indices") / f"concrete_{args.pilots}.npy"
        if not index_path.is_file():
            parser.error(f"ConcreteAE indices not found: {index_path}. Train the selector first.")
        indices = np.load(index_path, allow_pickle=False)
    indices = validate_pilot_indices(indices, args.pilots)

    output = args.output or Path("runs/pilot_design") / f"{args.model}_{args.method}_{args.pilots}"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = PilotPatternDataset(args.data_dir, "train", indices, args.pilots, seed=args.seed)
    validation_data = PilotPatternDataset(args.data_dir, "validation", indices, args.pilots, seed=args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = AttRNet(args.model, args.features, output_kernel=tuple(args.output_kernel)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best = float("inf")
    training_started = time.perf_counter()
    print(
        f"device={device}; model=AttRNet-{args.model.upper()}; method={args.method}; "
        f"pilots={args.pilots}; parameters={parameter_count(model):,}; input=(2,72,14)"
    )
    print(f"train={len(train_data):,}; validation={len(validation_data):,}; test data is not loaded during training")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x) - y).square().sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * x.shape[0]
            count += x.shape[0]
        train_mse = total / count
        validation_mse = evaluate_mse(model, validation_loader, device)
        print(f"epoch {epoch:03d}/{args.epochs}: train_mse={train_mse:.6g} validation_mse={validation_mse:.6g}")
        if validation_mse < best:
            best = validation_mse
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model": args.model,
                    "features": args.features,
                    "output_kernel": tuple(args.output_kernel),
                    "pilots": args.pilots,
                    "pilot_method": args.method,
                    "pilot_indices": indices,
                    "pilot_indices_path": str(index_path) if index_path else None,
                    "input_representation": "zero-filled 2-channel 72x14 LS grid",
                    "val_mse": validation_mse,
                    "epoch": epoch,
                    "seed": args.seed,
                },
                output / "best.pt",
            )
    training_seconds = time.perf_counter() - training_started
    config = vars(args).copy()
    config["pilot_indices_path"] = str(index_path) if index_path else None
    config["pilot_indices"] = indices.tolist()
    config["pilot_coordinates_zero_based_subcarrier_symbol"] = np.stack(
        (indices // 14, indices % 14), axis=1
    ).tolist()
    config["parameter_count"] = parameter_count(model)
    config["training_seconds"] = training_seconds
    config["input_representation"] = "zero-filled 2-channel 72x14 LS grid"
    (output / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    print(f"best validation MSE: {best:.8g}; checkpoint: {output / 'best.pt'}")
    print(f"training_seconds: {training_seconds:.3f}")


if __name__ == "__main__":
    main()
