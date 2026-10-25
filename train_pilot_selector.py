"""Train a ConcreteAE selector using train/validation channel splits only."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pilot_design.concrete_ae import ConcreteAutoencoder, complex_reconstruction_mse, selection_diversity_loss
from pilot_design.pilot_utils import (
    N_LOCATIONS,
    N_SYMBOLS,
    PILOT_BUDGETS,
    FullGridCandidateDataset,
    pilot_coordinates,
    selector_layout,
    validate_pilot_indices,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_discrete(model: ConcreteAutoencoder, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    selection = model.discrete_selection_matrix()
    total, count = 0.0, 0
    for noisy, target, _ in loader:
        candidates = selector_layout(noisy.to(device))
        target = selector_layout(target.to(device))
        selected = torch.einsum("km,bmc->bkc", selection, candidates)
        prediction = model.decoder(selected.flatten(1)).reshape(-1, N_LOCATIONS, 2)
        loss = complex_reconstruction_mse(prediction, target)
        total += loss.item() * noisy.shape[0]
        count += noisy.shape[0]
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilots", type=int, choices=PILOT_BUDGETS, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/matlab_epa7_24pilots"), help="Common full-channel dataset; pilot count in its manifest is not used by the selector.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--start-temperature", type=float, default=10.0)
    parser.add_argument("--min-temperature", type=float, default=0.1)
    parser.add_argument("--diversity-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = FullGridCandidateDataset(args.data_dir, "train", seed=args.seed)
    validation_data = FullGridCandidateDataset(args.data_dir, "validation", seed=args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = ConcreteAutoencoder(
        pilot_count=args.pilots,
        start_temperature=args.start_temperature,
        min_temperature=args.min_temperature,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, args.epochs * len(train_loader))
    temperature_decay = (args.min_temperature / args.start_temperature) ** (1.0 / total_steps)
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()

    print(f"device={device}; selector=ConcreteAE; pilots={args.pilots}; candidates={N_LOCATIONS}")
    print(f"train={len(train_data):,}; validation={len(validation_data):,}; test data is not loaded")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total, sample_count = 0.0, 0
        for noisy, target, _ in train_loader:
            candidates = selector_layout(noisy.to(device))
            target = selector_layout(target.to(device))
            optimizer.zero_grad(set_to_none=True)
            prediction, selections = model(candidates, stochastic=True)
            reconstruction = complex_reconstruction_mse(prediction, target)
            loss = reconstruction + args.diversity_weight * selection_diversity_loss(selections)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.temperature.mul_(temperature_decay).clamp_(min=args.min_temperature)
            train_total += reconstruction.item() * noisy.shape[0]
            sample_count += noisy.shape[0]

        train_mse = train_total / sample_count
        validation_mse = evaluate_discrete(model, validation_loader, device)
        print(
            f"epoch {epoch:03d}/{args.epochs}: train_mse={train_mse:.6g} "
            f"validation_discrete_mse={validation_mse:.6g} "
            f"temperature={model.temperature.item():.4g}"
        )
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping after {args.patience} epochs without validation improvement")
                break

    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Selector training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    indices = validate_pilot_indices(model.unique_indices(), args.pilots)
    coordinates = pilot_coordinates(indices)
    raw_indices = model.logits.detach().argmax(dim=-1).cpu().numpy()
    duplicate_count = args.pilots - len(np.unique(raw_indices))

    index_dir = args.output_dir / "pilot_indices"
    checkpoint_dir = args.output_dir / "checkpoints"
    index_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"concrete_{args.pilots}.npy"
    checkpoint_path = checkpoint_dir / f"concrete_{args.pilots}.pt"
    metadata_path = index_dir / f"concrete_{args.pilots}.json"
    np.save(index_path, indices)
    torch.save(
        {
            "state_dict": best_state,
            "pilots": args.pilots,
            "candidate_count": N_LOCATIONS,
            "flattening": "C order: index = subcarrier * 14 + OFDM_symbol",
            "seed": args.seed,
            "best_epoch": best_epoch,
            "validation_mse": best_validation,
            "training_seconds": training_seconds,
            "raw_independent_argmax_duplicate_count": int(duplicate_count),
            "selected_indices": indices,
        },
        checkpoint_path,
    )
    metadata = {
        "method": "ConcreteAE",
        "pilot_count": args.pilots,
        "selected_indices_flat_c_order": indices.tolist(),
        "coordinates_zero_based_subcarrier_symbol": coordinates.tolist(),
        "raw_independent_argmax_duplicate_count": int(duplicate_count),
        "flattening": "index = subcarrier * 14 + OFDM_symbol",
        "best_epoch": best_epoch,
        "validation_discrete_mse": best_validation,
        "training_seconds": training_seconds,
        "seed": args.seed,
        "noise_construction": "unit-QPSK transmission; Y=H*X+AWGN; LS=Y/X, with the configured sample SNR",
        "data_dir": str(args.data_dir),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"selected indices: {index_path}")
    print(f"coordinates: {metadata_path}")
    print(f"best validation discrete reconstruction MSE: {best_validation:.8g} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
