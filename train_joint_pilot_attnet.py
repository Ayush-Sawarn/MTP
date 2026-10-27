"""Jointly learn a fixed-budget pilot mask and FC AttRNet.

The forward pass always uses exactly K unique pilots. During backpropagation,
a bounded sigmoid relaxation approximates top-K selection; its threshold is
found by bisection so the soft mask has approximately K total mass and every
entry lies in [0, 1]. Checkpoints are selected using validation NMSE.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from attrnet_ofdm.model import AttRNet, parameter_count
from pilot_design.pilot_utils import (
    N_LOCATIONS,
    PILOT_BUDGETS,
    FullGridCandidateDataset,
    pilot_coordinates,
    validate_pilot_indices,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StraightThroughTopK(nn.Module):
    """Hard top-K forward mask with a bounded sigmoid top-K relaxation."""

    def __init__(self, candidate_count: int, pilot_count: int, initial_indices: np.ndarray) -> None:
        super().__init__()
        initial_indices = validate_pilot_indices(initial_indices, pilot_count)
        self.candidate_count = candidate_count
        self.pilot_count = pilot_count
        self.logits = nn.Parameter(torch.zeros(candidate_count, dtype=torch.float32))
        with torch.no_grad():
            # Start at the ConcreteAE solution, but allow scores to move freely.
            self.logits[torch.as_tensor(initial_indices, dtype=torch.long)] = 1.0

    @torch.no_grad()
    def _threshold(self, temperature: float) -> torch.Tensor:
        """Find a detached threshold whose sigmoid mask sums to approximately K."""
        scaled = self.logits / max(float(temperature), 1e-6)
        low = scaled.min() - 30.0
        high = scaled.max() + 30.0
        target = float(self.pilot_count)
        for _ in range(60):
            mid = (low + high) / 2.0
            mass = torch.sigmoid(scaled - mid).sum()
            # Increasing threshold decreases the total mask mass.
            if mass > target:
                low = mid
            else:
                high = mid
        return ((low + high) / 2.0).detach()

    def _soft_topk(self, temperature: float) -> torch.Tensor:
        temp = max(float(temperature), 1e-6)
        scaled_logits = self.logits / temp
        threshold = self._threshold(temp)
        # Entries are bounded in [0, 1]; total soft mass is approximately K.
        return torch.sigmoid(scaled_logits - threshold)

    def forward(self, temperature: float) -> torch.Tensor:
        chosen = torch.topk(self.logits, self.pilot_count, sorted=False).indices
        hard = torch.zeros_like(self.logits)
        hard[chosen] = 1.0
        soft = self._soft_topk(temperature)
        # Forward value is exactly hard; backward gradient comes from soft.
        return hard - soft.detach() + soft

    @torch.no_grad()
    def indices(self) -> np.ndarray:
        return self.logits.topk(self.pilot_count, sorted=True).indices.cpu().numpy().astype(np.int64)


@torch.no_grad()
def evaluate(
    estimator: AttRNet,
    selector: StraightThroughTopK,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Return (mean per-sample NMSE, MSE) using the discrete pilot mask."""
    estimator.eval()
    selector.eval()
    mask = torch.zeros(N_LOCATIONS, device=device)
    mask[torch.as_tensor(selector.indices(), device=device)] = 1.0
    nmse_sum, mse_sum, count = 0.0, 0.0, 0
    for full_ls, target, _ in loader:
        full_ls = full_ls.to(device)
        target = target.to(device)
        x = full_ls.reshape(-1, 2, N_LOCATIONS) * mask.view(1, 1, -1)
        prediction = estimator(x.reshape(-1, 2, 72, 14))
        error = (prediction - target).square().sum(dim=1)
        target_power = target.square().sum(dim=1)
        per_sample_nmse = error.sum(dim=(1, 2)) / (target_power.sum(dim=(1, 2)) + 1e-12)
        batch_size = full_ls.shape[0]
        nmse_sum += per_sample_nmse.sum().item()
        mse_sum += error.mean(dim=(1, 2)).sum().item()
        count += batch_size
    if count == 0:
        raise RuntimeError("Evaluation loader is empty.")
    return nmse_sum / count, mse_sum / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilots", type=int, choices=PILOT_BUDGETS, default=16)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--initial-indices", type=Path, required=True)
    parser.add_argument("--model", choices=["fc"], default="fc")
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--output-kernel", type=int, nargs=2, default=(36, 7))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AttRNet learning rate")
    parser.add_argument("--selector-learning-rate", type=float, default=3e-4,
                        help="Separate learning rate for pilot-selection logits")
    parser.add_argument("--start-temperature", type=float, default=1.0)
    parser.add_argument("--min-temperature", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.start_temperature <= args.min_temperature or args.min_temperature <= 0:
        parser.error("Require epochs >= 1 and start-temperature > min-temperature > 0.")
    if args.learning_rate <= 0 or args.selector_learning_rate <= 0:
        parser.error("Learning rates must be positive.")

    initial_indices = validate_pilot_indices(np.load(args.initial_indices, allow_pickle=False), args.pilots)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = FullGridCandidateDataset(args.data_dir, "train", seed=args.seed)
    validation_data = FullGridCandidateDataset(args.data_dir, "validation", seed=args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                   pin_memory=device.type == "cuda")

    estimator = AttRNet("fc", args.features, output_kernel=tuple(args.output_kernel)).to(device)
    selector = StraightThroughTopK(N_LOCATIONS, args.pilots, initial_indices).to(device)
    optimizer = torch.optim.Adam([
        {"params": estimator.parameters(), "lr": args.learning_rate},
        {"params": selector.parameters(), "lr": args.selector_learning_rate},
    ])
    total_steps = max(1, args.epochs * len(train_loader))
    temperature_decay = (args.min_temperature / args.start_temperature) ** (1.0 / total_steps)
    temperature = args.start_temperature
    best_validation_nmse = float("inf")
    best_validation_mse = float("inf")
    best_estimator_state = None
    best_selector_state = None
    best_epoch = 0
    started = time.perf_counter()

    print(f"device={device}; estimator=AttRNet-FC; selector=bounded sigmoid straight-through top-k; pilots={args.pilots}")
    print(f"train={len(train_data):,}; validation={len(validation_data):,}; test data is not loaded")
    print(f"learning_rates: attnet={args.learning_rate:g}, selector={args.selector_learning_rate:g}")
    for epoch in range(1, args.epochs + 1):
        estimator.train()
        selector.train()
        total_loss, sample_count = 0.0, 0
        selector_grad_norm_sum, batches = 0.0, 0
        for full_ls, target, _ in train_loader:
            full_ls, target = full_ls.to(device), target.to(device)
            mask = selector(temperature)
            x = full_ls.reshape(-1, 2, N_LOCATIONS) * mask.view(1, 1, -1)
            prediction = estimator(x.reshape(-1, 2, 72, 14))
            loss = (prediction - target).square().sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if selector.logits.grad is not None:
                selector_grad_norm_sum += selector.logits.grad.detach().norm().item()
            optimizer.step()
            total_loss += loss.item() * full_ls.shape[0]
            sample_count += full_ls.shape[0]
            batches += 1
            temperature = max(args.min_temperature, temperature * temperature_decay)

        train_mse = total_loss / max(sample_count, 1)
        validation_nmse, validation_mse = evaluate(estimator, selector, validation_loader, device)
        selected = selector.indices()
        changed = int(np.count_nonzero(~np.isin(initial_indices, selected)))
        mean_grad = selector_grad_norm_sum / max(batches, 1)
        print(
            f"epoch {epoch:03d}/{args.epochs}: train_mse={train_mse:.6g} "
            f"val_nmse={validation_nmse:.6g} val_mse={validation_mse:.6g} "
            f"temperature={temperature:.4g} changed_from_init={changed} "
            f"selector_grad_norm={mean_grad:.4g}"
        )
        if validation_nmse < best_validation_nmse:
            best_validation_nmse = validation_nmse
            best_validation_mse = validation_mse
            best_estimator_state = copy.deepcopy(estimator.state_dict())
            best_selector_state = copy.deepcopy(selector.state_dict())
            best_epoch = epoch

    training_seconds = time.perf_counter() - started
    if best_estimator_state is None or best_selector_state is None:
        raise RuntimeError("Joint training did not produce a validation checkpoint.")
    estimator.load_state_dict(best_estimator_state)
    selector.load_state_dict(best_selector_state)
    selected_indices = validate_pilot_indices(selector.indices(), args.pilots)
    output = args.output or Path(f"runs/pilot_design/fc_joint_{args.pilots}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pt"
    torch.save({
        "state_dict": best_estimator_state,
        "selector_state_dict": best_selector_state,
        "model": "fc",
        "features": args.features,
        "output_kernel": tuple(args.output_kernel),
        "pilots": args.pilots,
        "pilot_method": "joint",
        "pilot_indices": selected_indices,
        "initial_pilot_indices": initial_indices,
        "val_mse": best_validation_mse,
        "val_nmse": best_validation_nmse,
        "epoch": best_epoch,
        "seed": args.seed,
    }, checkpoint_path)
    config = vars(args).copy()
    config.update({
        "pilot_method": "joint",
        "pilot_indices": selected_indices.tolist(),
        "pilot_coordinates_zero_based_subcarrier_symbol": pilot_coordinates(selected_indices).tolist(),
        "initial_pilot_indices": initial_indices.tolist(),
        "validation_mse": best_validation_mse,
        "validation_nmse": best_validation_nmse,
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "parameter_count_attnet": parameter_count(estimator),
        "input_representation": "hard top-k forward mask with bounded sigmoid straight-through relaxation; zero-filled 2-channel 72x14 LS grid",
        "temperature_start": args.start_temperature,
        "temperature_end": temperature,
        "selector_learning_rate": args.selector_learning_rate,
    })
    (output / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    np.save(output / "pilot_indices.npy", selected_indices)
    print(f"best validation NMSE: {best_validation_nmse:.8g}; validation MSE: {best_validation_mse:.8g}; epoch {best_epoch}")
    print(f"selected pilot indices: {output / 'pilot_indices.npy'}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"training_seconds: {training_seconds:.3f}")


if __name__ == "__main__":
    main()
