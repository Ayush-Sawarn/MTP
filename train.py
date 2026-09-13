"""Train AttRNet on the paper's EPA OFDM channel-estimation task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from attrnet_ofdm.data import SavedOFDMDataset
from attrnet_ofdm.model import AttRNet, parameter_count


def mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            err = (model(x.to(device)) - y.to(device)).square().mean()
            total += err.item() * x.size(0)
            count += x.size(0)
    return total / count


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["fc", "conv"], default="fc")
    p.add_argument("--pilots", type=int, choices=[24, 48], default=24, help="Used to select the default data directory.")
    p.add_argument("--data-dir", type=Path, default=None, help="Directory created by simulate_data.py. Defaults to data/epa7_<pilots>pilots.")
    p.add_argument("--features", type=int, default=8)
    p.add_argument("--output-kernel", type=int, nargs=2, default=(36, 7))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--workers", type=int, default=0, help="Use 0 for Windows-safe deterministic loading.")
    p.add_argument("--output", type=Path, default=Path("runs/attrnet"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir or Path("data") / f"epa7_{args.pilots}pilots"
    train_data = SavedOFDMDataset(data_dir, "train")
    val_data = SavedOFDMDataset(data_dir, "validation")
    if train_data.pilots != args.pilots:
        p.error(f"--pilots={args.pilots} does not match dataset manifest ({train_data.pilots}).")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    model = AttRNet(args.model, args.features, output_kernel=tuple(args.output_kernel)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    best = float("inf")
    print(f"device={device}; parameters={parameter_count(model):,}; data={data_dir}")
    print(f"train={len(train_data):,}; validation={len(val_data):,}; pilot input={tuple(train_data.x.shape[1:])}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
        train_mse, val_mse = total / count, mse(model, val_loader, device)
        print(f"epoch {epoch:03d}/{args.epochs}: train_mse={train_mse:.6g} val_mse={val_mse:.6g}")
        if val_mse < best:
            best = val_mse
            torch.save({"state_dict": model.state_dict(), "model": args.model, "features": args.features, "output_kernel": args.output_kernel, "pilots": args.pilots, "val_mse": val_mse}, args.output / "best.pt")
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print(f"best validation MSE: {best:.6g}; checkpoint: {args.output / 'best.pt'}")


if __name__ == "__main__":
    main()
