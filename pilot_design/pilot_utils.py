"""Pilot-index conventions and sparse observation datasets.

Flattening is C order over ``[subcarrier, OFDM symbol]``: index
``k * 14 + n`` refers to subcarrier ``k`` and symbol ``n``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from attrnet_ofdm.data import SavedOFDMDataset, make_pilot_grid


N_SUBCARRIERS = 72
N_SYMBOLS = 14
N_LOCATIONS = N_SUBCARRIERS * N_SYMBOLS
PILOT_BUDGETS = (48, 24, 16, 12)


def validate_pilot_indices(indices: np.ndarray, budget: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len(indices) != budget:
        raise ValueError(f"Expected exactly {budget} pilot indices; received {len(indices)}.")
    if len(np.unique(indices)) != budget:
        raise ValueError("Pilot indices must be unique.")
    if np.any(indices < 0) or np.any(indices >= N_LOCATIONS):
        raise ValueError(f"Pilot indices must be in [0, {N_LOCATIONS - 1}].")
    return indices


def uniform_pilot_indices(budget: int) -> np.ndarray:
    if budget not in PILOT_BUDGETS:
        raise ValueError(f"Supported pilot budgets are {PILOT_BUDGETS}.")
    rows, cols = make_pilot_grid(N_SUBCARRIERS, N_SYMBOLS, budget)
    indices = np.asarray([row * N_SYMBOLS + col for row in rows for col in cols], dtype=np.int64)
    return validate_pilot_indices(indices, budget)


def pilot_coordinates(indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    return np.stack((indices // N_SYMBOLS, indices % N_SYMBOLS), axis=1)


class FullGridCandidateDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Generate noisy LS candidates on all grid points from saved clean channels.

    For each resource element this simulates unit-energy QPSK transmission,
    ``Y = H X + W``, then computes ``Y / X``. Thus the selector receives the
    same type of noisy LS observation as a physical pilot, at every candidate
    location. Samples are generated deterministically from split/index seeds;
    the existing saved MATLAB datasets are left untouched.
    """

    _SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}

    def __init__(self, data_dir: str | Path, split: str, seed: int = 2025) -> None:
        if split not in self._SPLIT_IDS:
            raise ValueError(f"split must be one of {tuple(self._SPLIT_IDS)}")
        self.source = SavedOFDMDataset(data_dir, split)
        self.split = split
        self.seed = seed

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = np.asarray(self.source.y[index], dtype=np.float32)
        channel = target[0] + 1j * target[1]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self._SPLIT_IDS[self.split], index]))
        qpsk = np.exp(1j * (np.pi / 4 + rng.integers(0, 4, size=(N_SUBCARRIERS, N_SYMBOLS)) * np.pi / 2))
        snr_db = float(self.source.snr_db[index])
        noise_sigma = np.sqrt(10.0 ** (-snr_db / 10.0) / 2.0)
        noise = noise_sigma * (
            rng.standard_normal((N_SUBCARRIERS, N_SYMBOLS))
            + 1j * rng.standard_normal((N_SUBCARRIERS, N_SYMBOLS))
        )
        noisy_ls = (channel * qpsk + noise) / qpsk
        inputs = np.stack((noisy_ls.real, noisy_ls.imag), axis=0).astype(np.float32)
        return (
            torch.from_numpy(inputs),
            torch.from_numpy(np.array(target, copy=True)),
            torch.tensor(snr_db, dtype=torch.float32),
        )


class PilotPatternDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Full-grid, zero-filled LS input for one fixed pilot pattern."""

    def __init__(self, data_dir: str | Path, split: str, indices: np.ndarray, budget: int, seed: int = 2025) -> None:
        self.candidates = FullGridCandidateDataset(data_dir, split, seed)
        self.indices = validate_pilot_indices(indices, budget)
        self.pilots = budget
        self.snr_db = self.candidates.source.snr_db

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_ls, target, snr = self.candidates[index]
        sparse = torch.zeros_like(full_ls)
        flat_full = full_ls.reshape(2, N_LOCATIONS)
        flat_sparse = sparse.reshape(2, N_LOCATIONS)
        pilot_tensor_indices = torch.as_tensor(self.indices, dtype=torch.long)
        flat_sparse[:, pilot_tensor_indices] = flat_full[:, pilot_tensor_indices]
        return sparse, target, snr


def selector_layout(batch: torch.Tensor) -> torch.Tensor:
    """Convert ``[B,2,72,14]`` real/imag maps to ``[B,1008,2]``."""
    if batch.ndim != 4 or batch.shape[1:] != (2, N_SUBCARRIERS, N_SYMBOLS):
        raise ValueError(f"Expected [B,2,{N_SUBCARRIERS},{N_SYMBOLS}], got {tuple(batch.shape)}")
    return batch.permute(0, 2, 3, 1).reshape(batch.shape[0], N_LOCATIONS, 2)
