"""EPA OFDM simulation and LS pilot observations used by the reproduction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import j0
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EPAConfig:
    """The published Table-I radio configuration.

    ``n_subcarriers`` is also the FFT size because the paper states 72
    subcarriers but does not specify unused guard carriers or a larger FFT.
    """
    n_subcarriers: int = 72
    n_symbols: int = 14
    subcarrier_spacing_hz: float = 15_000.0
    carrier_hz: float = 2.5e9
    speed_kmh: float = 50.0
    cp_length: int = 16
    paths: int = 7

    @property
    def sample_rate_hz(self) -> float:
        return self.n_subcarriers * self.subcarrier_spacing_hz

    @property
    def ofdm_symbol_period_s(self) -> float:
        return (self.n_subcarriers + self.cp_length) / self.sample_rate_hz

    @property
    def maximum_doppler_hz(self) -> float:
        return (self.speed_kmh / 3.6) * self.carrier_hz / 299_792_458.0


# 3GPP EPA delay profile (ns, relative power dB), as used by the paper.
EPA_DELAYS_NS = np.array([0, 30, 70, 90, 110, 190, 410], dtype=np.float64)
EPA_POWERS_DB = np.array([0, -1, -2, -3, -8, -17.2, -20.8], dtype=np.float64)


def make_pilot_grid(n_subcarriers: int, n_symbols: int, pilots: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a near-uniform rectangular pilot grid (24=6x4, 48=12x4)."""
    if pilots not in {24, 48}:
        raise ValueError("This paper evaluates 24 or 48 pilots.")
    n_freq, n_time = pilots // 4, 4
    rows = np.rint(np.linspace(0, n_subcarriers - 1, n_freq)).astype(np.int64)
    cols = np.rint(np.linspace(0, n_symbols - 1, n_time)).astype(np.int64)
    return rows, cols


def _qpsk(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    values = rng.integers(0, 4, size=shape)
    return np.exp(1j * (np.pi / 4 + values * np.pi / 2)).astype(np.complex64)


def _epa_channel(rng: np.random.Generator, config: EPAConfig) -> np.ndarray:
    """Generate one time-correlated EPA frequency response, [subcarrier, symbol].

    Every EPA tap is a complex Gaussian process with Clarke/Jakes covariance
    J0(2*pi*f_D*|delta_t|).  This is stronger than an AR approximation: all
    14 time samples have the desired Doppler covariance.
    """
    if not 1 <= config.paths <= len(EPA_DELAYS_NS):
        raise ValueError(f"paths must be in [1, {len(EPA_DELAYS_NS)}]")
    delays = EPA_DELAYS_NS[: config.paths] * 1e-9
    power = 10.0 ** (EPA_POWERS_DB[: config.paths] / 10.0)
    power /= power.sum()
    steps = np.abs(np.subtract.outer(np.arange(config.n_symbols), np.arange(config.n_symbols)))
    covariance = j0(2 * math.pi * config.maximum_doppler_hz * config.ofdm_symbol_period_s * steps)
    # The tiny diagonal term protects the Cholesky factor from round-off.
    chol = np.linalg.cholesky(covariance + 1e-12 * np.eye(config.n_symbols))
    innovations = (rng.standard_normal((config.paths, config.n_symbols)) + 1j * rng.standard_normal((config.paths, config.n_symbols))) / math.sqrt(2)
    gains = innovations @ chol.T
    gains *= np.sqrt(power)[:, None]
    frequencies = (np.arange(config.n_subcarriers) - config.n_subcarriers // 2) * config.subcarrier_spacing_hz
    steering = np.exp(-2j * np.pi * frequencies[:, None] * delays[None, :])
    return (steering @ gains).astype(np.complex64)


class OFDMDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Deterministic synthetic (noisy LS pilots, true EPA channel) samples."""

    def __init__(self, size: int, pilots: int, snr_db: float | tuple[float, float] = (0.0, 20.0), seed: int = 0, config: EPAConfig = EPAConfig()) -> None:
        self.size, self.pilots, self.snr_db, self.seed, self.config = size, pilots, snr_db, seed, config
        self.rows, self.cols = make_pilot_grid(config.n_subcarriers, config.n_symbols, pilots)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        channel = _epa_channel(rng, self.config)
        pilots = _qpsk(rng, (len(self.rows), len(self.cols)))
        if isinstance(self.snr_db, tuple):
            snr_db = rng.uniform(*self.snr_db)
        else:
            snr_db = self.snr_db
        noise_std = math.sqrt(10.0 ** (-snr_db / 10.0))
        noise = noise_std * (rng.standard_normal(pilots.shape) + 1j * rng.standard_normal(pilots.shape)) / math.sqrt(2)
        received = channel[np.ix_(self.rows, self.cols)] * pilots + noise
        ls = received / pilots
        x = np.stack((ls.real, ls.imag), axis=0).astype(np.float32)
        y = np.stack((channel.real, channel.imag), axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


class SavedOFDMDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A read-only memory-mapped split written by :mod:`simulate_data`.

    Keeping arrays memory-mapped means the 20,000-sample paper dataset is not
    duplicated in RAM during training.  Samples are copied only when handed to
    PyTorch, which also prevents workers from mutating mapped storage.
    """

    SPLITS = frozenset({"train", "validation", "test"})

    def __init__(self, data_dir: str | Path, split: str) -> None:
        self.data_dir = Path(data_dir)
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {sorted(self.SPLITS)}")
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.split = split
        self.x = np.load(self.data_dir / f"{split}_x.npy", mmap_mode="r")
        self.y = np.load(self.data_dir / f"{split}_y.npy", mmap_mode="r")
        self.snr_db = np.load(self.data_dir / f"{split}_snr_db.npy", mmap_mode="r")
        if len(self.x) != len(self.y) or len(self.x) != len(self.snr_db):
            raise ValueError(f"Inconsistent sample counts in {self.data_dir} {split} split")
        if self.x.ndim != 4 or self.y.ndim != 4 or self.x.shape[1] != 2 or self.y.shape[1:] != (2, 72, 14):
            raise ValueError("Dataset arrays do not have expected [N,2,Pf,Pt] and [N,2,72,14] shapes")

    @property
    def pilots(self) -> int:
        return int(self.manifest["pilot_configuration"]["count"])

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(np.array(self.x[index], copy=True)), torch.from_numpy(np.array(self.y[index], copy=True))
