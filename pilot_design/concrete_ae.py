"""PyTorch Concrete Autoencoder for time-frequency pilot selection.

This is a two-component adaptation of the selector in DeepPilotDesign:
one categorical selector chooses a resource element, and real/imaginary values
at that same resource element travel together through the decoder.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


class ConcreteAutoencoder(nn.Module):
    def __init__(
        self,
        pilot_count: int,
        candidate_count: int = 72 * 14,
        channels: int = 2,
        start_temperature: float = 10.0,
        min_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if not 1 <= pilot_count <= candidate_count:
            raise ValueError("pilot_count must be between one and candidate_count")
        self.pilot_count = pilot_count
        self.candidate_count = candidate_count
        self.channels = channels
        self.min_temperature = min_temperature
        self.logits = nn.Parameter(torch.empty(pilot_count, candidate_count))
        nn.init.xavier_normal_(self.logits)
        self.register_buffer("temperature", torch.tensor(float(start_temperature)))
        decoder_input = pilot_count * channels
        decoder_output = candidate_count * channels
        # MLP sizes and LeakyReLU/dropout sequence follow the reference repo's
        # interpolation decoder, with two outputs per complex resource element.
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, 150),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(150, 320),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(320, 780),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(780, decoder_output),
        )

    def selection_matrix(self, stochastic: bool) -> torch.Tensor:
        if stochastic:
            uniform = torch.rand_like(self.logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            scores = (self.logits + gumbel) / self.temperature.clamp_min(self.min_temperature)
        else:
            scores = self.logits / self.temperature.clamp_min(self.min_temperature)
        return F.softmax(scores, dim=-1)

    def forward(self, candidates: torch.Tensor, stochastic: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if candidates.ndim != 3 or candidates.shape[1:] != (self.candidate_count, self.channels):
            raise ValueError(
                f"Expected [B,{self.candidate_count},{self.channels}], got {tuple(candidates.shape)}"
            )
        selections = self.selection_matrix(stochastic)
        selected = torch.einsum("km,bmc->bkc", selections, candidates)
        reconstructed = self.decoder(selected.flatten(1)).reshape(
            -1, self.candidate_count, self.channels
        )
        return reconstructed, selections

    @torch.no_grad()
    def unique_indices(self) -> np.ndarray:
        """Project K independent learned categorical rows to K unique points.

        The reference uses an independent argmax for each row, which can
        produce duplicate indices. Maximum-weight bipartite assignment keeps
        the learned scores while enforcing the required distinct pilot budget.
        """
        score = self.logits.detach().cpu().numpy()
        rows, columns = linear_sum_assignment(-score)
        if len(rows) != self.pilot_count:
            raise RuntimeError("Could not assign a unique candidate to every selector.")
        indices = np.empty(self.pilot_count, dtype=np.int64)
        indices[rows] = columns
        return indices

    @torch.no_grad()
    def discrete_selection_matrix(self) -> torch.Tensor:
        indices = self.unique_indices()
        selection = torch.zeros_like(self.logits)
        selection[torch.arange(self.pilot_count, device=selection.device), torch.as_tensor(indices, device=selection.device)] = 1.0
        return selection


def selection_diversity_loss(selections: torch.Tensor) -> torch.Tensor:
    """Discourage selector rows from collapsing onto the same locations."""
    gram = selections @ selections.transpose(0, 1)
    off_diagonal = gram - torch.diag_embed(torch.diagonal(gram))
    return off_diagonal.square().mean()


def complex_reconstruction_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over samples/locations of squared real plus imaginary error."""
    return (prediction - target).square().sum(dim=-1).mean()
