from __future__ import annotations

import torch


class TrickSequenceEncoder(torch.nn.Module):
    def __init__(self, hidden_dim: int = 128, num_gru_layers: int = 1) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.proj = torch.nn.LazyLinear(hidden_dim)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, num_layers = num_gru_layers, batch_first = True)

    def forward(self, play_history: torch.Tensor, leader_history: torch.Tensor) -> torch.Tensor:
        # play_history:         (B, T, P, C)
        # leader_history.shape: (B, T, P)

        B, T, P, C = play_history.shape

        # (B, T, P * C + P)
        tricks = torch.concatenate([
            play_history.reshape(B, T, P * C),
            leader_history
        ], dim = -1)

        tricks_active = tricks.abs().sum(dim = -1) > 0          # (B, T)
        lengths = tricks_active.sum(dim = 1).clamp(min = 1)     # (B, )

        tricks = torch.relu(self.proj(tricks))                  # (B, T, hidden_dim)
        outputs, _ = self.gru(tricks)                           # (B, T, hidden_dim)

        idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.hidden_dim)
        last_output = outputs.gather(1, idx).squeeze(1)         # (B, hidden_dim)

        return last_output

