import torch
import torch.nn as nn


class EnergyNet(nn.Module):
    """Energy model for flow-matching-style objectives.

    The network takes a data point ``x`` together with a scalar time/corruption
    level ``t in [0, 1]`` and predicts one scalar energy per sample.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()

        layers = [nn.Linear(data_dim + 1, hidden_dim), nn.SiLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)

        if t.ndim != 1:
            raise ValueError("t must be a scalar, a batch vector, or shape (batch, 1)")
        if t.shape[0] != x.shape[0]:
            raise ValueError("x and t must have matching batch dimensions")

        xt = torch.cat([x, t.to(dtype=x.dtype, device=x.device).unsqueeze(-1)], dim=-1)
        return self.net(xt).squeeze(-1)
