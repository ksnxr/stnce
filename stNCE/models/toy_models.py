import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import model_utils


@model_utils.register_model(name="mlp")
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.data.dim
        dims = [self.dim]
        dims.extend(
            [
                config.model.num_hidden_units
                for _ in range(config.model.num_hidden_layers)
            ]
        )
        if config.model.out_dim == "1":
            dims.append(1)
        elif config.model.out_dim == "ratio":
            dims.append(
                int(max(config.estim.num_cla_classes, config.estim.num_noise_classes))
            )
        else:
            raise NotImplementedError
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i != (len(dims) - 2):
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@model_utils.register_model(name="ebm_mlp")
class EBMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.data.dim
        self.embed_dim = config.model.embed_dim
        dims = [self.dim + self.embed_dim]
        dims.extend(
            [
                config.model.num_hidden_units
                for _ in range(config.model.num_hidden_layers)
            ]
        )
        self.num_noise_classes = config.estim.num_noise_classes
        if config.model.out_dim == "data":
            dims.append(self.dim)
        elif config.model.out_dim == "1":
            dims.append(1)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i != (len(dims) - 2):
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        if self.embed_dim > 1:
            self.time_embed = nn.Embedding(self.num_noise_classes + 1, self.embed_dim)
        else:
            self.time_embed = lambda t: t.float().unsqueeze(1)

    def forward(self, x, t):
        t_embed = self.time_embed(t)
        return self.net(torch.cat([x, t_embed], dim=1))


@model_utils.register_model(name="precond_ebm_mlp")
class PrecondEBMMLP(EBMMLP):
    def __init__(self, config):
        super().__init__(config)
        self.sigma = 1.0
        self.sq_sigma = self.sigma**2
        self.a = 0.75

    def c_skip(self, t):
        return (
            t
            * self.sq_sigma
            / (torch.square(t) * self.sq_sigma + torch.square(1.0 - t))
        )

    def forward(self, x, t):
        t_embed = self.time_embed(t)
        out = self.net(torch.cat([x, t_embed], dim=1))
        F = torch.sum(out, dim=1, keepdim=True)

        t = t.unsqueeze(dim=-1)
        cs = self.c_skip(t)
        energies = (1.0 - t * cs) / 2 / torch.square(1.0 - self.a * t) * torch.sum(
            torch.square(x), dim=1, keepdim=True
        ) - t * self.sigma / (1.0 - self.a * t) * F

        return energies


# https://github.com/facebookresearch/flow_matching/blob/main/examples/image/models/nn.py
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResBlock(nn.Module):
    def __init__(self, h_dim, widen, use_emb, emb_dim):
        super().__init__()
        self.ln = nn.LayerNorm(h_dim)
        self.h = nn.Linear(h_dim, h_dim * widen)
        self.use_emb = use_emb
        if self.use_emb:
            self.e = nn.Linear(emb_dim, h_dim * widen)
        self.i = nn.Linear(h_dim * widen, h_dim * widen)
        self.o = nn.Linear(h_dim * widen, h_dim)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, x, emb=None):
        h = F.silu(self.ln(x))
        h = self.h(h)
        if self.use_emb:
            h = h + self.e(emb)
        h = F.silu(h)
        h = self.i(h)
        h = F.silu(h)
        h = self.o(h)
        return x + h


@model_utils.register_model(name="small_resnet")
class SmallResnet(nn.Module):
    """Resnet model with optional timestep embedding."""

    def __init__(self, config):
        super().__init__()
        x_dim = config.data.dim
        n_layers = config.model.n_layers
        h_dim = config.model.h_dim
        use_emb = config.model.use_emb
        emb_dim = config.model.emb_dim
        widen = config.model.widen
        emb_type = config.model.emb_type
        if config.model.out_dim == "data":
            out_dim = x_dim
        elif config.model.out_dim == "1":
            out_dim = 1
        else:
            raise NotImplementedError

        num_noise_classes = config.estim.num_noise_classes
        if num_noise_classes == float("inf"):
            assert emb_type == "sinusoidal"
        else:
            assert emb_type == "learned"
            self.n_steps = num_noise_classes + 1
        self.n_layers = n_layers
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.use_emb = use_emb
        self.emb_dim = emb_dim
        self.widen = widen
        self.emb_type = emb_type

        if self.use_emb:
            if emb_type == "learned":
                self.embed = nn.Embedding(self.n_steps, emb_dim)
            elif emb_type == "sinusoidal":
                self.embed = lambda t: timestep_embedding(t, emb_dim)
            else:
                raise NotImplementedError(f"Unknown emb_type: {emb_type}")
        self.inp = nn.Linear(x_dim, h_dim)
        self.blocks = nn.ModuleList(
            [ResBlock(h_dim, widen, use_emb, emb_dim) for _ in range(n_layers)]
        )
        self.final = nn.Linear(h_dim, out_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        x = x.view(x.shape[0], -1).to(torch.float32)

        if self.use_emb:
            emb = self.embed(t)
        else:
            emb = None
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, emb)
        out = self.final(h)
        return out


@model_utils.register_model(name="precond_small_resnet")
class PrecondSmallResnet(SmallResnet):
    def __init__(self, config):
        super().__init__(config)
        self.sigma = 1.0
        self.sq_sigma = self.sigma**2
        self.a = 0.75

    def c_skip(self, t):
        return (
            t
            * self.sq_sigma
            / (torch.square(t) * self.sq_sigma + torch.square(1.0 - t))
        )

    def forward(self, x, t=None):
        x = x.view(x.shape[0], -1).to(torch.float32)

        if self.use_emb:
            emb = self.embed(t)
        else:
            emb = None
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, emb)
        F = self.final(h)
        # not really needed when out_dim=1
        F = torch.sum(F, dim=1, keepdim=True)

        t = t.unsqueeze(dim=-1)
        cs = self.c_skip(t)
        energies = (1.0 - t * cs) / 2 / torch.square(1.0 - self.a * t) * torch.sum(
            torch.square(x), dim=1, keepdim=True
        ) - t * self.sigma / (1.0 - self.a * t) * F

        return energies
