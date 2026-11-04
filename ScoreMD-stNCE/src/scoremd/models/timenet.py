from dataclasses import dataclass
from typing import Callable, Optional

import flax.linen as nn
import jax.numpy as jnp
from jax.typing import ArrayLike


@dataclass
class TimeNet(nn.Module):
    """Time-dependent MLP with sinusoidal time embedding."""

    dim_out: int
    activation: Callable[[ArrayLike], ArrayLike] = nn.gelu
    num_layers: int = 4
    channels: int = 64
    last_bias_init: Optional[nn.initializers.Initializer] = None
    last_weight_init: Optional[nn.initializers.Initializer] = None

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute network output from diffusion time.

        Args:
            t: Array of shape `(batch_size,)` or `(batch_size, 1)`.

        Returns:
            Array of shape `(batch_size, dim_out)`.
        """
        t = jnp.asarray(t, dtype=jnp.float32).reshape(-1, 1)

        timestep_coeff = jnp.linspace(0.1, 100.0, self.channels, dtype=jnp.float32).reshape(1, -1)
        timestep_phase = self.param(
            "timestep_phase",
            nn.initializers.normal(),
            (1, self.channels),
        )

        angles = (timestep_coeff * t) + timestep_phase
        embed_t = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=1)

        num_hidden = max(self.num_layers - 1, 1)
        for _ in range(num_hidden):
            embed_t = nn.Dense(self.channels)(embed_t)
            embed_t = self.activation(embed_t)

        out_kernel_init = self.last_weight_init or nn.initializers.lecun_normal()
        out_bias_init = self.last_bias_init or nn.initializers.zeros
        return nn.Dense(
            self.dim_out,
            kernel_init=out_kernel_init,
            bias_init=out_bias_init,
        )(embed_t)