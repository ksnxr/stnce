from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .time_sampling import sample_s


def tcnce_loss_fn(
    key: jax.random.PRNGKey,
    sde,
    dsm_loss_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    score_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    log_q_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    log_q_and_score_fn: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]],
    x: ArrayLike,
    features: Optional[ArrayLike],
    t: ArrayLike,
    weight: float,
    cnce_noise_std: float,
    *,
    eps: float = 1e-5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Time-conditional NCE loss with Gaussian CNCE perturbations.

    This mirrors the ``get_reuse_main_noise(..., estimator="tcnce")`` path in
    ``st_nce_pytorch/noises_lib.py`` together with the energy-parametrized
    ``get_tcnce_loss_fn``. This loss is assumed to be used without DSM. It
    evaluates ``log_q_fn`` in the same three-call pattern used by
    ``tnce_loss_fn``. The PyTorch energy loss
    ``-2 * logsigmoid(-E(x) + E(y))`` becomes
    ``2 * softplus(log q(y) - log q(x))``.
    """
    del dsm_loss_fn, score_fn, log_q_and_score_fn

    batch_size = x.shape[0]
    s_key, noise_key, perturb_key = jax.random.split(key, 3)

    t, s = sample_s(s_key, t, eps=eps, time_std=0.0)
    ts = jnp.concatenate([t, s], axis=0)
    x0 = jnp.concatenate([x, x], axis=0)
    if features is not None:
        features_ts = jnp.concatenate([features, features], axis=0)
    else:
        features_ts = None

    mean, std = sde.marginal_prob(x0, ts)
    noise = jax.random.normal(
        noise_key, (2 * batch_size, x.shape[-1]), dtype=x.dtype
    )
    xt = mean + noise * std

    perturb_noise = jax.random.normal(
        perturb_key, (2 * batch_size, x.shape[-1]), dtype=x.dtype
    )
    xs = xt + cnce_noise_std * perturb_noise

    logp_xt_ts = log_q_fn(xt, features_ts, ts)
    logp_xt_t = logp_xt_ts[:batch_size]
    logp_xt_s = logp_xt_ts[batch_size:]

    xs_t = xs[:batch_size]
    xs_s = xs[batch_size:]
    logp_xs_t = log_q_fn(xs_t, features, t)
    logp_xs_s = log_q_fn(xs_s, features, s)

    Z1 = logp_xs_t - logp_xt_t
    Z2 = logp_xs_s - logp_xt_s

    per_sample = jax.nn.softplus(Z1) + jax.nn.softplus(Z2)
    tcnce_loss = per_sample.mean()
    return weight * tcnce_loss, jnp.array(0.0)
