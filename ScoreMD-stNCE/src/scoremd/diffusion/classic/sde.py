"""SDE class taken and adapted from diffusionjax library.
https://github.com/bb515/diffusionjax/blob/24fb2c8ee0ca85f618caf797c24614d2ee686be4/diffusionjax/sde.py
"""

import jax
import jax.numpy as jnp
from .utils import (
    batch_mul,
    get_exponential_sigma_function,
    get_linear_beta_function,
)


class RSDE:
    """Reverse SDE class."""

    def __init__(self, score, forward_sde, logp=None, logp_and_score=None):
        self.score = score
        self.forward_sde = forward_sde
        self.logp = logp
        self.logp_and_score = logp_and_score

    def sde(self, x, features, t):
        drift, diffusion = self.forward_sde(x, features, t)
        drift = -drift + batch_mul(diffusion**2, self.score(x, features, t))
        return drift, diffusion


class VE:
    """Variance exploding (VE) SDE, a.k.a. diffusion process with a time dependent diffusion coefficient."""

    def __init__(self, sigma=None, sigma_min=0.01, sigma_max=50.0):
        if sigma is None:
            self.sigma = get_exponential_sigma_function(
                sigma_min=sigma_min, sigma_max=sigma_max
            )
        else:
            self.sigma = sigma
        self.sigma_min = self.sigma(0.0)
        self.sigma_max = self.sigma(1.0)

    def sde(self, x, features, t):
        drift = jnp.zeros_like(x)
        diffusion = self.sigma(t) * jnp.sqrt(
            2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min))
        )

        return drift, diffusion

    def log_mean_coeff(self, t):
        return jnp.zeros_like(t)

    def std(self, t):
        return self.sigma(t)

    def variance(self, t):
        return self.sigma(t) ** 2

    def marginal_prob(self, x, t):
        return x, self.std(t)

    def prior(self, rng, shape):
        return jax.random.normal(rng, shape) * self.sigma_max

    def reverse(self, score, logp=None, logp_and_score=None):
        return RVE(
            score, self.sde, self.sigma, logp=logp, logp_and_score=logp_and_score
        )


class BetaVE:
    """
    This is a weird mix between VE and VP. It has a time dependent diffusion coefficient, but the variance is constant.
    We keep this for legacy reasons, because it was used in our initial experiments, but cannot be used to sample.
    Somehow the potential score model is able to learn the correct score for this SDE at t=0.
    """

    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta, self.log_mean_coeff = get_linear_beta_function(
            beta_min=beta_min, beta_max=beta_max
        )

    def sde(self, x, features, t):
        drift = jnp.zeros_like(x)
        diffusion = jnp.sqrt(self.beta(t))
        return drift, diffusion

    def marginal_prob(self, x, t):
        return x, self.std(t)

    def std(self, t):
        return jnp.sqrt(self.variance(t))

    def variance(self, t):
        return -jnp.expm1(self.log_mean_coeff(t) * 2)


class VP:
    """Variance preserving (VP) SDE, a.k.a. time rescaled Ohrnstein Uhlenbeck (OU) SDE."""

    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta, self.log_mean_coeff = get_linear_beta_function(
            beta_min=beta_min, beta_max=beta_max
        )

    def sde(self, x, features, t):
        beta_t = self.beta(t)
        diffusion = jnp.sqrt(beta_t)
        drift = -0.5 * batch_mul(beta_t, x)
        return drift, diffusion

    def std(self, t):
        return jnp.sqrt(self.variance(t))

    def variance(self, t):
        return -jnp.expm1(self.log_mean_coeff(t) * 2)

    def marginal_prob(self, x, t):
        return batch_mul(jnp.exp(self.log_mean_coeff(t)), x), self.std(t)

    def prior(self, rng, shape):
        return jax.random.normal(rng, shape)

    def forward_transition_params(self, xs, s, delta_t):
        """
        Computes the mean and standard deviation of the forward transition p(x_t | x_s).
        Assumes delta_t > 0, where t = s + delta_t.
        """
        t = s + delta_t
        log_mean_coeff_diff = self.log_mean_coeff(t) - self.log_mean_coeff(s)

        mean = batch_mul(jnp.exp(log_mean_coeff_diff), xs)
        std = jnp.sqrt(-jnp.expm1(2.0 * log_mean_coeff_diff))
        return mean, std

    def backward_transition_params(self, xt, t, delta_t, score_xt):
        """
        Reverse transition through Exponential Integrator
        """
        s = t - delta_t

        log_alpha_ratio = self.log_mean_coeff(s) - self.log_mean_coeff(t)
        alpha_ratio = jnp.exp(log_alpha_ratio)
        std_t = self.std(t)
        std_s = self.std(s)

        # alpha_ratio**2 * std_t**2 - std_s**2 simplifies to alpha_ratio**2 - 1.
        # prevent 0 variance
        variance = jnp.maximum(jnp.expm1(2.0 * log_alpha_ratio), 1e-5)
        denom = alpha_ratio * std_t + std_s
        denom = jnp.where(denom > 0.0, denom, 1.0)
        score_coeff = 2.0 * std_t * variance / denom

        mean = batch_mul(alpha_ratio, xt) + batch_mul(score_coeff, score_xt)

        std = jnp.sqrt(variance)

        return mean, std

    def reverse(self, score, logp=None, logp_and_score=None):
        return RVP(
            score,
            self.sde,
            self.beta,
            self.log_mean_coeff,
            logp=logp,
            logp_and_score=logp_and_score,
        )


class RVE(RSDE, VE):
    def __init__(self, score, forward_sde, sigma, *, logp=None, logp_and_score=None):
        RSDE.__init__(
            self, score, forward_sde, logp=logp, logp_and_score=logp_and_score
        )
        VE.__init__(self, sigma=sigma)

    def get_estimate_x_0_vmap(self, observation_map):
        """
        Get a function returning the MMSE estimate of x_0|x_t.

        Args:
          observation_map: function that operates on unbatched x.
          shape: optional tuple that reshapes x so that it can be operated on.
        """

        def estimate_x_0(x, t):
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            v_t = self.variance(t)
            s = self.score(x, t)
            x_0 = x + v_t * s
            return observation_map(x_0), (s, x_0)

        return estimate_x_0

    def get_estimate_x_0(self, observation_map, shape=None):
        """
        Get a function returning the MMSE estimate of x_0|x_t.

        Args:
          observation_map: function that operates on unbatched x.
          shape: optional tuple that reshapes x so that it can be operated on.
        """
        batch_observation_map = jax.vmap(observation_map)

        def estimate_x_0(x, t):
            v_t = self.variance(t)
            s = self.score(x, t)
            x_0 = x + batch_mul(v_t, s)
            if shape:
                return batch_observation_map(x_0.reshape(shape)), (s, x_0)
            else:
                return batch_observation_map(x_0), (s, x_0)

        return estimate_x_0

    def guide(self, get_guidance_score, observation_map, *args, **kwargs):
        guidance_score = get_guidance_score(self, observation_map, *args, **kwargs)
        return RVE(guidance_score, self.forward_sde, self.sigma)

    def correct(self, corrector):
        class CVE(RVE):
            def sde(x, t):
                return corrector(self.score, x, t)

        return CVE(self.score, self.forward_sde, self.sigma)


class RVP(RSDE, VP):
    def __init__(
        self,
        score,
        forward_sde,
        beta,
        log_mean_coeff,
        *,
        logp=None,
        logp_and_score=None,
    ):
        RSDE.__init__(
            self, score, forward_sde, logp=logp, logp_and_score=logp_and_score
        )
        self.beta = beta
        self.log_mean_coeff = log_mean_coeff

    def get_estimate_x_0_vmap(self, observation_map):
        """
        Get a function returning the MMSE estimate of x_0|x_t.

        Args:
          observation_map: function that operates on unbatched x.
          shape: optional tuple that reshapes x so that it can be operated on.
        """

        def estimate_x_0(x, t):
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            m_t = jnp.exp(self.log_mean_coeff(t))
            v_t = self.variance(t)
            s = self.score(x, t)
            x_0 = (x + v_t * s) / m_t
            return observation_map(x_0), (s, x_0)

        return estimate_x_0

    def get_estimate_x_0(self, observation_map, shape=None):
        """
        Get a function returning the MMSE estimate of x_0|x_t.

        Args:
          observation_map: function that operates on unbatched x.
          shape: optional tuple that reshapes x so that it can be operated on.
        """
        batch_observation_map = jax.vmap(observation_map)

        def estimate_x_0(x, features, t):
            m_t = jnp.exp(self.log_mean_coeff(t))
            v_t = self.variance(t)
            s = self.score(x, features, t)
            x_0 = batch_mul(x + batch_mul(v_t, s), 1.0 / m_t)
            if shape:
                return batch_observation_map(x_0.reshape(shape)), (s, x_0)
            else:
                return batch_observation_map(x_0), (s, x_0)

        return estimate_x_0

    def guide(self, get_guidance_score, observation_map, *args, **kwargs):
        guidance_score = get_guidance_score(self, observation_map, *args, **kwargs)
        return RVP(guidance_score, self.forward_sde, self.beta, self.log_mean_coeff)

    def correct(self, corrector):
        class CVP(RVP):
            def sde(x, features, t):
                return corrector(self.score, x, features, t)

        return CVP(self.score, self.forward_sde, self.beta, self.log_mean_coeff)
