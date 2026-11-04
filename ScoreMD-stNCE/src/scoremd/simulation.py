from typing import Tuple, Callable
import jax
import jax.numpy as jnp


def create_langevin_step_function(
    force: Callable[[jnp.ndarray], jnp.ndarray], mass: jnp.ndarray, gamma: float, num_steps: int, dt: float, kbT: float, with_mh: bool = False
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Implementation of this function is based on the Langevin integrator in OpenMM.
    http://docs.openmm.org/8.2.0/userguide/theory/04_integrators.html#langevinintegator

    Args:
        force: A function defining the forces acting on the system. It takes the current positions and returns the forces.
        For molecular systems the positions the potential takes are in nanometers.
        mass: The mass of the system. For molecular systems this is in atomic mass units.
        gamma: The friction coefficient, for molecular systems this is in units of inverse picoseconds. (i.e., we divide by gamma)
        num_steps: The number of steps that are performed at once.
        dt: The time step, again in ps for molecular systems.
        kbT: The thermal energy, in kJ/mol for molecular systems.
        with_mh: Whether to use Metropolis-Hastings acceptance criterion.
    Returns:
        A function that takes the current positions and velocities and maps them to new positions and velocities.
        For molecular systems all units are in nanometers, ps and kJ/mol.

    """
    assert num_steps >= 1, f"Number of steps should be at least 1. Got {num_steps}."

    def step_single(
        x: jnp.ndarray, v: jnp.ndarray, key: jax.random.PRNGKey, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Perform one plain Langevin step (no MH) as implemented in openmm."""
        assert x.shape == v.shape, f"Position and velocity should have the same shape. Shape {x.shape} != {v.shape}."
        alpha = jnp.exp(-gamma * dt)
        f_scale = (1 - alpha) / gamma
        f_x = force(x, **kwargs)
        new_v_det = alpha * v + f_scale * f_x / mass
        new_v = new_v_det + jnp.sqrt(kbT * (1 - alpha**2) / mass) * jax.random.normal(key, x.shape)
        return x + dt * new_v, new_v

    def step_with_mh(x: jnp.ndarray, v: jnp.ndarray, key: jax.random.PRNGKey, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Hamiltonian Monte Carlo (HMC): resample velocities from Maxwell-Boltzmann,
        run leapfrog integration (time-reversible, symplectic), apply MH acceptance.

        Unlike Langevin-based GHMC, leapfrog approximately conserves the model
        Hamiltonian H = U_model + KE, giving meaningful acceptance rates that
        directly reflect model energy quality:
          - Acceptance ~1: model force ≈ -∇U_model, dt is appropriate
          - Acceptance 0.5-0.9: reasonable, some discretization error
          - Acceptance <0.3: dt too large, or model force/energy inconsistent

        Requires force(x, return_energy=True) to return (force_val, energy_val).
        On rejection the momentum is negated (standard HMC detailed balance).
        """
        # Resample velocities from Maxwell-Boltzmann at each HMC step
        key, v_key = jax.random.split(key)
        v = jnp.sqrt(kbT / mass) * jax.random.normal(v_key, x.shape)

        x0, v0 = x, v

        # Initial force + energy (sum in case energy is shaped (1,), (1,1), etc.)
        f, e0 = force(x, return_energy=True, **kwargs)
        pot0 = jnp.sum(jnp.asarray(e0).reshape(-1))
        H0 = pot0 + 0.5 * jnp.sum(mass * v0**2)

        # Leapfrog integration: conserves H to O(dt^2), time-reversible
        v = v + 0.5 * dt * f / mass  # first half-step

        for _ in range(num_steps - 1):
            x = x + dt * v                  # full position step
            f = force(x, **kwargs)           # new force (energy not needed here)
            v = v + dt * f / mass            # full velocity step (two half-steps combined)

        x = x + dt * v                       # final position step

        # Final force + energy (reuses computation for last half-step and H_n)
        f, e_n = force(x, return_energy=True, **kwargs)
        v = v + 0.5 * dt * f / mass          # final half-step

        pot_n = jnp.sum(jnp.asarray(e_n).reshape(-1))
        H_n = pot_n + 0.5 * jnp.sum(mass * v**2)

        key, accept_key = jax.random.split(key)
        log_alpha = -(H_n - H0) / kbT
        log_alpha = jnp.asarray(log_alpha)
        u = jax.random.uniform(accept_key, shape=log_alpha.shape)
        accept = jnp.log(u) < log_alpha

        accept_expanded = accept
        while accept_expanded.ndim < x.ndim:
            accept_expanded = jnp.expand_dims(accept_expanded, axis=-1)

        new_x = jnp.where(accept_expanded, x, x0)
        new_v = jnp.where(accept_expanded, v, -v0)
        return new_x, new_v, jnp.where(accept, jnp.float32(1.0), jnp.float32(0.0))

    def step_no_mh(x: jnp.ndarray, v: jnp.ndarray, key: jax.random.PRNGKey, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        for _ in range(num_steps):
            key, step_key = jax.random.split(key)
            x, v = step_single(x, v, step_key, **kwargs)
        return x, v, jnp.array(1.0)

    if with_mh:
        return step_with_mh
    else:
        return step_no_mh


def simulate(
    x0: jnp.ndarray,
    v0: jnp.ndarray,
    step: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    n_steps: int,
    key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    @jax.jit
    def step_fn(carry, _):
        x, v, key = carry
        key, step_key = jax.random.split(key)
        new_x, new_v, accept = step(x, v, step_key)
        return (new_x, new_v, key), (new_x, new_v, accept)

    init_carry = (x0, v0, key)
    _, (trajectory, velocities, acceptances) = jax.lax.scan(step_fn, init_carry, None, length=n_steps)

    return trajectory, velocities, acceptances
