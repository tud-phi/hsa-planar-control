from jax import Array, debug, jacfwd, jit
import jax.numpy as jnp
from typing import Callable, Tuple


def linearize_actuation(
    q_eq: Array, phi_eq: Array, dynamical_matrices_fn: Callable
) -> Tuple[Array, Array]:
    """
    Linearize the actuation vector.
    Args:
        q_eq: configuration vector of shape (n_q, )
        phi_eq: current motor positions vector of shape (n_phi, )
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha.
    Returns:
        tau_eq: actuation vector of shape (n_q, )
        A: linearized actuation vector of shape (n_q, n_phi)
    """

    def alpha_fn(_phi: Array) -> Array:
        _, _, _, _, _, _alpha = dynamical_matrices_fn(q_eq, jnp.zeros_like(q_eq), _phi)
        return _alpha

    tau_eq = alpha_fn(phi_eq)  # torque at which the system is linearized
    # linearized actuation vector
    A = jacfwd(alpha_fn)(phi_eq)

    return tau_eq, A


def map_configuration_space_torque_to_twist_angle(
    q_eq: Array, phi_eq: Array, dynamical_matrices_fn: Callable, tau_q_des: Array
) -> Array:
    """
    Map a desired torque in configuration space to a desired twist angle by linearizing the actuation vector.
    Args:
        q_eq: configuration vector of shape (n_q, )
        phi_eq: current motor positions vector of shape (n_phi, )
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha.
        tau_q_des: desired torque in configuration space of shape (n_q, )
    Returns:
        phi_des: desired motor positions (n_phi, )
    """

    tau_eq, A = linearize_actuation(q_eq, phi_eq, dynamical_matrices_fn)

    # compute the desired phi
    phi_des = phi_eq + jnp.linalg.pinv(A) @ (tau_q_des - tau_eq)

    return phi_des
